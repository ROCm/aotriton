# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The flash perfmon entry, and the prime and coverage entry sets.

An entry is one tested call: a problem (what the caller asks for, apart from
the sizes) plus every size. Its fields are exactly the wire keys of the
runner's entry PON, in the same order, so `entry_pon()` is a plain render.

Entry sets are axes, not lists: dispatch offers one option per axis and
`--entry_set` only chooses the default values. `expand()` turns axes into
entries, pairing sequence lengths by `seqlen_shape` and sizing each entry
with the size rule (size.py).

Standard library only (plus aotriton.utils.pon): this module runs at dispatch
on a control host with no GPU, torch or engine.
"""

import itertools
from dataclasses import dataclass

#: problems.csv's header: the entry fields that are not sizes.
PROBLEM_COLUMNS = ('T_io', 'hdim_qk', 'hdim_v', 'causal', 'window_left', 'window_right',
                   'dropout_p', 'bias_type', 'bias_shape', 'gqa_ratio', 'varlen', 'storage_flip')

#: Layouts of the bias, by which dimensions are real (the rest broadcast with
#: stride 0); '-' is no bias.
BIAS_SHAPES = ('-', 'bhqk', 'b1qk', '11qk')

DTYPES = ('fp16', 'bf16', 'fp32')


@dataclass(frozen=True)
class FlashPerfEntry:
    T_io: str
    hdim_qk: int
    hdim_v: int
    causal: int
    window_left: int
    window_right: int
    dropout_p: float
    bias_type: int
    bias_shape: str
    gqa_ratio: int
    varlen: int
    storage_flip: int
    seqlen_q: int
    seqlen_k: int
    batch: int
    n_heads_q: int

    def __post_init__(self):
        if self.T_io not in DTYPES:
            raise ValueError(f'T_io must be one of {DTYPES}, got {self.T_io!r}')
        if self.bias_shape not in BIAS_SHAPES:
            raise ValueError(f'bias_shape must be one of {BIAS_SHAPES}, got {self.bias_shape!r}')
        if (self.bias_type == 0) != (self.bias_shape == '-'):
            raise ValueError(f'bias_type={self.bias_type} disagrees with bias_shape={self.bias_shape!r}')
        if self.gqa_ratio < 1 or self.n_heads_q % self.gqa_ratio:
            raise ValueError(f'n_heads_q={self.n_heads_q} is not a multiple of gqa_ratio={self.gqa_ratio}')

    @property
    def n_heads_kv(self) -> int:
        return self.n_heads_q // self.gqa_ratio


# -- the prime set ---------------------------------------------------------------

PRIME_HDIMS = (16, 32, 64, 128, 192, 256, 384, 512)
# Tuning's sequence-length ladder plus 16384, one point beyond what is tuned.
SEQLEN_LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
# 0.1 is the training value of every reported regression so far.
PRIME_DROPOUT = (0.0, 0.1)


def prime_axes() -> dict:
    return {
        'T_io': ['bf16', 'fp16'],
        'hdim': list(PRIME_HDIMS),
        'causal': [0, 1],
        'dropout_p': list(PRIME_DROPOUT),
        'bias_shape': ['-'],
        'gqa_ratio': [1],
        'window_left': [-1],
        'window_right': [-1],
        'varlen': [0],
        'storage_flip': [0],
        'seqlen': list(SEQLEN_LADDER),
        'seqlen_shape': ['diagonal'],
    }


# -- the coverage set --------------------------------------------------------------

# TODO(perfmon): the coverage set is meant to be every problem HEAD's flash
# operator supports. GQA, sliding windows, varlen and storage_flip still need
# hardware validation on the adapters, so they hold their defaults here until
# phase 1 confirms them; extend the axes then.
COVERAGE_HDIMS = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512)


def coverage_axes() -> dict:
    return {
        'T_io': ['bf16', 'fp16', 'fp32'],
        'hdim': list(COVERAGE_HDIMS),
        'causal': [0, 1],
        'dropout_p': list(PRIME_DROPOUT),
        'bias_shape': ['-', 'bhqk'],
        'gqa_ratio': [1],
        'window_left': [-1],
        'window_right': [-1],
        'varlen': [0],
        'storage_flip': [0],
        'seqlen': list(SEQLEN_LADDER),
        'seqlen_shape': ['l_shape'],
    }


SEQLEN_SHAPES = ('diagonal', 'l_shape')


def seqlen_pairs(seqlens, shape: str) -> list[tuple[int, int]]:
    """(seqlen_q, seqlen_k) pairs. 'diagonal' is seqlen_q = seqlen_k;
    'l_shape' adds every seqlen_q at the largest seqlen_k and every seqlen_k
    at the largest seqlen_q (3n - 2 pairs for n seqlens)."""
    seqlens = sorted(set(seqlens))
    if shape == 'diagonal':
        return [(s, s) for s in seqlens]
    if shape == 'l_shape':
        if not seqlens:
            return []
        top = seqlens[-1]
        pairs = {(s, s) for s in seqlens} | {(s, top) for s in seqlens} | {(top, s) for s in seqlens}
        return sorted(pairs)
    raise ValueError(f'seqlen_shape must be one of {SEQLEN_SHAPES}, got {shape!r}')


def supported_problem(problem: dict) -> bool:
    """Combinations the operator does not offer at all (not an arch limit):
    causal masking together with a bias."""
    return not (problem['causal'] and problem['bias_type'])


def problems(axes: dict):
    """The problems of `axes`, as dicts of the problem columns, in a fixed
    order (the dispatch insert order follows it)."""
    names = ('T_io', 'hdim', 'causal', 'dropout_p', 'bias_shape', 'gqa_ratio',
             'window_left', 'window_right', 'varlen', 'storage_flip')
    for values in itertools.product(*(axes[name] for name in names)):
        t_io, hdim, causal, dropout_p, bias_shape, gqa, wl, wr, varlen, flip = values
        problem = {
            'T_io': t_io, 'hdim_qk': int(hdim), 'hdim_v': int(hdim), 'causal': int(causal),
            'window_left': int(wl), 'window_right': int(wr), 'dropout_p': float(dropout_p),
            'bias_type': 0 if bias_shape == '-' else 1, 'bias_shape': bias_shape,
            'gqa_ratio': int(gqa), 'varlen': int(varlen), 'storage_flip': int(flip),
        }
        if supported_problem(problem):
            yield problem


def expand(axes: dict, size_rule, dropped: list | None = None):
    """Entries of `axes`: every problem, at every seqlen pair, sized by
    `size_rule(problem, seqlen_q, seqlen_k)`. Points the rule cannot fit are
    not generated; they are appended to `dropped` when given."""
    pairs = []
    for shape in axes['seqlen_shape']:
        for pair in seqlen_pairs(axes['seqlen'], shape):
            if pair not in pairs:
                pairs.append(pair)
    for problem in problems(axes):
        for seqlen_q, seqlen_k in pairs:
            sizes = size_rule(problem, seqlen_q, seqlen_k)
            if sizes is None:
                if dropped is not None:
                    dropped.append((problem, seqlen_q, seqlen_k))
                continue
            batch, n_heads_q = sizes
            yield FlashPerfEntry(**problem,
                                 seqlen_q=seqlen_q,
                                 seqlen_k=seqlen_k,
                                 batch=batch,
                                 n_heads_q=n_heads_q)
