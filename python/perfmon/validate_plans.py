# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Job lists for the phase 1 validation (validate.py), one generator per
checklist item, written as JSON lines on stdout:

    python -m aotriton.perfmon.validate_plans llm --subjects A,B,C > jobs.jsonl

    llm       the LLM cases of workloads.csv, both ifaces, unforced and every
              forced backend
    noise     bf16 hdim 128 entries at a few seqlens, for repeated runs
    features  one entry per feature (fp16, bias bhqk and b1qk, dropout, GQA,
              sliding window), both ifaces, unforced and every backend
    aiter     bf16 hdim 128, causal and not, over the seqlen ladder (to 8192),
              unforced and forced to aiter: known issue 1 (AITER capture)
    size      the largest prime entry of a few problems, both ifaces, every
              backend

Entries come from the flash PerfDescription (modules/flash/perfmon), so a
plan needs AOTRITON_MODULES_DIR or the checkout's modules directory.
"""

import argparse
import dataclasses
import json
import sys

IFACES = ('op_attn_fwd', 'op_attn_bwd')

NOISE_SEQLENS = (16, 64, 256, 4096, 16384)

#: name -> overrides of a bf16 hdim 128 non-causal entry. Sliding windows come
#: with and without the causal mask (256 tokens).
FEATURES = {
    'fp16': {'T_io': 'fp16'},
    'bias_bhqk': {'bias_type': 1, 'bias_shape': 'bhqk'},
    'bias_b1qk': {'bias_type': 1, 'bias_shape': 'b1qk'},
    'dropout': {'dropout_p': 0.1},
    'gqa4': {'gqa_ratio': 4},
    'window_causal': {'causal': 1, 'window_left': 256, 'window_right': 0},
    'window_both': {'window_left': 256, 'window_right': 256},
}

#: Features are checked for working, not for size: a modest batch keeps the run short.
FEATURE_MAX_BATCH = 32

SIZE_HDIMS = (64, 128, 256, 512)


def make_job(subject: str, entry, iface: str, backend: str, label: str) -> dict:
    return {'subject': subject,
            'entry': dataclasses.asdict(entry),
            'iface': iface,
            'backend': backend,
            'label': label}


def llm_jobs(desc, subjects: list[str], include_pending: bool = False) -> list[dict]:
    jobs = []
    for subject in subjects:
        for case in desc.llm_cases():
            if case.pending and not include_pending:
                continue
            for iface in desc.phase_ifaces(case.phase):
                jobs.append(make_job(subject, case.entry, iface, 'all', f'llm:{case.model}'))
    return jobs


def make_entry(desc, seqlen: int, hdim: int = 128, max_batch: int | None = None, **overrides):
    """An entry of the problem (bf16, hdim, non-causal, no features) changed by
    `overrides`, with the size rule's batch and heads (batch capped at
    `max_batch` when given)."""
    problem = {'T_io': 'bf16', 'hdim_qk': hdim, 'hdim_v': hdim, 'causal': 0, 'window_left': -1,
               'window_right': -1, 'dropout_p': 0.0, 'bias_type': 0, 'bias_shape': '-', 'gqa_ratio': 1,
               'varlen': 0, 'storage_flip': 0}
    problem.update(overrides)
    sizes = desc.size(problem, seqlen, seqlen)
    if sizes is None:
        raise ValueError(f'the size rule does not fit {problem} at seqlen {seqlen}')
    batch, n_heads_q = sizes
    if max_batch is not None:
        batch = min(batch, max_batch)
    return desc.ENTRY_CLASS(**problem, seqlen_q=seqlen, seqlen_k=seqlen, batch=batch, n_heads_q=n_heads_q)


def noise_jobs(desc, subjects: list[str], seqlens=NOISE_SEQLENS) -> list[dict]:
    jobs = []
    for subject in subjects:
        for seqlen in seqlens:
            entry = make_entry(desc, seqlen)
            for iface in IFACES:
                jobs.append(make_job(subject, entry, iface, '-', f'noise:{seqlen}'))
    return jobs


def feature_jobs(desc, subjects: list[str], seqlen: int = 1024) -> list[dict]:
    jobs = []
    for subject in subjects:
        for name, overrides in FEATURES.items():
            entry = make_entry(desc, seqlen, max_batch=FEATURE_MAX_BATCH, **overrides)
            for iface in IFACES:
                jobs.append(make_job(subject, entry, iface, 'all', f'feature:{name}'))
    return jobs


def size_jobs(desc, subjects: list[str], hdims=SIZE_HDIMS) -> list[dict]:
    jobs = []
    for subject in subjects:
        for hdim in hdims:
            for causal in (0, 1):
                axes = dict(desc.entry_sets()['prime'])
                axes.update(T_io=['bf16'], hdim=[hdim], causal=[causal], dropout_p=[0.1])
                entries = list(desc.entries(axes))
                largest = max(entries, key=lambda e: e.seqlen_q)
                # every ladder point of the problem is also useful: the budget
                # fails at the largest seqlen, the cap at the smallest
                for entry in (largest, min(entries, key=lambda e: e.seqlen_q)):
                    for iface in IFACES:
                        jobs.append(make_job(subject, entry, iface, 'all', f'size:hdim{hdim}:causal{causal}'))
    return jobs


def aiter_jobs(desc, subjects: list[str], seqlens=(16, 64, 256, 1024, 4096, 8192)) -> list[dict]:
    jobs = []
    for subject in subjects:
        for causal in (0, 1):
            for seqlen in seqlens:
                entry = make_entry(desc, seqlen, causal=causal)
                for iface in IFACES:
                    for backend in ('-', 'aiter'):
                        jobs.append(make_job(subject, entry, iface, backend, f'aiter:causal{causal}'))
    return jobs


PLANS = {'llm': llm_jobs, 'noise': noise_jobs, 'features': feature_jobs, 'size': size_jobs, 'aiter': aiter_jobs}


def main(argv=None) -> int:
    from aotriton.perfmon import registry
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('plan', choices=sorted(PLANS))
    p.add_argument('--subjects', required=True, help='comma list of subject names')
    p.add_argument('--out', default='-')
    p.add_argument('--seqlens', default='', help='noise plan: comma list of seqlens')
    args = p.parse_args(argv)
    desc = registry.make_perf_desc('flash')
    options = {}
    if args.seqlens:
        if args.plan != 'noise':
            p.error('--seqlens applies to the noise plan only')
        options['seqlens'] = tuple(int(x) for x in args.seqlens.split(','))
    jobs = PLANS[args.plan](desc, [s for s in args.subjects.split(',') if s], **options)
    out = sys.stdout if args.out == '-' else open(args.out, 'w', encoding='utf-8')
    for job in jobs:
        out.write(json.dumps(job) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
