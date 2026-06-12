# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd_kernel_dk_dv kernel (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 12).

A KEY backward kernel (computes dK/dV). Standalone full description in the
attn_fwd_ati.py style — it is the authoritative source of the bwd operand
practices, not a citing aux kernel. Parity target:
v3python/rules/flash/bwd_kernel_dk_dv.py + op_attn_bwd.py.

Note (rev1 §3.5): the B tensor's 3rd stride is `stride_bm` in the Triton
signature but the operator/golden call it `stride_bk`. ATI emits the real name in
the (cosmetic) pp_args comment; the access expression is identical. This intended
divergence is covered by the eventual full-ATI golden re-baseline.
"""

import os
from dataclasses import dataclass

import numpy as np

import v3python.template_instantiation as ati


def _block_dmodel_values():
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',')]


MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


@dataclass
class BwdKernelDkDvPerf:
    # NUM_XCDS defaults to 1 (non-MI) and is raised to 8 on gfx942/950 by the
    # perf-channel derive below (the legacy PROGRAMMATIC_PERFS NUM_XCDS).
    BLOCK_M:  np.int16 = 16
    BLOCK_N:  np.int16 = 16
    NUM_XCDS: np.int8 = 1


import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _common_ati import flash_disabled


def _bwd_disabled(f):
    """Functionals excluded for compiler/numerical correctness. Shares the common
    flash predicate (causal+bias, gfx11 hdim>256); the bwd-specific gfx950 bad head
    dims are {48, 80}."""
    return flash_disabled(f, gfx950_bad_hdims={48, 80})


def gen_autotune_configs(f):
    """Placeholder generator (one valid config); the DB path
    (translate_dataframe) does not use it. Real heuristics ported in a later step
    (postati TODO §6, mirroring the legacy gen_autotune_configs)."""
    kw = {
        'BLOCK_M': 16, 'BLOCK_N': 16,
        'NUM_XCDS': 8 if f.arch in ('gfx942', 'gfx950') else 1,
        'waves_per_eu': 1,
    }
    yield ati.tune.Config(kw, num_warps=4, num_stages=1)


def describe_bwd_kernel_dk_dv(kernel):
    specs = [
        # --- main tensors (rank 4, last stride contiguous) ---
        ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q'),
        ati.tensor('Q',  'T_io', strides='stride_q?',  contiguous=-1),
        ati.tensor('K',  'T_io', strides='stride_k?',  contiguous=-1),
        ati.tensor('V',  'T_io', strides='stride_v?',  contiguous=-1),
        ati.tensor('B',  'T_io', strides='stride_b?',  contiguous=-1),
        ati.tensor('DO', 'T_io', strides='stride_do?', contiguous=-1),
        ati.tensor('DK', 'T_io', strides='stride_dk?', contiguous=-1),
        ati.tensor('DV', 'T_io', strides='stride_dv?', contiguous=-1),
        ati.scalar('sm_scale', 'fp32'),
        ati.tensor('L', '*fp32:16', rank=2),
        ati.tensor('D', 'LazyTensor:*fp32:16', rank=2),

        # --- problem-size scalars ---
        ati.scalar(['num_head_q', 'num_head_k', 'hdim_qk', 'hdim_vo'], 'i32'),
        ati.tensor(['cu_seqlens_q', 'cu_seqlens_k',
                    'seq_strides_q', 'seq_strides_k'], '*i32:16', rank=1),
        ati.scalar(['num_seqlens', 'max_seqlen_q', 'max_seqlen_k'], 'i32'),

        # --- dropout + PRNG ---
        ati.scalar('dropout_p', 'fp32'),
        ati.tensor(['philox_seed_ptr', 'philox_offset1'], '*u64', rank=0),
        ati.scalar('philox_offset2', 'u64'),

        # --- windowed attention ---
        ati.scalar('Window_left', 'i32'),
        ati.scalar('Window_right', 'i32'),

        # --- constexpr features ---
        ati.scalar('BLOCK_DMODEL', options=_block_dmodel_values()),
        ati.scalar('CAUSAL_TYPE', options=[0, 3]),
        ati.scalar('ENABLE_DROPOUT', options=[False, True]),
        ati.scalar('PADDED_HEAD', options=[False, True]),
        ati.scalar('BIAS_TYPE', options=[0, 1]),

        # --- performance ---
        ati.tune.schema(BwdKernelDkDvPerf),
        ati.tune.configs(gen_autotune_configs),
        ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                         max_seqlen_k=ati.tune.binning.le),
        ati.tune.fallback(PADDED_HEAD=False),

        # NUM_XCDS perf-channel derive (arch -> 8 on MI, else default 1).
        ati.derives('NUM_XCDS', to=8,
                    when=lambda f: f.arch in ('gfx942', 'gfx950')),

        # --- conditional degradation (legacy CC cases) ---
        # bias off -> B's non-unit strides become constexpr 0. Unlike fwd, the bwd
        # kernels keep B itself a live tensor pointer (legacy op_attn_bwd has only
        # the stride delete_when, no CDETensor on B).
        ati.derives(['stride_bz', 'stride_bh', 'stride_bm'], to=0,
                    when=ati.eq('BIAS_TYPE', 0)),
        # dropout off -> dropout_p / philox become constexpr 0
        ati.derives(['dropout_p', 'philox_seed_ptr', 'philox_offset1',
                     'philox_offset2'], to=0,
                    when=ati.eq('ENABLE_DROPOUT', False)),
        # non-sliding-window -> Window_left/right become constexpr 0
        ati.derives(['Window_left', 'Window_right'], to=0,
                    when=ati.ne('CAUSAL_TYPE', 3)),

        # --- functional-disable (compiler/numerical correctness exclusions) ---
        ati.disable(when=_bwd_disabled),
    ]
    ati.describe(kernel, *specs)
    return kernel
