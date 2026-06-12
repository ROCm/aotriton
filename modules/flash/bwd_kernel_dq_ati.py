# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd_kernel_dq kernel (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 12).

A KEY backward kernel (computes dQ/dB). Standalone full description, the dQ/dB
analogue of bwd_kernel_dk_dv (which does dK/dV) — same operand vocabulary and
features, with DQ/DB and their strides in place of DK/DV. Parity target:
v3python/rules/flash/bwd_kernel_dq.py + op_attn_bwd.py.

Note (rev1 §3.5): the B tensor's 3rd stride is `stride_bm` in the Triton signature
but the operator/golden call it `stride_bk`; ATI emits the real name in the
cosmetic pp_args comment (the access expression is identical).
"""

import os
from dataclasses import dataclass

import numpy as np

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

import v3python.template_instantiation as ati
from _common_ati import flash_disabled


def _block_dmodel_values():
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',')]


MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


@dataclass
class BwdKernelDqPerf:
    BLOCK_M:  np.int16 = 16
    BLOCK_N:  np.int16 = 16
    NUM_XCDS: np.int8 = 1


def _bwd_disabled(f):
    """Shared flash disable predicate; bwd gfx950-bad head dims are {48, 80}."""
    return flash_disabled(f, gfx950_bad_hdims={48, 80})


def gen_autotune_configs(f):
    """Placeholder generator (one valid config); the DB path does not use it. Real
    heuristics ported later (postati TODO §6)."""
    kw = {
        'BLOCK_M': 16, 'BLOCK_N': 16,
        'NUM_XCDS': 8 if f.arch in ('gfx942', 'gfx950') else 1,
        'waves_per_eu': 1,
    }
    yield ati.tune.Config(kw, num_warps=4, num_stages=1)


def describe_bwd_kernel_dq(kernel):
    specs = [
        # --- main tensors (rank 4, last stride contiguous) ---
        ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q'),
        ati.tensor('Q',  'T_io', strides='stride_q?',  contiguous=-1),
        ati.tensor('K',  'T_io', strides='stride_k?',  contiguous=-1),
        ati.tensor('V',  'T_io', strides='stride_v?',  contiguous=-1),
        ati.tensor('B',  'T_io', strides='stride_b?',  contiguous=-1),
        ati.tensor('DO', 'T_io', strides='stride_do?', contiguous=-1),
        ati.tensor('DQ', 'T_io', strides='stride_dq?', contiguous=-1),
        ati.tensor('DB', 'T_io', strides='stride_db?', contiguous=-1),
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
        ati.tune.schema(BwdKernelDqPerf),
        ati.tune.configs(gen_autotune_configs),
        ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                         max_seqlen_k=ati.tune.binning.le),
        ati.tune.fallback(PADDED_HEAD=False),

        ati.derives('NUM_XCDS', to=8,
                    when=lambda f: f.arch in ('gfx942', 'gfx950')),

        # --- conditional degradation ---
        # bias off -> B and DB become constexpr 0; their non-unit strides cascade.
        ati.derives('B', to=0, when=ati.eq('BIAS_TYPE', 0)),
        ati.derives('DB', to=0, when=ati.eq('BIAS_TYPE', 0)),
        ati.derives(['dropout_p', 'philox_seed_ptr', 'philox_offset1',
                     'philox_offset2'], to=0,
                    when=ati.eq('ENABLE_DROPOUT', False)),
        ati.derives(['Window_left', 'Window_right'], to=0,
                    when=ati.ne('CAUSAL_TYPE', 3)),

        ati.disable(when=_bwd_disabled),
    ]
    ati.describe(kernel, *specs)
    return kernel
