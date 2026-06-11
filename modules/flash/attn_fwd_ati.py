# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full ATI description of the flash attn_fwd kernel (executive plan Step 4.2.1).

Mode B: the Triton source (tritonsrc/fwd_kernel.py) is untouched; this module
imports the kernel and attaches an ATI description via ati.describe(). It covers
all 74 parameters so describe() completeness passes.

Parity target: v3python/rules/flash/attn_fwd.py + op_attn_fwd.py. Conditional
arguments (the legacy CC/CDC/CDETensor cases) are expressed as @ati.overrides:
the axis fixes the struct ABI type, the override fixes the per-functional value
(ati+newbinds_rev1.md §6.2).
"""

import os
from dataclasses import dataclass

import numpy as np

import v3python.template_instantiation as ati


def _block_dmodel_values():
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',')]


# Main IO datatype shared by Q, K, V, Out, B, A (rank differs: A is rank 2).
MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


@dataclass
class AttnFwdPerf:
    PERSISTENT_TYPE: np.int8
    GRID_CU_MULTIP:  np.int8
    BLOCK_M:         np.int16
    BLOCK_N:         np.int16
    PRE_LOAD_V:      bool
    NUM_XCDS:        np.int8


def gen_autotune_configs(f):
    """Placeholder generator (real heuristics ported in Phase 6). Produces one
    valid config so build_for_tuning paths have something to emit; the DB-driven
    path (translate_dataframe) does not use this."""
    causal = f.choices.CAUSAL_TYPE
    kw = {
        'PERSISTENT_TYPE': 2 if causal != 0 else 0,
        'GRID_CU_MULTIP': 2,
        'BLOCK_M': 16, 'BLOCK_N': 16,
        'PRE_LOAD_V': False,
        'NUM_XCDS': 8 if f.arch in ('gfx942', 'gfx950') else 1,
        'waves_per_eu': 2,
    }
    yield ati.tune.Config(kw, num_warps=4, num_stages=1)


def describe_attn_fwd(attn_fwd):
    # Only T_io is multi-choice, so only it needs an explicit signature_name
    # (the argument recording it in the aks2 entry name / DB row key). T_seq and
    # T_u64 are single-choice (trivial) and never appear in the signature.
    T_io = ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q')
    T_seq = ati.tensor_dtype('T_seq', dtype=['*i32:16'])
    T_u64 = ati.tensor_dtype('T_u64', dtype=['*u64'])

    specs = [
        # --- main tensors (rank 4, last stride contiguous) ---
        ati.tensor('Q',   T_io, strides='stride_q?', contiguous=-1),
        ati.tensor('K',   T_io, strides='stride_k?', contiguous=-1),
        ati.tensor('V',   T_io, strides='stride_v?', contiguous=-1),
        ati.tensor('Out', T_io, strides='stride_o?', contiguous=-1),
        ati.tensor('B',   T_io, strides='stride_b?', contiguous=-1),
        ati.tensor('A',   T_io, strides='stride_a?', contiguous=-1),   # rank 2 (2 strides)
        ati.tensor('L', '*fp32:16', rank=2),
        ati.scalar('Sm_scale', 'fp32'),

        # --- INT8 descales: constexpr 0, int8 ABI (one shared axis) ---
        ati.scalar(['Q_descale', 'K_descale', 'P_scale', 'P_descale', 'V_descale'],
                   options=[0]),

        # --- MQA/GQA + varlen scalars ---
        ati.scalar(['Num_head_q', 'Num_head_k', 'Num_seqlens',
                    'Max_seqlen_q', 'Max_seqlen_k'], 'i32'),
        ati.tensor(['cu_seqlens_q', 'cu_seqlens_k',
                    'seq_strides_q', 'seq_strides_k'], T_seq, rank=1),

        # --- head dims ---
        ati.scalar('BLOCK_DMODEL', options=_block_dmodel_values()),
        ati.scalar(['Hdim_qk', 'Hdim_vo'], 'i32'),
        ati.scalar('PADDED_HEAD', options=[False, True]),

        # --- dropout + PRNG ---
        ati.scalar('ENABLE_DROPOUT', options=[False, True]),
        ati.scalar('dropout_p', 'fp32'),
        ati.tensor(['philox_seed_ptr', 'philox_offset1',
                    'philox_seed_output', 'philox_offset_output'], T_u64, rank=0),
        ati.scalar('philox_offset2', 'u64'),
        ati.scalar('RETURN_ENCODED_SOFTMAX', options=[False]),
        ati.tensor('encoded_softmax', T_io, rank=4),

        # --- causal / windowed ---
        ati.scalar('CAUSAL_TYPE', options=[0, 3]),
        ati.scalar('Window_left', 'i32'),
        ati.scalar('Window_right', 'i32'),

        # --- bias / alibi / int8 flags ---
        ati.scalar('BIAS_TYPE', options=[0, 1]),
        ati.scalar('USE_ALIBI', options=[False]),
        ati.scalar(['INT8', 'INT8_KV', 'USE_P_SCALE'], options=[False]),

        # --- persistent ---
        ati.tensor('persistent_atomic_counter', '*i32', rank=0),
        ati.scalar(['Num_CU', 'Batch'], 'i32'),

        # --- performance (perf schema claims these) ---
        ati.tune.schema(AttnFwdPerf),
        ati.tune.configs(gen_autotune_configs),
        ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                         Max_seqlen_k=ati.tune.binning.le),
        ati.tune.fallback(PADDED_HEAD=False),
        ati.tune.derived(NUM_XCDS=lambda f: 8 if f.arch in ('gfx942', 'gfx950') else 1),

        # --- conditional overrides (legacy CC/CDC/CDETensor) ---
        # bias off -> B and its non-unit strides become constexpr 0
        ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0)),
        ati.overrides(['stride_bz', 'stride_bh', 'stride_bm'], to=0,
                      when=ati.eq('BIAS_TYPE', 0)),
        # alibi off (always) -> A and its non-unit stride become constexpr 0
        ati.overrides('A', to=0, when=ati.eq('USE_ALIBI', False)),
        ati.overrides('stride_az', to=0, when=ati.eq('USE_ALIBI', False)),
        # padded head off -> Hdim_qk/vo defer to BLOCK_DMODEL
        ati.overrides('Hdim_qk', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False)),
        ati.overrides('Hdim_vo', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False)),
        # dropout off -> dropout_p / philox become constexpr 0
        ati.overrides(['dropout_p', 'philox_seed_ptr', 'philox_offset1',
                       'philox_offset2', 'philox_seed_output', 'philox_offset_output'],
                      to=0, when=ati.eq('ENABLE_DROPOUT', False)),
        # encoded softmax off (always) -> encoded_softmax becomes constexpr 0
        ati.overrides('encoded_softmax', to=0,
                      when=ati.eq('RETURN_ENCODED_SOFTMAX', False)),
        # non-sliding-window -> Window_left/right become constexpr 0
        ati.overrides(['Window_left', 'Window_right'], to=0,
                      when=ati.ne('CAUSAL_TYPE', 3)),
        # non-causal -> persistent_atomic_counter becomes constexpr 0
        ati.overrides('persistent_atomic_counter', to=0,
                      when=ati.eq('CAUSAL_TYPE', 0)),
    ]
    ati.describe(attn_fwd, *specs)
    return attn_fwd
