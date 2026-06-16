# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash attn_fwd kernel.

Stacked-@ form (rev0 §5a): @ati.source imports the Triton kernel from
../kernel/fwd_kernel.py and the @ati.* decorators above stack the full
instantiation description onto it. Covers all 74 parameters. Conditional arguments
(legacy CC/CDC/CDETensor) are @ati.derives; the axis fixes the struct ABI type and
the derive fixes the per-functional value (ati+newbinds_rev1.md §6.2).
"""

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from ._common import flash_disabled, block_dmodel_values, MAIN_DTYPES


@dataclass
class AttnFwdPerf:
    # Untuned/empty-path base values (tuned functionals read from the DB).
    # PERSISTENT_TYPE and NUM_XCDS are refined per-functional by @ati.derives below.
    PERSISTENT_TYPE: np.int8 = 0
    GRID_CU_MULTIP:  np.int8 = 2
    BLOCK_M:         np.int16 = 16
    BLOCK_N:         np.int16 = 16
    PRE_LOAD_V:      bool = False
    NUM_XCDS:        np.int8 = 1


def gen_autotune_configs(f):
    """Placeholder generator (real heuristics ported later); the DB-driven path
    (translate_dataframe) does not use this."""
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


def _attn_fwd_disabled(f):
    """Compiler/numerical correctness exclusions; fwd gfx950 bad head dim is {16}."""
    return flash_disabled(f, gfx950_bad_hdims={16})


@ati.kernel
# --- dtype variables (named; tensors below reference them by string) ---
@ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q')
@ati.tensor_dtype('T_seq', dtype=['*i32:16'])
@ati.tensor_dtype('T_u64', dtype=['*u64'])
# --- main tensors (rank 4, last stride contiguous) ---
@ati.tensor('Q',   'T_io', strides='stride_q?', contiguous=-1)
@ati.tensor('K',   'T_io', strides='stride_k?', contiguous=-1)
@ati.tensor('V',   'T_io', strides='stride_v?', contiguous=-1)
@ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1)
@ati.tensor('B',   'T_io', strides='stride_b?', contiguous=-1)
@ati.tensor('A',   'T_io', strides='stride_a?', contiguous=-1)   # rank 2
@ati.tensor('L', '*fp32:16', rank=2)
@ati.scalar('Sm_scale', 'fp32')
# --- INT8 descales: constexpr 0, int8 ABI (one shared axis) ---
@ati.scalar(['Q_descale', 'K_descale', 'P_scale', 'P_descale', 'V_descale'],
            options=[0])
# --- MQA/GQA + varlen scalars ---
@ati.scalar(['Num_head_q', 'Num_head_k', 'Num_seqlens',
             'Max_seqlen_q', 'Max_seqlen_k'], 'i32')
@ati.tensor(['cu_seqlens_q', 'cu_seqlens_k',
             'seq_strides_q', 'seq_strides_k'], 'T_seq', rank=1)
# --- head dims ---
@ati.scalar('BLOCK_DMODEL', options=block_dmodel_values())
@ati.scalar(['Hdim_qk', 'Hdim_vo'], 'i32')
@ati.scalar('PADDED_HEAD', options=[False, True])
# --- dropout + PRNG ---
@ati.scalar('ENABLE_DROPOUT', options=[False, True])
@ati.scalar('dropout_p', 'fp32')
@ati.tensor(['philox_seed_ptr', 'philox_offset1',
             'philox_seed_output', 'philox_offset_output'], 'T_u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
@ati.scalar('RETURN_ENCODED_SOFTMAX', options=[False])
@ati.tensor('encoded_softmax', 'T_io', rank=4)
# --- causal / windowed ---
@ati.scalar('CAUSAL_TYPE', options=[0, 3])
@ati.scalar('Window_left', 'i32')
@ati.scalar('Window_right', 'i32')
# --- bias / alibi / int8 flags ---
@ati.scalar('BIAS_TYPE', options=[0, 1])
@ati.scalar('USE_ALIBI', options=[False])
@ati.scalar(['INT8', 'INT8_KV', 'USE_P_SCALE'], options=[False])
# --- persistent ---
@ati.tensor('persistent_atomic_counter', '*i32', rank=0)
@ati.scalar(['Num_CU', 'Batch'], 'i32')
# --- performance ---
@ati.tune.schema(AttnFwdPerf)
@ati.tune.configs(gen_autotune_configs)
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                  Max_seqlen_k=ati.tune.binning.le)
@ati.tune.fallback(PADDED_HEAD=False)
# --- perf-value derives ---
@ati.derives('PERSISTENT_TYPE', to=2, when=ati.ne('CAUSAL_TYPE', 0))
@ati.derives('NUM_XCDS', to=8, when=lambda f: f.arch in ('gfx942', 'gfx950'))
# --- conditional overrides (legacy CC/CDC/CDETensor) ---
@ati.derives('B', to=0, when=ati.eq('BIAS_TYPE', 0))            # strides cascade
@ati.derives('A', to=0, when=ati.eq('USE_ALIBI', False))        # stride cascades
@ati.derives('Hdim_qk', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
@ati.derives('Hdim_vo', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
@ati.derives(['dropout_p', 'philox_seed_ptr', 'philox_offset1',
              'philox_offset2', 'philox_seed_output', 'philox_offset_output'],
             to=0, when=ati.eq('ENABLE_DROPOUT', False))
@ati.derives('encoded_softmax', to=0, when=ati.eq('RETURN_ENCODED_SOFTMAX', False))
@ati.derives(['Window_left', 'Window_right'], to=0, when=ati.ne('CAUSAL_TYPE', 3))
@ati.derives('persistent_atomic_counter', to=0, when=ati.eq('CAUSAL_TYPE', 0))
# --- functional-disable ---
@ati.disable(when=_attn_fwd_disabled)
@ati.source('../kernel/fwd_kernel.py')
def attn_fwd():
    pass
