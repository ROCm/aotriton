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

import itertools
import os
from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from ._common import flash_disabled, block_dmodel_values, MAIN_DTYPES, check_value


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


def _parse_preload_options():
    val = int(os.getenv('AOTRITON_PRE_LOAD_OPTIONS', default='2'))
    if val == 0:
        return [False]
    if val == 1:
        return [True]
    return [False, True]


PRE_LOAD_OPTIONS = _parse_preload_options()

# Wider forward search space for gfx1100 inference functionals: when
# _use_extended_search accepts one, these replace the wave32 block sizes/stages
# below and are the whole candidate space (they include the wave32 tiles).
EXTENDED_SEARCH_ARCHS = ('gfx1100',)
EXTENDED_SEARCH_HEAD_DIMS = (64, 80, 128)
# Tile sizes that are searched in the extended search space.
EXTENDED_SEARCH_BLOCK_SIZES = [
    (16, 16), (32, 16), (32, 32),
    (64, 16), (64, 32), (64, 64),
    (128, 16), (128, 32), (128, 64), (128, 128),
    (256, 16), (256, 32), (256, 64),
]
EXTENDED_SEARCH_NUM_STAGES = [1, 2]
# A third pipeline stage is searched on the BLOCK_N=16 tiles only -
# this is derived from heuristics and experimental tuning results.
EXTENDED_SEARCH_DEEP_PIPELINE_BLOCK_N = 16
EXTENDED_SEARCH_DEEP_PIPELINE_NUM_STAGES = [3]


def extended_search_grid():
    """The widened space as ((BLOCK_M, BLOCK_N), num_stages) pairs."""
    for tile in EXTENDED_SEARCH_BLOCK_SIZES:
        stages = list(EXTENDED_SEARCH_NUM_STAGES)
        if tile[1] == EXTENDED_SEARCH_DEEP_PIPELINE_BLOCK_N:
            stages += EXTENDED_SEARCH_DEEP_PIPELINE_NUM_STAGES
        for num_stages in stages:
            yield tile, num_stages


def _use_extended_search(f, arch, dtype, head_dim, causal_type):
    """Whether `f` is one of the inference functionals the wider space targets.
    """
    return (arch in EXTENDED_SEARCH_ARCHS
            and '*fp32' not in dtype
            and head_dim in EXTENDED_SEARCH_HEAD_DIMS
            and causal_type == 0
            and f.choices.BIAS_TYPE == 0
            and f.choices.ENABLE_DROPOUT is False
            and f.choices.PADDED_HEAD is False)


def gen_autotune_configs(f):
    """Generate architecture-aware forward tuning configurations."""
    arch = f.arch
    dtype = check_value(f, ['Q'])
    head_dim = f.choices.BLOCK_DMODEL
    causal_type = f.choices.CAUSAL_TYPE
    num_xcds = 8 if arch in ('gfx942', 'gfx950') else 1
    wave64 = AOTRITON_ARCH_WARPSIZE[arch] == 64

    if wave64:
        block_sizes = [(32, 16), (128, 64), (64, 64), (64, 32), (128, 128)]
    else:
        block_sizes = [(64, 32), (32, 32), (32, 16)]
        if '*fp32' not in dtype:
            block_sizes.append((16, 16))

    waves_per_eu = [1, 2, 3, 4]
    num_warps = [2, 4] if wave64 else [4, 8]
    num_stages = [1]

    if _use_extended_search(f, arch, dtype, head_dim, causal_type):
        tile_stages = list(extended_search_grid())
    else:
        tile_stages = [(tile, s) for tile in block_sizes for s in num_stages]

    if arch == 'gfx950':
        for waves, pre_load_v in itertools.product(waves_per_eu, PRE_LOAD_OPTIONS):
            kw = {
                'PERSISTENT_TYPE': 2 if causal_type != 0 else 0,
                'GRID_CU_MULTIP': 2,
                'BLOCK_M': 256,
                'BLOCK_N': 64,
                'waves_per_eu': waves,
                'PRE_LOAD_V': pre_load_v,
                'NUM_XCDS': num_xcds,
            }
            yield ati.tune.Config(kw, num_stages=4, num_warps=8)

    for ((block_m, block_n), stages), waves, warps, pre_load_v in itertools.product(
            tile_stages, waves_per_eu, num_warps, PRE_LOAD_OPTIONS):
        if head_dim >= 512 and block_m == 128 and block_n == 128 and warps == 2:
            continue  # Timeout
        if dtype == '*fp32:16':
            block_m //= 2
        if block_m < block_n:  # Faulty or duplicate
            continue
        kw = {
            'PERSISTENT_TYPE': 2 if causal_type != 0 else 0,
            'GRID_CU_MULTIP': 2,
            'BLOCK_M': block_m,
            'BLOCK_N': block_n,
            'waves_per_eu': waves,
            'PRE_LOAD_V': pre_load_v,
            'NUM_XCDS': num_xcds,
        }
        yield ati.tune.Config(kw, num_stages=stages, num_warps=warps)


def _attn_fwd_disabled(f):
    """Compiler/numerical correctness exclusions; fwd gfx950 bad head dim is {16}."""
    return flash_disabled(f, gfx950_bad_hdims={16})


@ati.start
# --- dtype variables (named; tensors below reference them by string) ---
@ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Q')
@ati.type_var('T_seq', dtype=['*i32:16'])
@ati.type_var('T_u64', dtype=['*u64'])
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
