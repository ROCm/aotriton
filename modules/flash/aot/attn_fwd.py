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

import os
import itertools

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from ._common import flash_disabled, block_dmodel_values, MAIN_DTYPES, check_value


def _parse_preload_options():
    val = int(os.getenv('AOTRITON_PRE_LOAD_OPTIONS', default='2'))
    if val == 0:
        return [False]
    elif val == 1:
        return [True]
    else:
        return [False, True]


PRE_LOAD_OPTIONS = _parse_preload_options()


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
    """Per-functional performance config generator (ported from 0.12b). Feeds the
    tuning build (AOTRITON_BUILD_FOR_TUNING); the DB-driven path
    (translate_dataframe) does not use this."""
    arch = f.arch
    dtype = check_value(f, ['Q'])
    HEAD_DIM = check_value(f, ['BLOCK_DMODEL'])
    CAUSAL_TYPE = check_value(f, ['CAUSAL_TYPE'])
    BIAS_TYPE = check_value(f, ['BIAS_TYPE'])
    ENABLE_DROPOUT = check_value(f, ['ENABLE_DROPOUT'])
    WAVE64 = AOTRITON_ARCH_WARPSIZE[arch] == 64
    WAVE32 = AOTRITON_ARCH_WARPSIZE[arch] == 32
    if WAVE64:
        BLOCK_SIZES = [(32, 16), (128, 64), (64, 64), (64, 32), (128, 128)]
    elif WAVE32:
        BLOCK_SIZES = [(64, 32), (32, 32), (32, 16)]
        if '*fp32' not in dtype:
            BLOCK_SIZES += [(16, 16)]
        else:
            # M //= 2 will effectively yield (16, 32), (16, 16)
            pass
    WAVES_PER_EU = [1, 2, 3, 4]
    NUM_WARPS = [2, 4] if WAVE64 else [4, 8]
    PRE_LOAD_V = PRE_LOAD_OPTIONS
    NUM_STAGES = [1]
    NUM_XCDS = 8 if arch in ('gfx942', 'gfx950') else 1
    if arch == 'gfx1250':
        # aiter gfx1250-MHA-DEFAULT.json's num_warps=4/waves_per_eu=2 crashes on
        # non-causal tasks at BLOCK_M/N=32 and 16x16 (tuning DB ~/wkdir.aiday: 4 and
        # 1 crashing task_ids respectively), and BLOCK_M=128 has no validated data at
        # all (only 3 non-causal tasks tuned, causal path untested). num_warps=8/
        # waves_per_eu=2 stayed crash/NaN-free across every tested block size and
        # causal/non-causal path, so use it with the largest block size that's
        # actually covered by data (64x32) in place of the untested 128x64.
        persistent_type = 2 if CAUSAL_TYPE != 0 else 0
        for M, N in ((64, 32), (32, 32), (16, 16)):
            kw = {'PERSISTENT_TYPE': persistent_type,
                  'GRID_CU_MULTIP': 2,
                  'BLOCK_M': M,
                  'BLOCK_N': N,
                  'waves_per_eu': 2,
                  'PRE_LOAD_V': True,
                  'NUM_XCDS': NUM_XCDS}
            yield ati.tune.Config(kw, num_stages=1, num_warps=8)
        def more_configs():
            for M, N in ((64, 32), (32, 32), (16, 16)):
                for pre in (True, False):
                    for waves in (1, 2, 3, 4):
                        if pre and waves == 2:
                            continue  # already yielded above
                        kw = {'PERSISTENT_TYPE': persistent_type,
                              'GRID_CU_MULTIP': 2,
                              'BLOCK_M': M,
                              'BLOCK_N': N,
                              'waves_per_eu': waves,
                              'PRE_LOAD_V': pre,
                              'NUM_XCDS': NUM_XCDS}
                        for nwarps in (4, 8):
                            yield ati.tune.Config(kw, num_stages=1, num_warps=nwarps)
        # HEAD_DIM=256 fp32 with bias+dropout enabled register-pressures the gfx1250
        # compiler hard enough that none of the waves_per_eu=2/PRE_LOAD_V=True
        # candidates above passes accuracy on every test case (task 1540: idx0 fails
        # 02_irregular_hdim, idx2 fails 04_irregular_both, both ~1000x+ over threshold
        # - a spill/scheduling symptom, not a real accuracy limit). waves_per_eu is a
        # target-occupancy hint (lower relaxes the register budget instead of forcing
        # a tighter one) and PRE_LOAD_V=False skips prefetching V into registers/LDS
        # early, both plausible register-pressure relief valves, so sweep both here.
        # Note this can't be scoped any tighter than the compile-time functional
        # (HEAD_DIM/dtype/BIAS_TYPE/ENABLE_DROPOUT) - it also applies to other seqlen
        # buckets sharing this functional, not just the 64x64 case that motivated it,
        # since seqlen is a runtime dispatch value, not a compile-time axis.
        if HEAD_DIM == 256 and dtype == '*fp32:16' and BIAS_TYPE == 1 and ENABLE_DROPOUT:
            yield from more_configs()
            return
        # HEAD_DIM=128 fp32 causal+dropout (no bias) and HEAD_DIM=256 fp32 non-causal
        # dropout (no bias) also have no shipped candidate passing every UT. Reuse the
        # same waves_per_eu x PRE_LOAD_V sweep as the bias=1 case above - crash-safety
        # here is unvalidated (that sweep crashed every extra candidate for the bias=1
        # functional), but broader coverage across these gaps is intentional.
        _fp32_reg_pressure_hdim = (HEAD_DIM == 128 and CAUSAL_TYPE != 0) or \
                                   (HEAD_DIM == 256 and CAUSAL_TYPE == 0)
        if dtype == '*fp32:16' and BIAS_TYPE == 0 and ENABLE_DROPOUT and _fp32_reg_pressure_hdim:
            yield from more_configs()
            return
        return
    if arch == 'gfx950':
        for waves, pre in itertools.product(WAVES_PER_EU, PRE_LOAD_V):
            persistent_type = 2 if CAUSAL_TYPE != 0 else 0
            kw = {'PERSISTENT_TYPE': persistent_type,
                  'GRID_CU_MULTIP': 2,
                  'BLOCK_M': 256,
                  'BLOCK_N': 64,
                  'waves_per_eu': waves,
                  'PRE_LOAD_V': pre,
                  'NUM_XCDS': NUM_XCDS}
            yield ati.tune.Config(kw, num_stages=4, num_warps=8)
    for (M, N), waves, warps, stages, pre in itertools.product(BLOCK_SIZES,
                                                               WAVES_PER_EU,
                                                               NUM_WARPS,
                                                               NUM_STAGES,
                                                               PRE_LOAD_V):
        if HEAD_DIM >= 512 and M == 128 and N == 128 and warps == 2:
            continue  # Timeout
        if dtype == '*fp32:16':
            M //= 2
        if M < N:  # Faulty or duplicate
            continue
        persistent_type = 2 if CAUSAL_TYPE != 0 else 0
        kw = {'PERSISTENT_TYPE': persistent_type,
              'GRID_CU_MULTIP': 2,
              'BLOCK_M': M,
              'BLOCK_N': N,
              'waves_per_eu': waves,
              'PRE_LOAD_V': pre,
              'NUM_XCDS': NUM_XCDS}
        # TODO: Add Dynamic PERSISTENT_TYPE IFF causal is enabled to tuning database
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
