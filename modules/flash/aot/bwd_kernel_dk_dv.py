# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd_kernel_dk_dv kernel (computes dK/dV).

A KEY backward kernel: a standalone full description (the authoritative source of
the bwd operand practices). Stacked-@ form (rev0 §5a) over ../kernel/bwd_kernel_dk_dv.py.

Note (rev1 §3.5): the B tensor's 3rd stride is `stride_bm` in the Triton signature
but the operator/golden call it `stride_bk`; ATI emits the real name in the cosmetic
pp_args comment (the access expression is identical).
"""

from dataclasses import dataclass

import numpy as np

import itertools

import aotriton.template_instantiation as ati
from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from ._common import flash_disabled, block_dmodel_values, MAIN_DTYPES, check_value


@dataclass
class BwdKernelDkDvPerf:
    BLOCK_M:  np.int16 = 16
    BLOCK_N:  np.int16 = 16
    NUM_XCDS: np.int8 = 1


def _bwd_disabled(f):
    """Shared flash disable predicate; bwd gfx950-bad head dims are {48, 80}."""
    return flash_disabled(f, gfx950_bad_hdims={48, 80})


def gen_autotune_configs(f):
    """Per-functional performance config generator (ported from 0.12b). Feeds the
    tuning build (AOTRITON_BUILD_FOR_TUNING); the DB path does not use it."""
    arch = f.arch
    dtype = check_value(f, ['Q'])
    HEAD_DIM = check_value(f, ['BLOCK_DMODEL'])
    CAUSAL_TYPE = check_value(f, ['CAUSAL_TYPE'])
    BIAS_TYPE = check_value(f, ['BIAS_TYPE'])
    ENABLE_DROPOUT = check_value(f, ['ENABLE_DROPOUT'])
    WAVE64 = AOTRITON_ARCH_WARPSIZE[arch] == 64
    WAVE32 = AOTRITON_ARCH_WARPSIZE[arch] == 32
    # TODO: right sizes for fp32?
    BLOCK_SIZES = [16, 32, 64] if dtype != '*fp32:16' else [16, 32]
    WAVES_PER_EU = [1, 2, 3, 4]
    NUM_WARPS = [4, 8] if WAVE32 else [2, 4]
    NUM_STAGES = [1]
    NUM_XCDS = 8 if arch in ('gfx942', 'gfx950') else 1
    if arch == 'gfx1250':
        # aiter gfx1250-MHA-DEFAULT.json's BLOCK_M=BLOCK_N=64/waves_per_eu=2 produces
        # NaN output on 2 tuned task_configs (tuning DB ~/wkdir.aiday, task_ids 45,72);
        # waves_per_eu=1 at the same block size stayed clean, so drop only that knob.
        kw = {'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        kw = {'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        kw = {'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        def more_configs():
            for M, N in ((32, 32), (32, 16), (16, 16)):
                for waves in (1, 2, 3, 4):
                    kw = {'BLOCK_M': M,
                          'BLOCK_N': N,
                          'waves_per_eu': waves,
                          'NUM_XCDS': NUM_XCDS}
                    yield ati.tune.Config(kw, num_stages=1, num_warps=8)
        # HEAD_DIM=256 fp32 (no bias), either non-causal+dropout or causal+no-dropout,
        # has no shipped candidate passing every UT. Add two extra block-size options
        # at the same baseline copts (nw4/we2) rather than introducing new copts.
        if HEAD_DIM == 256 and dtype == '*fp32:16' and BIAS_TYPE == 0 and (
                (CAUSAL_TYPE == 0 and ENABLE_DROPOUT) or
                (CAUSAL_TYPE != 0 and not ENABLE_DROPOUT)):
            yield from more_configs()
            return
        return
    for M, N, waves, warps, stages in itertools.product(BLOCK_SIZES,
                                                        BLOCK_SIZES,
                                                        WAVES_PER_EU,
                                                        NUM_WARPS,
                                                        NUM_STAGES):
        if M < N:
            continue  # deduplicate
        if WAVE64 and M == 64 and N == 64 and warps == 4:
            continue  # No optimal kernel according to 0.8b tuning db
        if WAVE32 and M * N >= 32 * 32 and warps < 4:
            continue  # Timeout
        if WAVE32 and M * N >= 32 * 16 and warps < 2:
            continue  # Timeout
        kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves,
              'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=stages, num_warps=warps)


@ati.start
@ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Q')
@ati.tensor('Q',  'T_io', strides='stride_q?',  contiguous=-1)
@ati.tensor('K',  'T_io', strides='stride_k?',  contiguous=-1)
@ati.tensor('V',  'T_io', strides='stride_v?',  contiguous=-1)
@ati.tensor('B',  'T_io', strides='stride_b?',  contiguous=-1)
@ati.tensor('DO', 'T_io', strides='stride_do?', contiguous=-1)
@ati.tensor('DK', 'T_io', strides='stride_dk?', contiguous=-1)
@ati.tensor('DV', 'T_io', strides='stride_dv?', contiguous=-1)
@ati.scalar('sm_scale', 'fp32')
@ati.tensor('L', '*fp32:16', rank=2)
@ati.tensor('D', 'LazyTensor:*fp32:16', rank=2)
@ati.scalar(['num_head_q', 'num_head_k', 'hdim_qk', 'hdim_vo'], 'i32')
@ati.tensor(['cu_seqlens_q', 'cu_seqlens_k',
             'seq_strides_q', 'seq_strides_k'], '*i32:16', rank=1)
@ati.scalar(['num_seqlens', 'max_seqlen_q', 'max_seqlen_k'], 'i32')
@ati.scalar('dropout_p', 'fp32')
@ati.tensor(['philox_seed_ptr', 'philox_offset1'], '*u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
@ati.scalar('Window_left', 'i32')
@ati.scalar('Window_right', 'i32')
@ati.scalar('BLOCK_DMODEL', options=block_dmodel_values())
@ati.scalar('CAUSAL_TYPE', options=[0, 3])
@ati.scalar('ENABLE_DROPOUT', options=[False, True])
@ati.scalar('PADDED_HEAD', options=[False, True])
@ati.scalar('BIAS_TYPE', options=[0, 1])
@ati.tune.schema(BwdKernelDkDvPerf)
@ati.tune.configs(gen_autotune_configs)
@ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                  max_seqlen_k=ati.tune.binning.le)
@ati.tune.fallback(PADDED_HEAD=False)
@ati.derives('NUM_XCDS', to=8, when=lambda f: f.arch in ('gfx942', 'gfx950'))
@ati.derives('B', to=0, when=ati.eq('BIAS_TYPE', 0))           # strides cascade
@ati.derives(['dropout_p', 'philox_seed_ptr', 'philox_offset1', 'philox_offset2'],
             to=0, when=ati.eq('ENABLE_DROPOUT', False))
@ati.derives(['Window_left', 'Window_right'], to=0, when=ati.ne('CAUSAL_TYPE', 3))
@ati.disable(when=_bwd_disabled)
@ati.source('../kernel/bwd_kernel_dk_dv.py')
def bwd_kernel_dk_dv():
    pass
