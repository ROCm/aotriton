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

import itertools
from dataclasses import dataclass

import numpy as np

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
    """Generate architecture-aware dK/dV tuning configurations."""
    arch = f.arch
    dtype = check_value(f, ['Q'])
    num_xcds = 8 if arch in ('gfx942', 'gfx950') else 1
    wave64 = AOTRITON_ARCH_WARPSIZE[arch] == 64
    wave32 = AOTRITON_ARCH_WARPSIZE[arch] == 32
    block_sizes = [16, 32, 64] if dtype != '*fp32:16' else [16, 32]
    waves_per_eu = [1, 2, 3, 4]
    num_warps = [4, 8] if wave32 else [2, 4]

    for block_m, block_n, waves, warps in itertools.product(
            block_sizes, block_sizes, waves_per_eu, num_warps):
        if block_m < block_n:
            continue  # Duplicate
        if wave64 and block_m == 64 and block_n == 64 and warps == 4:
            continue  # No optimal kernel according to the 0.8b tuning database
        if wave32 and block_m * block_n >= 32 * 32 and warps < 4:
            continue  # Timeout
        if wave32 and block_m * block_n >= 32 * 16 and warps < 2:
            continue  # Timeout
        kw = {
            'BLOCK_M': block_m,
            'BLOCK_N': block_n,
            'NUM_XCDS': num_xcds,
            'waves_per_eu': waves,
        }
        yield ati.tune.Config(kw, num_stages=1, num_warps=warps)


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
# Named by ROLE rather than by mode: ?0 is the LENGTH source, ?1 the POSITION
# source. Which is read (and whether either is) is what varlen_bits says.
@ati.tensor(['seqinfo_q0', 'seqinfo_k0',
             'seqinfo_q1', 'seqinfo_k1'], '*i32:16', rank=1)
# varlen_bits replaces the tri-state num_seqlens, which is gone outright here:
# the sequence count N is tl.num_programs(2) in the backward kernels, so it was
# never needed as an argument.
@ati.scalar(['varlen_bits', 'max_seqlen_q', 'max_seqlen_k'], 'i32')
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
