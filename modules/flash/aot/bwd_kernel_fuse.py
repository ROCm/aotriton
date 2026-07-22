# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd_kernel_fuse kernel.

An ALTERNATIVE fused bwd backend (computes dQ/dK/dV/dB in one kernel), NOT called
by the triton_split metro. It cites the metro's sub-kernels for the merged operand
vocabulary (citation mode (b), rev0 §4.4), declaring only what differs:
BLOCK_DMODEL capped <= 256 (legacy CHOICE_FILTERS), Out (whose stride names differ
from the cited preprocess, rev1 §5), and its own perf (no NUM_XCDS). Stacked-@ form
over ../kernel/bwd_kernel_fuse.py.
"""

from dataclasses import dataclass

import numpy as np

import itertools

import aotriton.template_instantiation as ati
from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from ._common import block_dmodel_values, check_value


def _block_dmodel_values_capped():
    # The fused bwd kernel caps BLOCK_DMODEL at 256 (legacy v3python rule
    # rules/flash/bwd_kernel_fuse.py: `BLOCK_DMODEL: lambda x: x <= 256`).
    return [d for d in block_dmodel_values() if d <= 256]


@dataclass
class BwdKernelFusePerf:
    BLOCK_M: np.int16 = 16
    BLOCK_N: np.int16 = 16


def gen_autotune_configs(f):
    """Per-functional performance config generator (ported from 0.12b). Feeds the
    tuning build (AOTRITON_BUILD_FOR_TUNING); the DB path does not use it."""
    arch = f.arch
    dtype = check_value(f, ['Q'])
    WAVE64 = AOTRITON_ARCH_WARPSIZE[arch] == 64
    WAVE32 = AOTRITON_ARCH_WARPSIZE[arch] == 32
    # TODO: right sizes for fp32?
    BLOCK_SIZES = [16, 32, 64] if dtype != '*fp32:16' else [16, 32]
    WAVES_PER_EU = [1, 2, 3, 4]
    NUM_WARPS = [4, 8] if WAVE32 else [2, 4]
    NUM_STAGES = [1]
    if arch == 'gfx1250':
        # aiter gfx1250-MHA-DEFAULT.json: fwd.default, plus smaller backups.
        kw = {'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 2}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        kw = {'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        kw = {'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
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
        if WAVE32 and M == 32 and N == 32 and warps != 4:
            continue  # Timeout
        kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves}
        yield ati.tune.Config(kw, num_stages=stages, num_warps=warps)


@ati.start
# Cite the bwd metro's sub-kernels for the merged operand practices (mode b).
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dk_dv')
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dq')
@ati.cite('op_attn_bwd.triton_split.bwd_preprocess')
# Out declared locally: its strides stride_o{z,h,m,k} differ from the cited
# preprocess's stride_o{z,h,m,n}, so they cannot be inherited by name (rev1 §5);
# the dtype T_io still comes from the cite.
@ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1)
# Local override: cap BLOCK_DMODEL at <= 256 (legacy CHOICE_FILTERS) -> 11 values.
@ati.scalar('BLOCK_DMODEL', options=_block_dmodel_values_capped())
# Own perf schema (no NUM_XCDS, unlike dk_dv/dq).
@ati.tune.schema(BwdKernelFusePerf)
@ati.tune.configs(gen_autotune_configs)
@ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                  max_seqlen_k=ati.tune.binning.le)
@ati.tune.fallback(PADDED_HEAD=False)
# No local @ati.disable: fuse inherits the cited bwd kernels' disable predicate.
@ati.source('../kernel/bwd_kernel_fuse.py')
def bwd_kernel_fuse():
    pass
