# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake bwd_kernel_fuse: alternative fused bwd backend citing the bwd
metro's sub-kernels for merged operands (citation mode b)."""

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from ._common import block_dmodel_values


def _block_dmodel_values_capped():
    # The fused bwd kernel caps BLOCK_DMODEL at 256 (legacy v3python rule
    # rules/flash/bwd_kernel_fuse.py: `BLOCK_DMODEL: lambda x: x <= 256`).
    return [d for d in block_dmodel_values() if d <= 256]


@dataclass
class BwdKernelFusePerf:
    BLOCK_M: np.int16 = 16
    BLOCK_N: np.int16 = 16


def gen_autotune_configs(f):
    """Placeholder generator (one valid config); the DB path does not use it."""
    yield ati.tune.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1},
                          num_warps=4, num_stages=1)


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
