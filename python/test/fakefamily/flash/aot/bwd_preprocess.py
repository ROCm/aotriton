# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake bwd_preprocess: auxiliary bwd kernel citing a key kernel for
shared operands; exercises apparel renames (Delta->D, D_HEAD->BLOCK_DMODEL)."""

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from ._common import block_dmodel_values, MAIN_DTYPES


@dataclass
class BwdPreprocessPerf:
    BLOCK_M: np.int16 = 128


@ati.start
# Cite a bwd key kernel (citing the whole metro would be circular — the metro calls
# this preprocess). The 3-segment cite resolves via the flat kernel registry.
# The cited bwd_kernel_dk_dv carries _bwd_disabled (reads CAUSAL_TYPE/BLOCK_DMODEL/
# BIAS_TYPE), which are not in this preprocess kernel's choice space. Replace with
# no_disable — preprocess has no correctness exclusions of its own (rev0 §4.5).
@ati.no_disable()
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dk_dv')
@ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Out')
@ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1)
@ati.tensor('DO',  'T_io', strides='stride_do?', contiguous=-1)
@ati.tensor('Delta', 'LazyTensor:*fp32:16', rank=2, wires_to='D')
@ati.scalar('D_HEAD', options=block_dmodel_values(), wires_to='BLOCK_DMODEL')
@ati.scalar('PADDED_HEAD', options=[False, True])
@ati.tune.schema(BwdPreprocessPerf)         # schema-only -> untunable
@ati.tune.fallback(PADDED_HEAD=False)
@ati.source('../kernel/bwd_preprocess.py')
def bwd_preprocess():
    pass
