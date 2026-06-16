# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of bwd_preprocess_varlen (auxiliary bwd kernel; varlen variant of
bwd_preprocess). Same shape as bwd_preprocess; the extra seq_strides_q is a GAP
inherited from the cite. Stacked-@ form over ../kernel/bwd_preprocess.py (the
varlen kernel is defined there too).
"""

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from ._common import block_dmodel_values, MAIN_DTYPES


@dataclass
class BwdPreprocessVarlenPerf:
    BLOCK_M: np.int16 = 128


@ati.kernel
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dk_dv')
@ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Out')
@ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1)
@ati.tensor('DO',  'T_io', strides='stride_do?', contiguous=-1)
@ati.tensor('Delta', 'LazyTensor:*fp32:16', rank=2, wires_to='D')
@ati.scalar('D_HEAD', options=block_dmodel_values(), wires_to='BLOCK_DMODEL')
@ati.scalar('PADDED_HEAD', options=[False, True])
@ati.tune.schema(BwdPreprocessVarlenPerf)   # schema-only -> untunable
@ati.tune.fallback(PADDED_HEAD=False)
@ati.source('../kernel/bwd_preprocess.py', name='bwd_preprocess_varlen')
def bwd_preprocess_varlen():
    pass
