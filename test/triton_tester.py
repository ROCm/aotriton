#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
assert os.getenv('BWD_IMPL', default=None) is None, ("BWD_IMPL must not be set to run `pytest triton_tester.py`. "
                                                     "This environment variable will be set to 0 to force using Split kernel")
assert os.getenv('V3_API', default=None) is None, ("V3_API must not be set to run `pytest triton_tester.py`. "
                                                   "This environment variable will be set to 0 to force using Split kernel")
os.environ['BWD_IMPL'] = '0'
os.environ['V3_API'] = '0'

import pytest
from _core_test_backward import (
    DTYPES,
    BWDOP_ids,
    fmt_nheads,
    fmt_hdim,
    gpufilelock,
    torch_gpu,
    test_logsumexp_scaling,
    core_test_op_bwd,
    core_test_large_bf16_nan_values,
)

REGULAR_HDIMS = [48, 80, 128, 192, 224, 256]
IRREGULAR_HDIMS = [40, 72, 120, 180, 216,]

ALL_HEADDIMS = sorted(list(set(REGULAR_HDIMS + IRREGULAR_HDIMS)))

@pytest.mark.parametrize('BATCH', [3])
@pytest.mark.parametrize('N_HEADS', [5, (10, 2)], ids=fmt_nheads)
@pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q', [11, 523, 2048])
@pytest.mark.parametrize('seqlen_k', [31, 256, 1063])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('sm_scale', ['l1'])
@pytest.mark.parametrize('storage_flip', [False, (0,1), (1, 2)], ids=['BHSD', 'HBSD', 'BSHD'])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_triton_compiler(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    core_test_op_bwd(request, args, device=torch_gpu)

@pytest.mark.parametrize('D_HEAD', [48])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_large_bf16_nan_values(BWDOP, D_HEAD):
    core_test_large_bf16_nan_values(D_HEAD)
