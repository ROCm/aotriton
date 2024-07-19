#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os

from _common_backward import _do_test_op_bwd

@pytest.mark.parametrize('BATCH', [1, 3])
@pytest.mark.parametrize('N_HEADS', [1, 5])
@pytest.mark.parametrize('D_HEAD', [16])
@pytest.mark.parametrize('seqlen_q', [16, 32])
@pytest.mark.parametrize('seqlen_k', [16, 32])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [False])
def test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [128])
@pytest.mark.parametrize('N_HEADS', [32])
@pytest.mark.parametrize('D_HEAD', [64, 128])
@pytest.mark.parametrize('seqlen_q', [512])
@pytest.mark.parametrize('seqlen_k', [512])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('sm_scale', [1.2])
@pytest.mark.parametrize('storage_flip', [False])
def test_op_perf_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

