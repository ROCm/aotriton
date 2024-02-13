#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import json

from rocm_arch import rocm_get_gpuarch
import triton
from attn_torch_function import attention

# FIXME: Use fixture instead
class BestConfigRecord:
    best_config_database = []

@pytest.fixture
def teardown(scope="module"):
    arch = rocm_get_gpuarch()
    yield
    with open(f'tune-flash-{arch}.json', 'w') as f:
        d = {
                'arch' : arch,
                'tune_info': BestConfigRecord.best_config_database,
        }
        json.dump(d, f, indent=4)

@pytest.mark.parametrize('BATCH', [4])
@pytest.mark.parametrize('N_HEADS', [4])
@pytest.mark.parametrize('D_HEAD', [16,32,64,128,256])
# @pytest.mark.parametrize('seqlen_q', [8,16,32,64,128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [8,16,32,64,128,256,512,1024])
@pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
@pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dropout_p', [0.5, 0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('D_HEAD', [64])
# @pytest.mark.parametrize('seqlen_q', [128])
# @pytest.mark.parametrize('seqlen_k', [128])
# @pytest.mark.parametrize('causal', [True, False])
# @pytest.mark.parametrize('dropout_p', [0.5])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [1.2])
# @pytest.mark.parametrize('return_encoded_softmax', [False, True])  # Runtime Error with True
@pytest.mark.parametrize('return_encoded_softmax', [False])
def test_tune_fwd(teardown, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype):
    q = torch.randn((BATCH, N_HEADS, seqlen_q, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, seqlen_k, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, N_HEADS, seqlen_k, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    autotune = True
    return_autotune = True
    tri_out, encoded_softmax, best_configs = attention(q, k, v, causal, sm_scale, dropout_p, return_encoded_softmax, autotune, return_autotune)
    print(f'{id(best_configs)=}')
    dout = torch.randn_like(q)
    tri_out.backward(dout)
    BestConfigRecord.best_config_database += best_configs
