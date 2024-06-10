#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import numpy as np
import os

from varlen_attn_torch_function import varlen_attention
from _common_test import VarlenSdpaContext, SdpaParams

# SEQLEN_Q = [4, 8, 64, 143, 256, 512, 1024, 2048]
# SEQLEN_K = [4, 8, 64, 128, 256, 587, 1024, 2048]

SEQLEN_Q = [4, 8, 64, 143]
SEQLEN_K = [4, 8, 63, 128]

# SEQLEN_Q = [16]
# SEQLEN_K = [16]

POSSIBLE_SEQLEN = sorted(set(SEQLEN_Q + SEQLEN_K))

def rng_seqlens(n_seqlen):
    return np.random.choice(POSSIBLE_SEQLEN, n_seqlen)
    # assert n_seqlen == 2
    # return np.array([4, 4])
    # seqlens = np.random.choice(POSSIBLE_SEQLEN, n_seqlen)
    # cu_seqlens = [0] + np.cumsum(seqlens).tolist()
    # return cu_seqlens

def _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype):
    import numpy as np
    if causal and not np.allclose(seqlens_q, seqlens_k):
        pytest.skip("PyTorch's Flash V2 does not accept casual=True when seqlen_q != seqlen_k. Skipping")
    # Data creation
    SKIP_DK_DV = True  # No backward
    SKIP_DQ = True  # No backward
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    ctx = VarlenSdpaContext(N_HEADS, D_HEAD, seqlens_q, seqlens_k, dtype, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=True)
    return_encoded_softmax = True if dropout_p > 0.0 else False
    q, k, v, b = ctx.dev_tensors
    # Forward
    tri_out, encoded_softmax, _ = varlen_attention(q, k, v, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, return_encoded_softmax, USE_AUTOTUNE)
    dropout_mask = encoded_softmax >= 0 if dropout_p > 0.0 else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    # # Backward
    # dout = torch.rand_like(tri_out)
    # ctx.compute_backward(tri_out, dout)
    # is_allclose, adiff, grads_allclose, grads_adiff = ctx.validate_with_reference(tri_out, ctx.dout_tensors)
    is_allclose, adiff = ctx.validate_with_reference(tri_out)

    # Test Forward
    if not is_allclose:
        import numpy as np
        print(f'{ref_out.shape=}')
        print(f'{tri_out.shape=}')
        print(f'{seqlens_q=}')
        print(f'{seqlens_k=}')
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        # print(f'{q=}')
        # print(f'{k=}')
        # print(f'{v=}')
        # print(f'{tri_out=}')
        # print(f'{ref_out=}')
        # print(f'{q[15,:4,:4]=}')
        # print(f'{k[15,:4,:4]=}')
        # print(f'{v[15,:4,:4]=}')
        # print(f'{tri_out[15,:4,:4]=}')
        # print(f'{ref_out[15,:4,:4]=}')
        # print(f'{q[16,:4,:4]=}')
        # print(f'{k[16,:4,:4]=}')
        # print(f'{v[16,:4,:4]=}')
        # print(f'{tri_out[16,:4,:4]=}')
        # print(f'{ref_out[16,:4,:4]=}')
    assert is_allclose, f'Forward pass {is_allclose=}'

@pytest.mark.parametrize('N_HEADS', [1, 4])
# @pytest.mark.parametrize('D_HEAD', [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256])
@pytest.mark.parametrize('D_HEAD', [4])  # Faster "collecting items"
# @pytest.mark.parametrize('n_seqlen', range(1, 24, 5))
@pytest.mark.parametrize('n_seqlen', [1, 2])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
def test_op_bwd(N_HEADS, D_HEAD, n_seqlen, causal, sm_scale, dropout_p, dtype):
    np.random.seed(8139)
    seqlens_q = rng_seqlens(n_seqlen)
    seqlens_k = seqlens_q if causal else rng_seqlens(n_seqlen)
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype)
