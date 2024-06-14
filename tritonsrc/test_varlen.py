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

# SEQLEN_Q = [4]
# SEQLEN_K = [4]

POSSIBLE_SEQLEN = sorted(set(SEQLEN_Q + SEQLEN_K))

def rng_seqlens(n_seqlen):
    return np.random.choice(POSSIBLE_SEQLEN, n_seqlen)

def _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype):
    import numpy as np
    if causal and not np.allclose(seqlens_q, seqlens_k):
        pytest.skip("PyTorch's Flash V2 does not accept casual=True when seqlen_q != seqlen_k. Skipping")
    # Data creation
    SKIP_DK_DV = False
    SKIP_DQ = False
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
    dout = torch.rand_like(tri_out)
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff = ctx.validate_with_reference(tri_out, ctx.dout_tensors)
    torch.set_printoptions(threshold=114514, linewidth=200)

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
    assert is_allclose, f'Forward pass {is_allclose=}'

    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=dtype)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{torch.isnan(ref_dv).any()=}')

    if dv_allclose and not dk_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')

    if dk_allclose and dv_allclose and not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

    if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
        print(f'{err_idx=}')
        print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')

    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}'
    print(f'{adiff=} {grads_adiff=}')

@pytest.mark.parametrize('N_HEADS', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [2])
@pytest.mark.parametrize('D_HEAD', [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256])
# @pytest.mark.parametrize('D_HEAD', [16])  # Faster "collecting items"
@pytest.mark.parametrize('n_seqlen', range(1, 24, 5))
# @pytest.mark.parametrize('n_seqlen', [2])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
def test_op_bwd(N_HEADS, D_HEAD, n_seqlen, causal, sm_scale, dropout_p, dtype):
    np.random.seed(8139)
    seqlens_q = rng_seqlens(n_seqlen)
    seqlens_k = seqlens_q if causal else rng_seqlens(n_seqlen)
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype)

def main():
    N_HEADS = 2
    D_HEAD = 4
    seqlens_q = np.array([4, 8])
    seqlens_k = seqlens_q
    causal = False
    sm_scale = 1.2
    dropout_p = 0.5
    dtype = torch.float16
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype)

if __name__ == '__main__':
    main()
