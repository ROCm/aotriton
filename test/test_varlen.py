#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import numpy as np
import math
import os

from varlen_attn_torch_function import varlen_attention, AttentionExtraArgs
from _common_test import (
    VarlenSdpaContext,
    PaddedVarlenSdpaContext,
    StridedVarlenSdpaContext,
    SdpaParams,
    fmt_hdim,
)

FOR_RELEASE = bool(int(os.getenv('FOR_RELEASE', default='0')))

POT_HEADDIMS = [16, 32, 64, 128, 256]
NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
PRIME_HEADDIMS = [7, 23, 37, 53, 67, 73, 89, 113, 149, 179, 211, 241]

# SEQLEN_Q = [4, 8, 64, 143, 256, 512, 1024, 2048]
# SEQLEN_K = [4, 8, 64, 128, 256, 587, 1024, 2048]

SEQLEN_Q = [4, 8, 64, 143]
SEQLEN_K = [4, 8, 63, 128]

# SEQLEN_Q = [4]
# SEQLEN_K = [4]

POSSIBLE_SEQLEN = sorted(set(SEQLEN_Q + SEQLEN_K))
POSSIBLE_PADLEN = [0, 4, 7]

def rng_seqlens(n_seqlen):
    return np.random.choice(POSSIBLE_SEQLEN, n_seqlen)

def rng_padlens(n_seqlen):
    return np.random.choice(POSSIBLE_PADLEN, n_seqlen)

VARLEN_FACTORY = {
    "compact": VarlenSdpaContext,
    "padded": PaddedVarlenSdpaContext,
    "strided": StridedVarlenSdpaContext,
}

def _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype, varlen_type):
    assert varlen_type in VARLEN_FACTORY.keys(), f"_do_test_varlen: unknown varlen_type {varlen_type}"
    if isinstance(D_HEAD, int):
        HDIM_QK = HDIM_VO = D_HEAD
    else:
        HDIM_QK, HDIM_VO = D_HEAD
    HDIM_MAX = max(HDIM_QK, HDIM_VO)
    if sm_scale == 'l1':
        sm_scale = 1.0 / HDIM_QK
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(HDIM_QK)
    # Data creation
    SKIP_DK_DV = False
    SKIP_DQ = False
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    factory = VARLEN_FACTORY[varlen_type]
    ctx = factory(N_HEADS, D_HEAD, seqlens_q, seqlens_k, dtype, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=True)
    q, k, v, b = ctx.dev_tensors
    # Forward
    ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                             autotune=USE_AUTOTUNE,
                             return_autotune=False,
                             fillnan=True)
    tri_out, encoded_softmax, _ = varlen_attention(q, k, v, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, varlen_type, ext)
    dropout_mask = encoded_softmax >= 0 if dropout_p > 0.0 else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    # # Backward
    dout = torch.rand_like(tri_out)
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff, tfts = ctx.validate_with_reference(tri_out, ctx.dout_tensors, return_target_fudge_factors=True)
    torch.set_printoptions(threshold=114514, linewidth=200)

    # Test Forward
    if not is_allclose:
        import numpy as np
        print(f'{ref_out.shape=}')
        print(f'{tri_out.shape=}')
        print(f'{seqlens_q=}')
        print(f'{seqlens_k=}')
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out.cpu() - tri_out.cpu())).numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        # print(f'{tri_out=}')
        # print(f'{ref_out=}')
    assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'

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

@pytest.mark.parametrize('N_HEADS', [3])
@pytest.mark.parametrize('D_HEAD', [8, 64, 184, (24, 152), (120, 8)], ids=fmt_hdim)
@pytest.mark.parametrize('n_seqlen', range(2, 24, 5))
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', ['l1'] if not FOR_RELEASE else ['l1', 'l2'])
@pytest.mark.parametrize('varlen_type', ['compact', 'padded', 'strided'])
def test_op_bwd(N_HEADS, D_HEAD, n_seqlen, causal, sm_scale, dropout_p, dtype, varlen_type):
    np.random.seed(8139)
    seqlens_q = rng_seqlens(n_seqlen)
    seqlens_k = seqlens_q if causal else rng_seqlens(n_seqlen)
    if varlen_type == 'strided':
        padlens_q = rng_padlens(n_seqlen)
        padlens_k = padlens_q if causal else rng_padlens(n_seqlen)
        seqlens_q = np.array([seqlens_q, padlens_q])
        seqlens_k = np.array([seqlens_k, padlens_k])
    _do_test_varlen(N_HEADS, D_HEAD,
                    seqlens_q, seqlens_k,
                    causal, sm_scale, dropout_p, dtype, varlen_type)

def main1():
    N_HEADS = 3
    D_HEAD = 8
    seqlens_q = np.array([ 4, 143, 128, 143, 143,])
    seqlens_k = np.array([ 8,  63,   8,  63,  63,])
    # seqlens_q = np.array([4, 8])
    # seqlens_k = seqlens_q
    causal = False
    sm_scale = 1.0 / 8.0
    # dropout_p = 0.5
    dropout_p = 0.0
    dtype = torch.float16
    # varlen_type = 'compact'
    varlen_type = 'padded'
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype, varlen_type)

def main2():
    N_HEADS = 3
    D_HEAD = 8
    # seqlens_q = np.array([ 4,  31, 8])
    # seqlens_k = np.array([ 8,  63, 8])
    # padlens_q = np.array([ 2,   3, 0])
    # padlens_k = np.array([ 5,   7, 0])
    seqlens_q = np.array([ 8, 8, 8])
    seqlens_k = np.array([ 8, 8, 8])
    padlens_q = np.array([ 0, 32, 0])
    padlens_k = np.array([ 0, 32, 0])
    causal = False
    seqlens_q = np.array([seqlens_q, padlens_q])
    seqlens_k = np.array([seqlens_k, padlens_k])
    sm_scale = 1.0 / 8.0
    dropout_p = 0.0
    dtype = torch.float16
    varlen_type = 'strided'
    _do_test_varlen(N_HEADS, D_HEAD, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, dtype, varlen_type)


if __name__ == '__main__':
    main2()
