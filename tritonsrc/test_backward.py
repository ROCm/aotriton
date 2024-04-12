#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch

from attn_torch_function import attention

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    """
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    """
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        if dropout_mask is not None:
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    av = attn_weight @ value
    return av, attn_weight

def _make_block_eyes(q, base=1.0, inc=0.0):
    dhead = q.shape[-1]
    seqlen = q.shape[2]
    assert seqlen % dhead == 0
    scale = base
    for i in range(0, seqlen, dhead):
        q[:, :, i:i+dhead, :] = torch.eye(dhead, device=q.device, dtype=q.dtype) * scale
        scale += inc

def RP(x):
    rounded = 2 ** (x - 1).bit_length()
    return max(16, rounded)

'''
Flash Attention is batch operator that evaluates sm(QK')V
Q = batch_size x ... x seqlen_q x head_size
K = batch_size x ... x seqlen_k x head_size
    => K' = batch_size x ... x head_size x seqlen_k
V = batch_size x ... x seqlen_k x head_size
sm(.) = softmax(.)
The output size is
batch_size x ... x seqlen_q x head_size

Note: In Flash V2 API the ... is denoted as "num_heads", serving as uniformly sized sequences
but in PyTorch API it does not present at all
'''

def query_key_value_clones(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor, dtype: torch.dtype = None, device=None):
    """ Clones the query, key, and value tensors and moves them to the specified dtype. """
    if dtype is None:
        dtype = query.dtype
    query_ref = query.clone().detach().to(dtype=dtype, device=device).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype=dtype, device=device).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype=dtype, device=device).requires_grad_(value.requires_grad)
    bias_ref = bias.clone().detach().to(dtype=dtype, device=device).requires_grad_(bias.requires_grad) if bias is not None else None
    return query_ref, key_ref, value_ref, bias_ref

def _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if causal and seqlen_q != seqlen_k:
        pytest.skip("PyTorch's Flash V2 does not accept casual=True when seqlen_q != seqlen_k. Skipping")
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    SKIP_DK_DV = False
    SKIP_DQ = False
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    qdims = (BATCH, N_HEADS, seqlen_q, D_HEAD)
    kdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
    vdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
    bdims = (BATCH, N_HEADS, seqlen_q, seqlen_k)
    if storage_flip:
        qdims = (qdims[0], qdims[2], qdims[1], qdims[3])
        kdims = (kdims[0], kdims[2], kdims[1], kdims[3])
        vdims = (vdims[0], vdims[2], vdims[1], vdims[3])
        bdims = (bdims[0], bdims[2], bdims[1], bdims[3])
    q = torch.empty(qdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    k = torch.empty(kdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty(vdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    if bias_type is None:
        b = None
    elif bias_type == 'matrix':
        b = torch.empty(bdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    else:
        assert False, f'Unsupported bias_type {bias_type}'
    if storage_flip:
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        if b is not None:
            b = torch.transpose(b, 1, 2)
    if not SKIP_DQ:
        q.requires_grad_()
    if not SKIP_DK_DV:
        k.requires_grad_()
        v.requires_grad_()
    return_encoded_softmax = True

    # q_ref_lp, k_ref_lp, v_ref_lp = query_key_value_clones(q, k, v, dtype=dtype)
    higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    # REF_DEVICE='cpu'
    REF_DEVICE=None
    q_ref, k_ref, v_ref, b_ref = query_key_value_clones(q, k, v, b, dtype=higher_precision_dtype, device=REF_DEVICE)
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=dtype)
    # autotune = True
    # # triton implementation
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, return_encoded_softmax, USE_AUTOTUNE)
    dropout_mask = encoded_softmax >= 0
    '''
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    '''
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q_ref, k_ref, v_ref,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                attn_mask=b_ref,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    dout = torch.randn_like(q)
    tri_out.backward(dout)
    tri_dv, v.grad = None if SKIP_DK_DV else v.grad.clone(), None
    tri_dk, k.grad = None if SKIP_DK_DV else k.grad.clone(), None
    tri_dq, q.grad = None if SKIP_DQ else q.grad.clone(), None
    '''
    ref_out.backward(dout, None)
    ref_dv, v.grad = None if SKIP_DK_DV else v.grad.clone(), None
    ref_dk, k.grad = None if SKIP_DK_DV else k.grad.clone(), None
    ref_dq, q.grad = None if SKIP_DQ else q.grad.clone(), None
    '''
    ref_out.backward(dout.to(device=ref_out.device, dtype=ref_out.dtype))
    ref_dv, v.grad = None if SKIP_DK_DV else v_ref.grad.clone(), None
    ref_dk, k.grad = None if SKIP_DK_DV else k_ref.grad.clone(), None
    ref_dq, q.grad = None if SKIP_DQ else q_ref.grad.clone(), None
    # compare
    if dtype==torch.bfloat16:
        ATOL = 1e-1 * max(1.0, (seqlen_q + seqlen_k) / 128.0)
    else:
        ATOL = 1e-2 * max(1.0, (seqlen_q + seqlen_k) / 128.0)
    # RTOL=1e-2 if dtype==torch.float16 else 5e-2
    RTOL=0.02
    print(f'Forward Using ATOL={ATOL} RTOL={RTOL}')
    # FIXME: Need to raise tolerance
    '''
    is_allclose = torch.allclose(ref_out, tri_out, atol=ATOL, rtol=RTOL)
    '''
    is_allclose = torch.allclose(TO(ref_out), tri_out, atol=ATOL, rtol=RTOL)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
    assert is_allclose, 'Forward pass {is_allclose=}'
    if dtype == torch.bfloat16:
        ATOL = 1e-1 * max(1.0, (RP(seqlen_q) + RP(seqlen_k) + RP(D_HEAD)) / 32.0)
    elif dtype == torch.float32:
        ATOL = 1e-3 * max(1.0, (RP(seqlen_q) + RP(seqlen_k) + RP(D_HEAD)) / 32.0)
    else:
        ATOL = 1e-2 * max(1.0, (RP(seqlen_q) + RP(seqlen_k) + RP(D_HEAD)) / 32.0)
    print(f'Backward Using ATOL={ATOL} RTOL={RTOL}')

    dv_allclose = SKIP_DK_DV or torch.allclose(TO(ref_dv), tri_dv, atol=ATOL, rtol=RTOL)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{dropout_mask.shape=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{torch.isnan(ref_dv).any()=}')
        '''
        any_nan = torch.isnan(ref_dv).any()
        if any_nan:
            torch.set_printoptions(linewidth=200)
            print(f'{q=}')
            print(f'{k=}')
            print(f'{v=}')
            print(f'{dropout_p=}')
            print(f'{causal=}')
            print(f'{sm_scale=}')
        '''
        if seqlen_q <= 16 or True:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dk[0,0]=}')
            print(f'{ref_dk[0,0]=}')
            print(f'{tri_dv[0,0]=}')
            print(f'{ref_dv[0,0]=}')
            # print(f'{tri_dq[0,0]=}')
            # print(f'{ref_dq[0,0]=}')

    dk_allclose = SKIP_DK_DV or torch.allclose(TO(ref_dk), tri_dk, atol=ATOL, rtol=RTOL)
    if dv_allclose and not dk_allclose:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

    dq_allclose = SKIP_DQ or torch.allclose(TO(ref_dq), tri_dq, atol=ATOL, rtol=RTOL)
    if dk_allclose and dv_allclose and not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')
    assert dk_allclose and dv_allclose and dq_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=}'

# @pytest.mark.parametrize('BATCH', [1])
# @pytest.mark.parametrize('N_HEADS', [1])
@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [1, 4])
# @pytest.mark.parametrize('D_HEAD', [16, 32, 64, 128, 256])
# Irregular-only PyTorch set
# @pytest.mark.parametrize('D_HEAD', [8, 21, 72, 96, 160, 192, 203])
# @pytest.mark.parametrize('seqlen_q', [1, 4, 32, 128, 256, 512, 1024, 7, 394, 250, 399, 511, 1019])
# @pytest.mark.parametrize('seqlen_k', [1, 4, 32, 128, 256, 512, 1024, 3, 217, 339, 313, 491, 988])
# PyTorch set
@pytest.mark.parametrize('D_HEAD', [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256])
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# Currently debugging
# @pytest.mark.parametrize('D_HEAD', range(8,128+1,4))
# @pytest.mark.parametrize('D_HEAD', [84,92,108, 203] + list(range(128, 256+1, 4)))
# @pytest.mark.parametrize('D_HEAD', [84,203])
# @pytest.mark.parametrize('D_HEAD', range(8,64+1,4))
# @pytest.mark.parametrize('seqlen_q', [128, 2048, 4096])
# @pytest.mark.parametrize('seqlen_k', [128, 2048, 4096])
# Minimal set
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16])
# @pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('sm_scale', [1.2])
# @pytest.mark.parametrize('storage_flip', [False])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 2, 4])
@pytest.mark.parametrize('N_HEADS', [1, 2, 4])
@pytest.mark.parametrize('D_HEAD', [16,32,64,128,256])
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [128, 113])
# @pytest.mark.parametrize('seqlen_k', [128, 79])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_bwd_with_matrix_bias(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
