import pytest
import torch
import ctypes

from pyaotriton.v2.flash import attn_fwd as fa_forward, attn_bwd as fa_backward
from pyaotriton import T1, T2, T4, DType, Stream

def cast_dtype(dtype):
    assert not dtype.is_complex
    bits = dtype.itemsize * 8
    if dtype.is_floating_point:
        maintype = 'Float' if 'bfloat' not in str(dtype) else 'BFloat'
    else:
        maintype = 'Int' if 'uint' not in str(dtype) else 'UInt'
    typename = f'k{maintype}{bits}'
    return getattr(DType, typename)

def mk_aotensor(q, if_empty_then_like=None):
    rank = len(q.shape) if q is not None else len(if_empty_then_like.shape)
    if rank == 1:
        klass = T1
    elif rank == 2:
        klass = T2
    elif rank == 4:
        klass = T4
    else:
        assert False, f'Unsupported tensor rank {rank}, shape {q.shape}'
    if q is None:
        return klass(0, [0] * rank, [1] * rank, cast_dtype(if_empty_then_like.dtype))
    return klass(q.data_ptr(), tuple(q.size()), q.stride(), cast_dtype(q.dtype))

def aotrition_attn_fwd(q, k, v, sm_scale, M, o,
                       dropout_p, philox_seed, philox_offset, encoded_softmax, is_causal):
    err = fa_forward(mk_aotensor(q),
                     mk_aotensor(k),
                     mk_aotensor(v),
                     float(sm_scale),
                     mk_aotensor(M),
                     mk_aotensor(o),
                     float(dropout_p),
                     int(philox_seed),
                     int(philox_offset),
                     mk_aotensor(encoded_softmax, if_empty_then_like=q),
                     is_causal,
                     Stream(torch.cuda.current_stream().cuda_stream))
    print(f'{err=}')

def aotrition_attn_bwd(q, k, v, sm_scale, o, dout, dq, dk, dv, L, delta,
                       dropout_p, philox_seed, philox_offset, is_causal):
    err = fa_backward(mk_aotensor(q),
                      mk_aotensor(k),
                      mk_aotensor(v),
                      float(sm_scale),
                      mk_aotensor(o),
                      mk_aotensor(dout),
                      mk_aotensor(dq),
                      mk_aotensor(dk),
                      mk_aotensor(dv),
                      mk_aotensor(L),
                      mk_aotensor(delta),
                      float(dropout_p),
                      int(philox_seed),
                      int(philox_offset),
                      is_causal,
                      Stream(torch.cuda.current_stream().cuda_stream))
                      #Stream(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
    print(f'{err=}')


@pytest.mark.parametrize('BATCH', [1, 2, 4])
@pytest.mark.parametrize('N_HEADS', [1, 2, 4])
@pytest.mark.parametrize('D_HEAD', [16,32,64,128,256])
@pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
@pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [True, False])
def test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    torch.manual_seed(20)
    print(f"test_op_fwd {BATCH=}, {N_HEADS=}, {seqlen_q=}, {seqlen_k=}, {D_HEAD=}, {causal=}")
    SPARSE_HEAD_SINCE = 3
    SPARSE_SEQ_SINCE = 3
    Z = BATCH
    H = N_HEADS
    qdims = (BATCH, N_HEADS, seqlen_q, D_HEAD)
    kdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
    vdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
    if storage_flip:
        qdims = (qdims[0], qdims[2], qdims[1], qdims[3])
        kdims = (kdims[0], kdims[2], kdims[1], kdims[3])
        vdims = (vdims[0], vdims[2], vdims[1], vdims[3])
    q = (
        torch.empty(qdims, dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty(kdims, dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty(vdims, dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    o = torch.zeros_like(q)
    o2 = torch.zeros_like(q)
    if storage_flip:
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        o = torch.transpose(o, 1, 2)
        o2 = torch.transpose(o2, 1, 2)
        assert q.shape == (BATCH, N_HEADS, seqlen_q, D_HEAD)
        assert k.shape == (BATCH, N_HEADS, seqlen_k, D_HEAD)
        assert v.shape == (BATCH, N_HEADS, seqlen_k, D_HEAD)
        assert o.shape == (BATCH, N_HEADS, seqlen_q, D_HEAD)
        assert o2.shape == (BATCH, N_HEADS, seqlen_q, D_HEAD)
    return_encoded_softmax = dropout_p > 0.0
    M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    M2 = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    philox_seed = 114514
    philox_offset = 1919810
    if return_encoded_softmax:
        encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=q.dtype)
    else:
        encoded_softmax = None
    aotrition_attn_fwd(q, 
                       k, 
                       v, 
                       sm_scale, 
                       M, 
                       o, 
                       dropout_p,
                       philox_seed,
                       philox_offset,
                       encoded_softmax,
                       causal)
    aotrition_attn_fwd(q, 
                       k, 
                       v, 
                       sm_scale, 
                       M2, 
                       o2, 
                       dropout_p,
                       philox_seed,
                       philox_offset,
                       None,
                       causal)

    is_allclose_o1_o2 = torch.allclose(o, o2, atol=0, rtol=0)
    assert is_allclose_o1_o2

    dropout_mask = encoded_softmax > 0 if encoded_softmax is not None else None
    # assert torch.allclose(dropout_mask, dropout_mask_naive)
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    tri_out = o
    if dtype==torch.bfloat16:
        ATOL = 1e-1 * (seqlen_q / 128.0) if seqlen_q >= 16 else 1e-1
    else:
        ATOL = 2e-2 * (seqlen_q / 128.0) if seqlen_q >= 16 else 1e-2
    print(f'Using ATOL={ATOL}')
    is_allclose = torch.allclose(ref_out, tri_out, atol=ATOL, rtol=0)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=} {ref_out[err_idx]=} error: {tri_out[err_idx] - ref_out[err_idx]}')
    assert is_allclose


