#!/usr/bin/env python

import pytest
import torch

from attn_torch_function import attention

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

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          (4, 48, 4096, 64),
                          (8, 4, 256, 16),
                          (8, 4, 64, 16),
                          (8, 4, 256, 64),
                          (1, 1, 128, 64),
                          #(4, 48, 8192, 64),
                          #(4, 48, 16384, 64)
                          ])
# @pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('causal', [False]) # split kernel only handles for causal=False
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.5, 0.0])
@pytest.mark.parametrize('qseqlen_override', [None])
def test_op_bwd(Z, H, N_CTX, D_HEAD, causal, sm_scale, dtype, qseqlen_override):
    torch.manual_seed(20)
    qseqlen = N_CTX if qseqlen_override is None else qseqlen_override
    kseqlen = N_CTX
    q = torch.empty((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0.5
    split_kernel = True
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((qseqlen, kseqlen), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale, split_kernel)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        assert torch.allclose(ref_dv, tri_dv, atol=5e-2, rtol=0)
    assert torch.allclose(ref_dk, tri_dk, atol=5e-2, rtol=0)
    assert torch.allclose(ref_dq, tri_dq, atol=5e-2, rtol=0)
