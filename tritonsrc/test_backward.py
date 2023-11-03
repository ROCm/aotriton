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
                          (1, 1, 64, 64),
                          (1, 1, 16, 16),
                          #(4, 48, 8192, 64),
                          #(4, 48, 16384, 64)
                          ])
@pytest.mark.parametrize('causal', [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [1.0, 0.5, 0.0])
@pytest.mark.parametrize('qseqlen_override', [None])
def test_op_bwd(Z, H, N_CTX, D_HEAD, causal, sm_scale, dtype, qseqlen_override):
    torch.manual_seed(20)
    qseqlen = N_CTX if qseqlen_override is None else qseqlen_override
    kseqlen = N_CTX
    '''
    q = torch.empty((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    '''
    q = torch.ones((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda") * 1.0
    k = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 2.0
    v = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 3.0
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    if causal == False:
        split_kernel = False
    else: # split kernel only handles for causal=True
        split_kernel = True
    '''
    dout = torch.randn_like(q)
    '''
    dout = torch.ones_like(q) * 0.5
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
    if False and q.shape[-2] <= 16 and q.shape[-1] <= 16:
        print(f'{tri_out[0][0][:][:]=}')
        print(f'{ref_out[0][0][:][:]=}')
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    assert torch.allclose(ref_dq, tri_dq, atol=5e-2, rtol=0)
    print(f'{tri_dv[0][0][:4, :4]=}')
    print(f'{ref_dv[0][0][:4, :4]=}')
    # print(f'{tri_dk[0][0][:4, :4]=}')
    # print(f'{ref_dk[0][0][:4, :4]=}')
    # print(f'{tri_dq[0][0][:4, :4]=}')
    # print(f'{ref_dq[0][0][:4, :4]=}')
    if False:
        qk_scale = sm_scale * 1.44269504
        qk = q[0,0] @ k[0,0].transpose(-1, -2)
        qk_bad = q[0,0] @ k[0,0]
        print(f'{q[0,0,0,:].dot(k[0,0,0,:])=}')
        print(f'{q[0,0][:4, :4]=}')
        print(f'{k[0,0][:4, :4]=}')
        print(f'Manual {qk[:4, :4]=}')
        print(f'Manual {qk_bad[:4, :4]=}')
        print(f'Triton qk {tri_dq[0,0][:4, :4]=}')
        # assert False
        # l_i = tl.load(l_ptrs + offs_m_curr)
        # p = tl.math.exp2(qk * qk_scale - l_i[:, None])
        # do = tl.load(do_ptrs)
        # dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
    if torch.version.hip is None:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        assert torch.allclose(ref_dv, tri_dv, atol=5e-2, rtol=0)
    assert torch.allclose(ref_dk, tri_dk, atol=5e-2, rtol=0)
