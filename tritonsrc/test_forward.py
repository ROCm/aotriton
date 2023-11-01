#!/usr/bin/env python

import pytest
import torch

from forward_torch_function import attention

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
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.5, 0.0])
@pytest.mark.parametrize('qseqlen_override', [None, 512])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, sm_scale, dtype, qseqlen_override):
    torch.manual_seed(20)
    qseqlen = N_CTX if qseqlen_override is None else qseqlen_override
    kseqlen = N_CTX
    print(f"test_op_fwd {Z=}, {H=}, {qseqlen=}, {kseqlen=}, {D_HEAD=}, {causal=}")
    q = (
        torch.empty((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        .requires_grad_()
    )
    '''
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    '''
    # triton implementation
    out_ref = torch.ops.aten._scaled_dot_product_attention_math(q, k, v, dropout_p=0.0, is_causal=causal, scale=sm_scale)[0]
    tri_out = attention(q, k, v, causal, sm_scale)
    print(f'{q.shape=} {q.stride()=}')
    print(f'{k.shape=} {k.stride()=}')
    print(f'{v.shape=} {v.stride()=}')
    is_allclose = torch.allclose(out_ref, tri_out, atol=1e-2, rtol=0)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(out_ref - tri_out)).cpu().numpy(), out_ref.shape)
        print(f'{tri_out[0][0][0][:]=}')
        print(f'{out_ref[0][0][0][:]=}')
        print(f'{tri_out[-1][0][0][:]=}')
        print(f'{out_ref[-1][0][0][:]=}')
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{out_ref[err_idx]=}')
        # tri_cpu = tri_out[0, 0].cpu().detach().numpy()
        # print(f'{tri_cpu.shape=}')
    # compare
    assert is_allclose
