#!/usr/bin/env python

import pytest
import torch
import triton
import triton.language as tl

from fused_attention_trimmed import attn_fwd

VERBOSE=False

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, split_kernel=False):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        stage = 3 if causal else 1
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[0] * q.shape[1],
            1
        )
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if VERBOSE:
            print(f'{q.shape=}')
            print(f'{k.shape=}')
            print(f'{v.shape=}')
            print(f'{o.shape=}')
            print(f'{q.data_ptr()=:x}')
            print(f'{k.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{M.data_ptr()=:x}')
            print(f'{o.data_ptr()=:x}')
            print(f'{stage=}')
            print(f'seqlen_q={q.shape[2]}')
            print(f'seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')

        attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            seqlen_q=q.shape[2],
            seqlen_k=k.shape[2],
            BLOCK_DMODEL=Lk,
            STAGE=stage,
            BLOCK_M=min(128, q.shape[2], k.shape[2]), BLOCK_N=64, pre_load_v=False
        )

        '''
        ## restore the grid for bwd kernel
        best_config = attn_fwd.get_best_config(N_CTX = q.shape[2], STAGE = stage)
        block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.split_kernel = split_kernel
        '''
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplemented("backward")

attention = _attention.apply

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
@pytest.mark.parametrize('qseqlen_override', [None, 512])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, dtype, qseqlen_override):
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
    # sm_scale = 0.5
    sm_scale = 0.0
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
    import numpy as np
    err_idx = np.unravel_index(torch.argmax(torch.abs(out_ref - tri_out)).cpu().numpy(), out_ref.shape)
    if not is_allclose:
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
