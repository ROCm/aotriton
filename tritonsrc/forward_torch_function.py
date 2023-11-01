#!/usr/bin/env python

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
