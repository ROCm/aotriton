#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import numpy as np
import torch
import triton
import triton.language as tl
from flash import (
    attn_fwd_varlen as bare_attn_fwd_varlen,
)
from attn_torch_function import DEFAULT_PHILOX_SEED, DEFAULT_PHILOX_OFFSET

VERBOSE = True

class _varlen_attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, seqlens_q, seqlens_k, causal, sm_scale, dropout_p, return_encoded_softmax,
                autotune=False, return_autotune=False):
        dtype = q.dtype
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        head_dim_rounded = 2 ** (Lk - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != Lk
        # Varlen packed all batches of seqlens into dim[0]
        batch = len(seqlens_q)
        num_heads = q.shape[1]
        max_seqlen_q = int(np.max(seqlens_q))
        max_seqlen_k = int(np.max(seqlens_k))
        cu_seqlens_q = torch.tensor([0] + np.cumsum(seqlens_q).tolist(), dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.tensor([0] + np.cumsum(seqlens_k).tolist(), dtype=torch.int32, device=q.device)
        o = torch.zeros_like(q)

        grid = lambda META: (
            triton.cdiv(max_seqlen_q, META['BLOCK_M']),
            num_heads,
            batch,
        )
        # Fixed M is (Batch, num_heads, seqlen)
        # Varlen M then will be (batch, num_heads, max_seqlen_q)
        # TODO: Fix the kernel
        M = torch.empty((batch, num_heads, max_seqlen_q), device=q.device, dtype=torch.float32)
        if return_encoded_softmax:
            encoded_softmax = torch.ones((batch, num_heads, max_seqlen_q, max_seqlen_k),
                    device=q.device,
                    dtype=_varlen_attention.DEBUG_MASK_DTYPE) * 114.514
            print(f'{encoded_softmax.shape=}')
        else:
            encoded_softmax = None
        if False or VERBOSE:
            print(f'{q.shape=}')
            print(f'{k.shape=}')
            print(f'{v.shape=}')
            print(f'{o.shape=}')
            print(f'{q.data_ptr()=:x}')
            print(f'{k.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{M.data_ptr()=:x}')
            print(f'{o.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_seed = DEFAULT_PHILOX_SEED
        philox_offset = DEFAULT_PHILOX_OFFSET

        use_small_block = dropout_p > 0.0 or return_encoded_softmax
        use_medium_block = False # reserved
        if use_small_block:
            BLOCK_M = 64
            BLOCK_N = 32
        elif use_medium_block:
            BLOCK_M = 64
            BLOCK_N = 64
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if dtype == torch.float32:
            BLOCK_M //= 2

        if autotune:
            assert False, "Prototype, won't test autotune for now"
            tuned_attn_fwd[grid](
                q, k, v, b, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(1), o.stride(2),
                seqlen_q=q.shape[2],
                seqlen_k=k.shape[2],
                head_dim=Lk,
                dropout_p=dropout_p,
                philox_seed=philox_seed,
                philox_offset_base=philox_offset,
                encoded_softmax=encoded_softmax,
                CAUSAL=causal,
                BLOCK_DMODEL=head_dim_rounded,
                ENABLE_DROPOUT=dropout_p > 0.0,
                RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
                PADDED_HEAD=padded_head,
            )
        else:
            RETURN_ENCODED_SOFTMAX=encoded_softmax is not None
            # DEBUG
            BLOCK_M = BLOCK_N = 16
            print(f'{BLOCK_M=} {BLOCK_N=} {RETURN_ENCODED_SOFTMAX=} seqlens_q={seqlens_q} seqlens_k={seqlens_k} {cu_seqlens_q=} {cu_seqlens_k=} {q.shape=} {q.stride()=}',
                    flush=True)
            bare_attn_fwd_varlen[grid](
                q, k, v, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(1), o.stride(2),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                head_dim=Lk,
                dropout_p=dropout_p,
                philox_seed=philox_seed,
                philox_offset_base=philox_offset,
                encoded_softmax=encoded_softmax,
                CAUSAL=causal,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=head_dim_rounded,
                BLOCK_N=BLOCK_N,
                pre_load_v=False,
                ENABLE_DROPOUT=dropout_p > 0.0,
                RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
                PADDED_HEAD=padded_head,
                num_stages=1,
            )

        ctx.autotune = autotune
        ctx.return_autotune = return_autotune
        if autotune and return_autotune:
            ## restore the grid for bwd kernel
            best_config = tuned_attn_fwd.get_best_config()
            tuning_result = copy.deepcopy(best_config)
            block_m = int(best_config.kwargs['BLOCK_M'])
        else:
            tuning_result = None
            block_m = min(128, q.shape[2], k.shape[2])
        grid = (triton.cdiv(max_seqlen_q, block_m), num_heads, batch)
        print(f'{grid=}')
        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.head_dim = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.tuning_result = [('attn_fwd_varlen', tuning_result)] if tuning_result is not None else None
        if ctx.tuning_result is not None:
            for kernel_name, best in ctx.tuning_result:
                print(f'{kernel_name=} {best.kwargs=} {best.num_warps=} {best.num_stages=}')
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward(ctx, do, _, fwd_tuning_result):
        pass

varlen_attention = _varlen_attention.apply
