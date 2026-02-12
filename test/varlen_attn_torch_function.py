#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from aotriton_flash import (
    attn_fwd_varlen,
    attn_bwd_varlen,
)
from attn_torch_function import AttentionExtraArgs, BWD_IMPL, V3_API

VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2

# Varlen now always use V3_API for full feature coverage
from aotriton_flash import lazy_dq_acc, lazy_delta

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

class _attention_varlen(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    # DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, seqlens_q, seqlens_k, causal, sm_scale, dropout_p,
                varlen_type,
                attn_extra_args=AttentionExtraArgs()):
        return_encoded_softmax = attn_extra_args.return_encoded_softmax
        autotune = attn_extra_args.autotune
        return_autotune = attn_extra_args.return_autotune
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk
        # assert Lk in {16, 32, 64, 128}
        if varlen_type == 'strided':
            seqlens_q, padlens_q = seqlens_q
            seqlens_k, padlens_k = seqlens_k
            total_seqlen_q = int(np.sum(seqlens_q + padlens_q))  # Be explicit
        else:
            total_seqlen_q = int(np.sum(seqlens_q))
        batch = len(seqlens_q)
        num_heads = q.shape[1]
        max_seqlen_q = int(np.max(seqlens_q))
        max_seqlen_k = int(np.max(seqlens_k))
        cu_seqlens_q = np.cumsum(seqlens_q)
        cu_seqlens_k = np.cumsum(seqlens_k)
        cu_seqlens_q = torch.tensor([0] + cu_seqlens_q.tolist(), dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.tensor([0] + cu_seqlens_k.tolist(), dtype=torch.int32, device=q.device)
        if varlen_type in ['compact', 'padded']:
            seq_strides_q = None
            seq_strides_k = None
        elif varlen_type == 'strided':
            seq_strides_q = np.cumsum(seqlens_q + padlens_q)
            seq_strides_k = np.cumsum(seqlens_k + padlens_k)
            seq_strides_q = torch.tensor([0] + seq_strides_q.tolist(), dtype=torch.int32, device=q.device)
            seq_strides_k = torch.tensor([0] + seq_strides_k.tolist(), dtype=torch.int32, device=q.device)
        else:
            assert False
        o = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]), device=q.device, dtype=q.dtype)
        b = torch.empty((0,0,0,0), device=q.device, dtype=q.dtype)

        if varlen_type == 'padded':
            M = torch.zeros((batch * num_heads, max_seqlen_q), device=q.device, dtype=torch.float32)
        else:
            M = torch.empty((num_heads, total_seqlen_q), device=q.device, dtype=torch.float32)
        if attn_extra_args.fillnan:
            for t in (o, M):
                t.fill_(float('nan'))
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((batch, num_heads, max_seqlen_q, max_seqlen_k), device=q.device, dtype=q.dtype)
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

        philox_null = torch.empty([0], device=q.device, dtype=torch.uint64)
        if dropout_p > 0.0:
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        else:
            philox_seed = philox_null
            philox_offset1 = philox_null
            philox_offset2 = 0
            philox_seed_output = philox_null
            philox_offset_output = philox_null

        if causal:
            atomic = torch.zeros([1], device=q.device, dtype=torch.int32)
        else:
            atomic = torch.empty([0], device=q.device, dtype=torch.int32)

        attn_fwd_varlen(q, k, v,
                        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        seq_strides_q, seq_strides_k,
                        b, sm_scale, M, o,
                        dropout_p, philox_seed, philox_offset1, philox_offset2,
                        philox_seed_output, philox_offset_output,
                        encoded_softmax, causal, atomic, varlen_type)

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.seqlens_q = seqlens_q
        ctx.seqlens_k = seqlens_k
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.seq_strides_q = seq_strides_q
        ctx.seq_strides_k = seq_strides_k
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed_output
        ctx.philox_offset = philox_offset_output
        ctx.philox_offset1 = philox_offset1
        ctx.philox_offset2 = philox_offset2
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.varlen_type = varlen_type
        ctx.attn_extra_args = attn_extra_args
        return o, encoded_softmax, None

    @staticmethod
    def backward(ctx, do, _, __):
        q, k, v, b, o, L = ctx.saved_tensors
        print(f'{b=}')
        seqlens_q = ctx.seqlens_q
        seqlens_k = ctx.seqlens_k
        cu_seqlens_q = ctx.cu_seqlens_q
        cu_seqlens_k = ctx.cu_seqlens_k
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        batch = len(seqlens_q)
        sm_scale = ctx.sm_scale
        dropout_p = ctx.dropout_p
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset
        causal = ctx.causal
        # if q.shape[-1] <= 32:
        # do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        if ctx.attn_extra_args.fillnan:
            for t in (dq, dk, dv, db):
                if t is not None:
                    t.fill_(float('nan'))
        delta = lazy_delta(L)
        attn_bwd_varlen(q, k, v,
                        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        ctx.seq_strides_q, ctx.seq_strides_k,
                        b, sm_scale, o, do, dq, dk, dv, db, L, delta,
                        dropout_p, philox_seed, philox_offset, 0, causal, ctx.varlen_type);
        return dq, dk, dv, None, None, None, None, None, None, None, None

varlen_attention = _attention_varlen.apply
