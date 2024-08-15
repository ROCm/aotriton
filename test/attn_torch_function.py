#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import queue
from torch.multiprocessing import Process
from aotriton_flash import (
    attn_fwd,
    attn_bwd,
    debug_fill_dropout_rng,
    FwdExtraArguments,
    BwdExtraArguments,
    hipError_t,
)
from collections import namedtuple

AttentionExtraArgs = namedtuple('AttentionExtraArgs',
        ['return_encoded_softmax',
         'autotune',
         'return_autotune',
         'is_testing'
         ],
        defaults=[False, False, False, True])

VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

class _attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    # DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, b, causal, sm_scale, dropout_p,
                attn_extra_args=AttentionExtraArgs()):
        return_encoded_softmax, autotune, return_autotune = attn_extra_args[:3]
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # assert Lk in {16, 32, 64, 128}
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        o = torch.empty_like(q)

        # def round_to_16x(x):
        #     return ((x + 15) // 16) * 16
        # M_padded = torch.empty((q.shape[0] * q.shape[1], round_to_16x(q.shape[2])), device=q.device, dtype=torch.float32)
        # M = M_padded[:,:q.shape[2]]
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=q.dtype)
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
            print(f'{stage=}')
            print(f'seqlen_q={q.shape[2]}')
            print(f'seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_null = torch.empty([0], device=q.device, dtype=torch.uint64)
        assert philox_null.data_ptr() == 0
        philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
        philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint32)
        philox_offset2 = DEFAULT_PHILOX_OFFSET_2

        # Check GPU kernel accepts nullptr for philox_*_output
        if attn_extra_args.is_testing:
            attn_fwd(q, k, v, b, sm_scale, M, o,
                     dropout_p, philox_seed, philox_offset1, philox_offset2,
                     philox_null, philox_null,
                     encoded_softmax, causal)

        philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        attn_fwd(q, k, v, b, sm_scale, M, o,
                 dropout_p, philox_seed, philox_offset1, philox_offset2,
                 philox_seed_output, philox_offset_output,
                 encoded_softmax, causal)
        tuning_result = None

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed_output
        ctx.philox_offset = philox_offset_output
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.tuning_result = [('attn_fwd', tuning_result)] if tuning_result is not None else None
        ctx.fwd_tuning_result = tuning_result
        ctx.attn_extra_args = attn_extra_args
        ctx.autotune = autotune
        ctx.return_autotune = return_autotune
        assert not torch.isnan(M).any()
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward(ctx, do, _, __):
        q, k, v, b, o, L = ctx.saved_tensors
        # print(f'{b=}')
        sm_scale = ctx.sm_scale
        dropout_p = ctx.dropout_p
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset
        causal = ctx.causal
        attn_extra_args = ctx.attn_extra_args
        autotune = ctx.autotune
        return_autotune = ctx.return_autotune
        # if q.shape[-1] <= 32:
        # do = do.contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        delta = torch.empty_like(L)
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]

        ret = attn_bwd(q, k, v, b, sm_scale, o, do, dq, dk, dv, db, L, delta,
                       dropout_p, philox_seed, philox_offset, 0, causal)
        assert ret == hipError_t.hipSuccess, ret
        tuning_result = None

        if tuning_result is not None:
            ctx.tuning_result += tuning_result

        assert not torch.isnan(delta).any()
        return dq, dk, dv, db, None, None, None, None, None

attention = _attention.apply
