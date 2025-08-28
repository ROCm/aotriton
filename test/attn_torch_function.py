#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import torch
import queue
from torch.multiprocessing import Process
from aotriton_flash import IGNORE_BACKWARD_IMPORT
from aotriton_flash import (
    attn_fwd,
    debug_simulate_encoded_softmax,
    FwdExtraArguments,
    hipError_t,
    hipGetLastError,
    AOTRITON_TORCH_ONLY_USE_CPU,
    HipMemory,
    attn_options,
)
if not IGNORE_BACKWARD_IMPORT:
    from aotriton_flash import (
        attn_bwd,
        attn_bwd_fused,
        BwdExtraArguments,
        FusedBwdExtraArguments,
        attn_bwd_aiter,
    )
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

BWD_IMPL = int(os.getenv('BWD_IMPL', default='0'))
V3_API = bool(int(os.getenv('V3_API', default='0')))
if BWD_IMPL == 2:
    PROBE_UNSUPPORTED = bool(int(os.getenv('PROBE_UNSUPPORTED', default='0')))
else:
    PROBE_UNSUPPORTED = False

if BWD_IMPL == 2 or V3_API:
    from aotriton_flash import lazy_dq_acc, lazy_delta
else:
    def lazy_dq_acc(dq):
        return None
    def lazy_delta(L):
        return torch.empty_like(L)

FORCE_BWD_BACKEND = V3_API and (os.getenv('BWD_IMPL', default=None) is not None)

def empty_handler():
    pass

@dataclass
class AttentionExtraArgs:
    return_encoded_softmax : bool = False
    autotune : bool = False
    return_autotune : bool = False
    is_testing : bool = True
    fillnan : bool = False
    return_logsumexp : bool = False
    illaddr_handler : Callable = empty_handler

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
        return_encoded_softmax = attn_extra_args.return_encoded_softmax
        autotune = attn_extra_args.autotune
        return_autotune = attn_extra_args.return_autotune
        if return_autotune and attn_extra_args.return_logsumexp:
            assert False, 'Cannot set return_autotune and return_logsumexp at the same time. Both are returned as 3rd value'
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
        if attn_extra_args.fillnan:
            for t in (o, M):
                t.fill_(float('nan'))
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
        if dropout_p > 0.0:
            assert philox_null.data_ptr() == 0
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            assert philox_seed_output.data_ptr() != 0
            assert philox_offset_output.data_ptr() != 0
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

        # print(f'{attn_extra_args=}')
        # Check GPU kernel accepts nullptr for philox_*_output
        if attn_extra_args.is_testing:
            ret = attn_fwd(q, k, v, b, sm_scale, M, o,
                           dropout_p, philox_seed, philox_offset1, philox_offset2,
                           philox_null, philox_null,
                           encoded_softmax, causal, atomic, call_operator=V3_API)
            assert ret == hipError_t.hipSuccess, ret

        ret = attn_fwd(q, k, v, b, sm_scale, M, o,
                       dropout_p, philox_seed, philox_offset1, philox_offset2,
                       philox_seed_output, philox_offset_output,
                       encoded_softmax, causal, atomic, call_operator=V3_API)
        if attn_extra_args.is_testing:
            try:
                torch.cuda.synchronize()
            except:
                pass
            last_err = hipGetLastError()
            if last_err == hipError_t.hipErrorIllegalAddress:
                attn_extra_args.illaddr_handler()
            assert last_err == hipError_t.hipSuccess, last_err
        else:
            assert ret == hipError_t.hipSuccess, ret
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
        if attn_extra_args.is_testing:
            assert not torch.isnan(M).any(), f'L tensor has NaN'
        ret3 = M if attn_extra_args.return_logsumexp else ctx.tuning_result
        return o, encoded_softmax, ret3

    @staticmethod
    def backward_split(ctx, do, _, __):
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
        dq_acc = lazy_dq_acc(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        delta = lazy_delta(L)
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        if FORCE_BWD_BACKEND:
            extargs = attn_options()
            extargs.force_backend_index = BWD_IMPL
        else:
            extargs = None

        ret = attn_bwd(q, k, v, b, sm_scale, o, do, dq, dk, dv, db, dq_acc, L, delta,
                       dropout_p, philox_seed, philox_offset, 0, causal,
                       extargs=extargs, call_operator=V3_API)
        if PROBE_UNSUPPORTED and ret == hipError_t.hipErrorPeerAccessUnsupported:
            raise NotImplementedError()
        assert ret == hipError_t.hipSuccess, ret
        tuning_result = None

        if tuning_result is not None:
            ctx.tuning_result += tuning_result

        # fused bwd does not need delta
        # TODO: Make delta lazy tensor
        if not V3_API and attn_extra_args.is_testing:
            assert not torch.isnan(delta).any(), f'{delta=}'
        return dq, dk, dv, db, None, None, None, None, None

    @staticmethod
    def backward_fused(ctx, do, _, __):
        # print("runing backward_fuse")
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
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]

        assert not V3_API, 'attn_bwd_fused is not exposed in V3 API'
        ret = attn_bwd_fused(q, k, v, b, sm_scale, o, do, dq, dk, dv, db, L,
                             dropout_p, philox_seed, philox_offset, 0, causal)
        assert ret == hipError_t.hipSuccess, ret
        tuning_result = None

        if tuning_result is not None:
            ctx.tuning_result += tuning_result

        return dq, dk, dv, db, None, None, None, None, None

    @staticmethod
    def backward_aiter(ctx, do, _, __):
        # print("runing backward_aiter")
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

        # don't do zeros_like, dq_acc only supports BSHD
        # dq_acc = torch.zeros(*q.shape, dtype=torch.float32, device=q.device)
        dq = torch.empty_like(q)
        dq_acc = lazy_dq_acc(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        delta = lazy_delta(L)
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]

        assert not V3_API, 'attn_bwd_fused is not exposed in V3 API'
        ret = attn_bwd_aiter(q, k, v, b, sm_scale, o, do, dq, dk, dv, db, dq_acc, L, delta,
                             dropout_p, philox_seed, philox_offset, 0, causal)
        if PROBE_UNSUPPORTED and ret == hipError_t.hipErrorPeerAccessUnsupported:
            raise NotImplementedError()
        assert ret == hipError_t.hipSuccess, ret
        tuning_result = None

        if tuning_result is not None:
            ctx.tuning_result += tuning_result

        return dq, dk, dv, db, None, None, None, None, None
    backward = backward_split if BWD_IMPL == 0 or V3_API else (backward_fused if BWD_IMPL == 1 else backward_aiter)

attention = _attention.apply
