#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import queue
from torch.multiprocessing import Process
from aotriton_flash import attn_fwd, ipc_attn_fwd, attn_bwd, debug_fill_dropout_rng, ExtraArguments, hipError_t
from collections import namedtuple
from cpp_autotune import do_bench, cpp_autotune

AttentionExtraArgs = namedtuple('AttentionExtraArgs',
        ['return_encoded_softmax',
         'autotune',
         'return_autotune',
         'autotune_validator',
         'cpp_autotune_tqdm_position',
         'cpp_autotune_tqdm_prefix',
         'gpu_device',
         'tune_worker',
         ],
        defaults=[False, False, False, None, None, '', None, None])

VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET = 0x1D4B42

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
        o = torch.zeros_like(q)

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

        philox_seed = DEFAULT_PHILOX_SEED
        philox_offset = DEFAULT_PHILOX_OFFSET

        if autotune and return_autotune:
            assert attn_extra_args.autotune_validator is not None
            def sameprocess_func(extargs):
                args = (q, k, v, b, sm_scale, M, o,
                        dropout_p, philox_seed, philox_offset, encoded_softmax, causal,
                        extargs)
                try:
                    ret = attn_fwd(*args)
                except Exception as e:
                    print(e)
                    return hipError_t.hipErrorLaunchFailure, None
                return ret, (o,)
            def ipc_func(force_kernel_index):
                shard = attn_extra_args.gpu_device
                tune_worker = attn_extra_args.tune_worker
                def factory():
                    ipc_to_worker = torch.multiprocessing.Queue()
                    ipc_from_worker = torch.multiprocessing.Queue()
                    ipc_to_worker.cancel_join_thread()
                    ipc_from_worker.cancel_join_thread()
                    p = Process(target=ipc_attn_fwd,
                                args=(ipc_to_worker, ipc_from_worker))
                    p.start()
                    return (ipc_to_worker, ipc_from_worker, p)
                ipc_to_worker, ipc_from_worker, p = tune_worker.request_cached_gpukernel_process(ipc_attn_fwd, factory)
                # print(f'{q.data_ptr()=:x}')
                # print(f'{k.data_ptr()=:x}')
                # print(f'{v.data_ptr()=:x}')
                # print(f'{b.data_ptr()=:x}')
                # print(f'{M.data_ptr()=:x}')
                # print(f'{o.data_ptr()=:x}')
                ipc_to_worker.put((q, k, v, b, sm_scale, M, o,
                                   dropout_p, philox_seed, philox_offset, encoded_softmax, causal,
                                   force_kernel_index, shard))
                while p.is_alive():
                    try:
                        iret = ipc_from_worker.get(timeout=1)
                        break
                    except queue.Empty:
                        # print(f'Process timeout {p.is_alive()=}')
                        pass
                # print(f'Process attn_fwd starting')
                if not p.is_alive():
                    # print(f'Process exitcode {p.exitcode}')
                    tune_worker.invalid_gpukernel_process_cache(ipc_attn_fwd)
                    p.join()
                    ret = hipError_t.hipErrorLaunchFailure
                else:
                    ret = hipError_t.hipSuccess if iret == 0 else hipError_t.hipErrorLaunchFailure
                # print(f'Process attn_fwd joined')
                # print(f'Process exitcode {p.exitcode}')
                return ret, (o,)
            def func(extargs, is_testing):
                # print(f'{is_testing=}')
                if not is_testing:
                    return sameprocess_func(extargs)
                o.fill_(float('nan'))
                return ipc_func(extargs.force_kernel_index)
                # print(f'running attn_fwd with {extargs.force_kernel_index=}')
            tuning_result = cpp_autotune(ExtraArguments, func,
                                         attn_extra_args.autotune_validator,
                                         tqdm_position=attn_extra_args.cpp_autotune_tqdm_position,
                                         tqdm_prefix=attn_extra_args.cpp_autotune_tqdm_prefix)
        else:
            attn_fwd(q, k, v, b, sm_scale, M, o,
                     dropout_p, philox_seed, philox_offset, encoded_softmax, causal);
            tuning_result = None

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.tuning_result = [('attn_fwd', tuning_result)] if tuning_result is not None else None
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
                       dropout_p, philox_seed, philox_offset, causal);
        assert ret == hipError_t.hipSuccess, ret
        return dq, dk, dv, db, None, None, None, None, None

attention = _attention.apply
