#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import numpy as np
import torch
import triton
import triton.language as tl
from flash import (
    attn_fwd as bare_attn_fwd,
    bwd_preprocess_varlen as bare_bwd_preprocess_varlen,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq
)
from attn_torch_function import (
        DEFAULT_PHILOX_SEED,
        DEFAULT_PHILOX_OFFSET,
        tuned_attn_fwd
)

VERBOSE = True

class _varlen_attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    # TODO: rename seqlen_q -> seqlens_q
    @staticmethod
    def forward(ctx, q, k, v, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax,
                autotune=False, return_autotune=False):
        dtype = q.dtype
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        head_dim_rounded = 2 ** (Lk - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != Lk
        # Varlen packed all batches of seqlens into dim[0]
        batch = len(seqlen_q)
        num_heads = q.shape[1]
        max_seqlen_q = int(np.max(seqlen_q))
        max_seqlen_k = int(np.max(seqlen_k))
        cu_seqlens_q = torch.tensor([0] + np.cumsum(seqlen_q).tolist(), dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.tensor([0] + np.cumsum(seqlen_k).tolist(), dtype=torch.int32, device=q.device)
        o = torch.zeros_like(q)

        grid = lambda META: (
            triton.cdiv(max_seqlen_q, META['BLOCK_M']),
            num_heads,
            batch,
        )
        # Fixed M is (Batch, num_heads, seqlen)
        # Varlen M then will be (batch, num_heads, max_seqlen_q)
        # TODO: Ensure this tensor follows PyTorch's convention
        M = torch.zeros((batch, num_heads, max_seqlen_q), device=q.device, dtype=torch.float32)
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
        b = torch.empty((0,0,0,0), device=q.device, dtype=q.dtype)
        BIAS_TYPE = 0

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
            print(f'{BLOCK_M=} {BLOCK_N=} {RETURN_ENCODED_SOFTMAX=} seqlen_q={seqlen_q} seqlen_k={seqlen_k} {cu_seqlens_q=} {cu_seqlens_k=} {q.shape=} {q.stride()=}',
                    flush=True)
            bare_attn_fwd[grid](
                q, k, v, b, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                num_seqlens=len(cu_seqlens_q),
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
                BIAS_TYPE=BIAS_TYPE,
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
        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.seqlen_q = seqlen_q
        ctx.seqlen_k = seqlen_k
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.head_dim = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.bias_type = 0
        ctx.tuning_result = [('attn_fwd_varlen', tuning_result)] if tuning_result is not None else None
        if ctx.tuning_result is not None:
            for kernel_name, best in ctx.tuning_result:
                print(f'{kernel_name=} {best.kwargs=} {best.num_warps=} {best.num_stages=}')
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward(ctx, do, _, fwd_tuning_result):
        q, k, v, b, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv and Lk == ctx.head_dim
        head_dim_rounded = 2 ** (ctx.head_dim - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != ctx.head_dim

        seqlen_q = ctx.seqlen_q
        seqlen_k = ctx.seqlen_k
        cu_seqlens_q = ctx.cu_seqlens_q
        cu_seqlens_k = ctx.cu_seqlens_k
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        batch = len(seqlen_q)
        num_heads = q.shape[1]

        dq = torch.empty_like(q)
        dq.fill_(float('nan'))
        dk = torch.empty_like(k)
        dk.fill_(float('nan'))
        dv = torch.empty_like(v)
        dv.fill_(float('nan'))
        db = torch.empty_like(b)
        delta = torch.empty_like(L)
        MAX_BLOCK = 64 if ctx.dropout_p == 0 else 16
        # BLOCK = min(max_seqlen_q, max_seqlen_k, q.shape[-1], MAX_BLOCK)
        # BLOCK = BLOCK if is_supported_by_tl_dot(max_seqlen_q) and is_supported_by_tl_dot(max_seqlen_k) else 1
        if not ctx.autotune:
            BLOCK = 16 # FIXME: Variable block size
        else:
            BLOCK = 128
        return_autotune = ctx.tuning_result is not None

        grid_prep = (triton.cdiv(max_seqlen_q, BLOCK), num_heads, batch)
        bare_bwd_preprocess_varlen[grid_prep](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            head_dim=Lk,
            BLOCK_M=BLOCK, D_HEAD=head_dim_rounded,
            PADDED_HEAD=padded_head,
        )
        if False or VERBOSE:
            print(f'{q.shape=} {q.stride()=}')
            print(f'{k.shape=} {k.stride()=}')
            print(f'{v.shape=} {v.stride()=}')
            print(f'{o.shape=} {o.stride()=}')
            print(f'{dq.shape=} {dq.stride()=}')
            print(f'{dk.shape=} {dk.stride()=}')
            print(f'{dv.shape=} {dv.stride()=}')
            print(f'{do.shape=} {do.stride()=}')
            print(f'{L.shape=} {L=}')
            for i in range(num_heads):
                print(f'L[:,{i},:]={L[:,i,:]}')
            print(f'{delta.shape=} {delta=}')
            print(f'{BLOCK=}')

        use_small_block = ctx.dropout_p > 0.0
        use_medium_block = ctx.bias_type != 0
        if use_small_block:
            # DQ_BLOCK_M = min(max_seqlen_q, BLOCK)
            BLOCK_M = 32
            BLOCK_N = 16
        elif use_medium_block:
            BLOCK_M = 64
            BLOCK_N = 32
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if q.dtype == torch.float32:
            BLOCK_M = max(16, BLOCK_M // 2)
            BLOCK_N = max(16, BLOCK_N // 2)
        # debug_mask = torch.zeros((q.shape[0], q.shape[1], max_seqlen_q, max_seqlen_k), device=q.device, dtype=ctx.encoded_softmax.dtype)
        grid_dk_dv = lambda META: (
            triton.cdiv(max_seqlen_k, META['BLOCK_N']),
            num_heads,
            batch,
        )
        stride_dbz, stride_dbh, stride_dbm, stride_dbn = db.stride()
        if db.numel() == 0 or not b.requires_grad:
            # Passing all zeros to indicate no elements
            stride_dbz, stride_dbh, stride_dbm, stride_dbn = 0,0,0,0
        else:
            db.fill_(float('nan'))
        print(f'backward {ctx.bias_type=} {ctx.autotune=} {BLOCK_M=} {BLOCK_N=} {stride_dbz=} {stride_dbh=} {stride_dbm=} {stride_dbn=}')
        if k.requires_grad and v.requires_grad:
            if ctx.autotune:
                sized_tuned_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    num_seqlens=len(cu_seqlens_q),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                if return_autotune:
                    dkdv_best_config = copy.deepcopy(sized_tuned_bwd_kernel_dk_dv.get_best_config())
                    # BLOCK_M/N are missing with sized_tuned_bwd_kernel_*
                    dkdv_best_config.kwargs['BLOCK_M'] = BLOCK_M
                    dkdv_best_config.kwargs['BLOCK_N'] = BLOCK_N
                    tuning_result = copy.deepcopy(dkdv_best_config)
                    """
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'max_seqlen_q': max_seqlen_q,
                        'max_seqlen_k': max_seqlen_k,
                        'head_dim' : ctx.BLOCK_DMODEL,
                        'BLOCK_DMODEL' : head_dim_rounded,
                        'CAUSAL'  : ctx.causal,
                        'ENABLE_DROPOUT' : ctx.dropout_p > 0.0,
                    }
                    tuned_kernel = dict(dkdv_best_config.kwargs)
                    compiler_options = {
                        'num_warps' : dkdv_best_config.num_warps,
                        'num_stages': dkdv_best_config.num_stages,
                    }
                    tuning_result = {
                        'kernel_name' : 'bwd_kernel_dk_dv',
                        'inputs' : inputs,
                        'tuned_kernel' : tuned_kernel,
                        'compiler_options' : compiler_options,
                    }
                    """
                    ctx.tuning_result.append(('bwd_kernel_dk_dv', tuning_result))
                    print(f'{id(ctx.tuning_result)=}')
            else:
                bare_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    num_seqlens=len(cu_seqlens_q),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    # debug_mask=debug_mask,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    num_warps=4,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
        # mask_allclose = torch.allclose(debug_mask < 0, ctx.encoded_softmax < 0)
        if False:
            mask_allclose = torch.allclose(torch.abs(debug_mask), torch.abs(ctx.encoded_softmax)) # Stores QK
            if not mask_allclose:
                torch.set_printoptions(linewidth=200, threshold=2000)
                import sys
                print(f'bwd mask: {torch.abs(debug_mask[:,:,:2,16:])}')
                print(f'fwd mask: {torch.abs(ctx.encoded_softmax[:,:,:2,16:])}')
                print(f'Full bwd mask: {debug_mask[0,0]}')
                print(f'Full fwd mask: {ctx.encoded_softmax[0,0]}')
                print(f'Full mask div: {debug_mask[0,0] / ctx.encoded_softmax[0,0]}')
                print(f'Full dv: {dv}')
                if max_seqlen_q == 32:
                    print(f'2nd block bwd mask: {debug_mask[0,0, 16:]}')
                    print(f'2nd block fwd mask: {ctx.encoded_softmax[0,0, 16:]}')
            # print(f'Full q: {q}', file=sys.stderr)
            # assert mask_allclose
        grid_dq = lambda META: (
            triton.cdiv(max_seqlen_q, META['BLOCK_M']),
            num_heads,
            batch,
        )
        if q.requires_grad:
            if ctx.autotune:
                sized_tuned_bwd_kernel_dq[grid_dq](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    num_seqlens=len(cu_seqlens_q),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                if return_autotune:
                    dq_best_config = copy.deepcopy(sized_tuned_bwd_kernel_dq.get_best_config())
                    # BLOCK_M/N are missing with sized_tuned_bwd_kernel_*
                    dq_best_config.kwargs['BLOCK_M'] = BLOCK_M
                    dq_best_config.kwargs['BLOCK_N'] = BLOCK_N
                    tuning_result = dq_best_config
                    """
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'max_seqlen_q': max_seqlen_q,
                        'max_seqlen_k': max_seqlen_k,
                        'head_dim' : ctx.BLOCK_DMODEL,
                        'BLOCK_DMODEL' : head_dim_rounded,
                        'CAUSAL'  : ctx.causal,
                        'ENABLE_DROPOUT' : ctx.dropout_p > 0.0,
                    }
                    tuned_kernel = dict(dq_best_config.kwargs)
                    compiler_options = {
                        'num_warps' : dq_best_config.num_warps,
                        'num_stages': dq_best_config.num_stages,
                    }
                    tuning_result = {
                        'kernel_name' : 'bwd_kernel_dq',
                        'inputs' : inputs,
                        'tuned_kernel' : tuned_kernel,
                        'compiler_options' : compiler_options,
                    }
                    """
                    ctx.tuning_result.append(('bwd_kernel_dq', tuning_result))
            else:
                bare_bwd_kernel_dq[grid_dq](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    num_seqlens=len(cu_seqlens_q),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    num_warps=4, waves_per_eu=1,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None if db.numel() == 0 else db, None, None, None, None, None, None, None

varlen_attention = _varlen_attention.apply
