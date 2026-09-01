#!/usr/bin/env python
# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`torch.autograd.Function` driving the Triton kernels in varlen mode.

Resynced against the VarlenBits kernel signature.  What it looked like before
is worth recording, because it explains why `test_varlen.py` in this directory
had not been catching anything: it passed `Head_dim=Lk` (the argument had been
split into `Hdim_qk`/`Hdim_vo`), omitted `seq_strides_q/k` and `NUM_XCDS`
entirely, called the now-merged `bwd_preprocess_varlen`, and allocated `M` as
`(batch, num_head_q, max_seqlen_q)` -- the *old* padded LSE layout.  None of
those would even bind, so the file had drifted out of the reachable set rather
than merely gone stale.
"""

import copy
import numpy as np
import torch
import triton
import triton.language as tl
from flash import (
    attn_fwd as bare_attn_fwd,
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq,
    debug_simulate_encoded_softmax,
)
from varlen_bits import (
    VARLEN_BITS_COMPACT,
    VARLEN_BITS_PADDED,
    VARLEN_BITS_STRIDED,
)
from tuned_bwd import NUM_XCDS
from attn_torch_function import (
        DEFAULT_PHILOX_SEED,
        DEFAULT_PHILOX_OFFSET_1,
        DEFAULT_PHILOX_OFFSET_2,
        DEFAULT_PHILOX_OFFSET,
        PersistentType,
        AttentionExtraArgs,
        factor_head_dim,
)

class CausalType:
    NONE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2
    WINDOWED = 3

class VarlenWindowValue:
    TOP_LEFT = -2147483647          # np.array([0x80000001]).astype(np.int32)
    BOTTOM_RIGHT = -2147483646      # np.array([0x80000002]).astype(np.int32)

# The bits for each of the three varlen layouts this launcher can build.
# `strided` is the only one that passes a position array.
VARLEN_TYPE_BITS = {
    'compact': VARLEN_BITS_COMPACT,
    'padded': VARLEN_BITS_PADDED,
    'strided': VARLEN_BITS_STRIDED,
}

def translate_causal_varlen(causal):
    window_left, window_right = 0, 0
    if isinstance(causal, tuple):
        window_left, window_right = causal
        causal_type = CausalType.WINDOWED
    elif isinstance(causal, bool):
        # causal_type = CausalType.TOP_LEFT if causal else CausalType.NONE
        causal_type = CausalType.WINDOWED if causal else CausalType.NONE
        if causal:
            window_left = VarlenWindowValue.TOP_LEFT
            window_right = VarlenWindowValue.TOP_LEFT
    else:
        assert causal in [CausalType.NONE, CausalType.TOP_LEFT, CausalType.BOTTOM_RIGHT]
        if causal == CausalType.TOP_LEFT:
            causal_type = CausalType.WINDOWED
            window_left = VarlenWindowValue.TOP_LEFT
            window_right = VarlenWindowValue.TOP_LEFT
        elif causal == CausalType.BOTTOM_RIGHT:
            causal_type = CausalType.WINDOWED
            window_left = VarlenWindowValue.BOTTOM_RIGHT
            window_right = VarlenWindowValue.BOTTOM_RIGHT
        else:
            causal_type = causal
    return causal_type, window_left, window_right


def build_seqinfo(seqlens_q, seqlens_k, varlen_type, device):
    """`(bits, seqinfo_q0, seqinfo_k0, seqinfo_q1, seqinfo_k1)`.

    `seqinfo_?0` always carries the cumulative lengths; `seqinfo_?1` carries the
    cumulative *slot* positions and is only passed by `strided`, where the two
    differ.  Compact reuses the length array for its position, which is the
    whole reason REUSE exists -- see varlen_bits.py.
    """
    null_tensor = torch.empty((0), device=device, dtype=torch.int32)

    def cumsum_i32(lens):
        return torch.tensor([0] + np.cumsum(lens).tolist(),
                            dtype=torch.int32, device=device)

    if varlen_type == 'strided':
        seqlens_q, padlens_q = seqlens_q
        seqlens_k, padlens_k = seqlens_k
        seqinfo_q1 = cumsum_i32(np.asarray(seqlens_q) + np.asarray(padlens_q))
        seqinfo_k1 = cumsum_i32(np.asarray(seqlens_k) + np.asarray(padlens_k))
    else:
        seqinfo_q1 = null_tensor
        seqinfo_k1 = null_tensor
    bits = VARLEN_TYPE_BITS[varlen_type]
    return (bits, cumsum_i32(seqlens_q), cumsum_i32(seqlens_k),
            seqinfo_q1, seqinfo_k1, seqlens_q, seqlens_k)


VERBOSE = False

class _varlen_attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, seqlens_q, seqlens_k, causal, sm_scale, dropout_p,
                attn_extra_args=AttentionExtraArgs(), varlen_type='compact'):
        return_encoded_softmax = attn_extra_args.return_encoded_softmax
        autotune = attn_extra_args.autotune
        return_autotune = attn_extra_args.return_autotune
        assert not autotune, 'Autotuning the varlen JIT launcher is unsupported'
        dtype = q.dtype
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk
        head_dim_factors = factor_head_dim(Lk)
        head_dim_rounded = sum(head_dim_factors)
        padded_head = head_dim_rounded != Lk
        num_head_q = q.shape[1]
        num_head_k = k.shape[1]

        (varlen_bits, seqinfo_q0, seqinfo_k0, seqinfo_q1, seqinfo_k1,
         seqlens_q, seqlens_k) = build_seqinfo(seqlens_q, seqlens_k,
                                               varlen_type, q.device)
        batch = len(seqlens_q)
        max_seqlen_q = int(np.max(seqlens_q))
        max_seqlen_k = int(np.max(seqlens_k))
        o = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[3]),
                        device=q.device, dtype=q.dtype)

        causal_type, window_left, window_right = translate_causal_varlen(causal)

        persistent_type = attn_extra_args.persistent_type
        if persistent_type == PersistentType.AUTOSELECT:
            persistent_type = PersistentType.NONE if causal_type == CausalType.NONE else PersistentType.DYNAMIC

        # Host and kernel each compute this predicate, and they must agree
        # exactly: the kernel's spelling is
        # `(Varlen_bits & 0xFFFF) != 0` in fwd_kernel.py. Disagreeing means a
        # persistent-shaped grid against a non-persistent walk.
        unsupported_by_persistent = (varlen_bits & 0xFFFF) != 0

        null_tensor = torch.empty((0), device=q.device, dtype=torch.int32)
        if persistent_type == PersistentType.DYNAMIC:
            persistent_atomic_counter = torch.zeros([1], device=q.device, dtype=torch.int32)
        else:
            persistent_atomic_counter = null_tensor

        # The fallback-ed kernel needs fallback launch options
        if persistent_type == PersistentType.NONE or unsupported_by_persistent:
            def grid(META):
                S = triton.cdiv(max_seqlen_q, META['BLOCK_M'])
                return (S, num_head_q, batch) if META['NUM_XCDS'] == 1 else (num_head_q, S, batch)
            Num_CU = 0
        else:
            Num_CU = torch.cuda.get_device_properties(q.device).multi_processor_count
            grid = lambda META: (min(Num_CU * META['GRID_CU_MULTIP'],
                                     triton.cdiv(max_seqlen_q, META['BLOCK_M']) * num_head_q * batch), )

        # LSE layout, derived from the bits rather than chosen here:
        #   compact / strided -> (H, T), T the total token count
        #   padded            -> (batch * H, Max_seqlen_q)
        # No strides are passed to the kernel, so this must be contiguous.
        if varlen_type == 'padded':
            M = torch.empty((batch * num_head_q, max_seqlen_q),
                            device=q.device, dtype=torch.float32)
        elif varlen_type == 'strided':
            total_tokens = int(seqinfo_q1[batch].item())
            M = torch.empty((num_head_q, total_tokens), device=q.device, dtype=torch.float32)
        else:
            M = torch.empty((num_head_q, int(np.sum(seqlens_q))),
                            device=q.device, dtype=torch.float32)
        if attn_extra_args.fillnan:
            for t in (o, M):
                t.fill_(float('nan'))
        if return_encoded_softmax:
            encoded_softmax = torch.ones((batch, num_head_q, max_seqlen_q, max_seqlen_k),
                    device=q.device,
                    dtype=_varlen_attention.DEBUG_MASK_DTYPE) * 114.514
        else:
            encoded_softmax = None
        if VERBOSE:
            print(f'{q.shape=} {q.stride()=}')
            print(f'{k.shape=} {k.stride()=}')
            print(f'{v.shape=} {v.stride()=}')
            print(f'{o.shape=} {M.shape=} {varlen_bits=:#x}')

        if dropout_p > 0.0:
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        else:
            u64nulltensor = torch.empty([0], device=q.device, dtype=torch.uint64)
            philox_seed = u64nulltensor
            philox_offset1 = u64nulltensor
            philox_offset2 = 0
            philox_seed_output = u64nulltensor
            philox_offset_output = u64nulltensor

        b = torch.empty((0,0,0,0), device=q.device, dtype=q.dtype)
        BIAS_TYPE = 0

        # TODO alibi_slopes
        alibi_slopes = torch.empty((0,0), device=q.device, dtype=q.dtype)

        # TODO: int8
        q_descale = k_descale = p_scale = p_descale = v_descale = 0

        use_small_block = dropout_p > 0.0 or return_encoded_softmax
        if use_small_block:
            BLOCK_M = 64
            BLOCK_N = 32
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if dtype == torch.float32:
            BLOCK_M //= 2

        cfg = triton.Config({'BLOCK_M': BLOCK_M,
                             'BLOCK_N': BLOCK_N,
                             'waves_per_eu': 2,
                             'PRE_LOAD_V': False},
                            num_stages=1, num_warps=4)
        fwd_notuner = triton.autotune(configs=[cfg],
                                      key=['Max_seqlen_q', 'Max_seqlen_k', 'CAUSAL_TYPE'])
        attn_fwd = fwd_notuner(bare_attn_fwd)
        attn_fwd[grid](
            # Basic SDPA
            q, k, v, b, alibi_slopes, sm_scale, M, o,
            q_descale, k_descale, p_scale, p_descale, v_descale,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            *b.stride(),
            *alibi_slopes.stride(),
            # MQA/GQA
            Num_head_q=num_head_q,
            Num_head_k=num_head_k,
            # Varlen
            Varlen_bits=varlen_bits,
            seqinfo_q0=seqinfo_q0,
            seqinfo_k0=seqinfo_k0,
            Max_seqlen_q=max_seqlen_q,
            Max_seqlen_k=max_seqlen_k,
            seqinfo_q1=seqinfo_q1,
            seqinfo_k1=seqinfo_k1,
            # Head Dimensions
            BLOCK_DMODEL=head_dim_rounded,
            Hdim_qk=Lk,
            Hdim_vo=Lv,
            PADDED_HEAD=padded_head,
            # dropout and PRNG
            ENABLE_DROPOUT=dropout_p > 0.0,
            dropout_p=dropout_p,
            philox_seed_ptr=philox_seed,
            philox_offset1=philox_offset1,
            philox_offset2=philox_offset2,
            philox_seed_output=philox_seed_output,
            philox_offset_output=philox_offset_output,
            RETURN_ENCODED_SOFTMAX=False,
            encoded_softmax=None,
            # Causal
            CAUSAL_TYPE=causal_type,
            Window_left=window_left,
            Window_right=window_right,
            # bias
            BIAS_TYPE=BIAS_TYPE,
            # alibi
            USE_ALIBI=False,
            # INT8
            INT8=False,
            INT8_KV=False,
            USE_P_SCALE=False,
            # Persistent related arguments
            PERSISTENT_TYPE=persistent_type,
            persistent_atomic_counter=persistent_atomic_counter,
            Num_CU=Num_CU,
            GRID_CU_MULTIP=2,
            Batch=batch,
            # Performance related, but fixed for arch
            NUM_XCDS=NUM_XCDS,
        )
        if return_encoded_softmax:
            es_grid = lambda META: (
                triton.cdiv(encoded_softmax.shape[2], META['BLOCK_M']),
                encoded_softmax.shape[1],
                encoded_softmax.shape[0],
            )
            debug_simulate_encoded_softmax[es_grid](encoded_softmax,
                                                    *encoded_softmax.stride(),
                                                    dropout_p,
                                                    Num_head_q=encoded_softmax.shape[1],
                                                    Max_seqlen_q=encoded_softmax.shape[2],
                                                    Max_seqlen_k=encoded_softmax.shape[3],
                                                    philox_seed_ptr=philox_seed,
                                                    philox_offset1=philox_offset1,
                                                    philox_offset2=philox_offset2,
                                                    BLOCK_M=32,
                                                    BLOCK_N=32)

        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.varlen_bits = varlen_bits
        ctx.varlen_type = varlen_type
        ctx.seqinfo = (seqinfo_q0, seqinfo_k0, seqinfo_q1, seqinfo_k1)
        ctx.seqlens_q = seqlens_q
        ctx.seqlens_k = seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.sm_scale = sm_scale
        ctx.head_dim = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed_output
        ctx.philox_offset = philox_offset_output
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.bias_type = BIAS_TYPE
        ctx.tuning_result = None
        ctx.attn_extra_args = attn_extra_args
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward(ctx, do, _, fwd_tuning_result):
        q, k, v, b, o, L = ctx.saved_tensors
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == ctx.head_dim
        head_dim_factors = factor_head_dim(ctx.head_dim)
        head_dim_rounded = sum(head_dim_factors)
        padded_head = head_dim_rounded != min(Lk, Lv)
        attn_extra_args = ctx.attn_extra_args
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset
        varlen_bits = ctx.varlen_bits
        seqinfo_q0, seqinfo_k0, seqinfo_q1, seqinfo_k1 = ctx.seqinfo
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        batch = len(ctx.seqlens_q)
        num_head_q = int(q.shape[1])
        num_head_k = int(k.shape[1])
        causal_type, window_left, window_right = translate_causal_varlen(ctx.causal)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b)
        # Delta always mirrors LSE's shape and strides.
        delta = torch.empty_like(L)
        if attn_extra_args.fillnan:
            for t in (dq, dk, dv, db, delta):
                t.fill_(float('nan'))

        use_small_block = ctx.dropout_p > 0.0
        use_medium_block = ctx.bias_type != 0
        if use_small_block:
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

        PREP_BLOCK = 16
        grid_prep = (triton.cdiv(max_seqlen_q, PREP_BLOCK), num_head_q, batch)
        bare_bwd_preprocess[grid_prep](
            o, do, delta,
            *o.stride(),
            *do.stride(),
            seqinfo_q0=seqinfo_q0,
            varlen_bits=varlen_bits,
            max_seqlen_q=max_seqlen_q,
            seqinfo_q1=seqinfo_q1,
            hdim_vo=Lv,
            BLOCK_M=PREP_BLOCK, D_HEAD=head_dim_rounded,
            PADDED_HEAD=padded_head,
        )

        def grid_dk_dv(META):
            S = triton.cdiv(max_seqlen_k, META['BLOCK_N'])
            return (S, num_head_k, batch) if META['NUM_XCDS'] == 1 else (num_head_k, S, batch)
        stride_dbz, stride_dbh, stride_dbm, stride_dbn = db.stride()
        if db.numel() == 0 or not b.requires_grad:
            # Passing all zeros to indicate no elements
            stride_dbz, stride_dbh, stride_dbm, stride_dbn = 0, 0, 0, 0
        cfg = triton.Config({'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'waves_per_eu': 1},
                            num_stages=1, num_warps=4)
        if k.requires_grad and v.requires_grad:
            bwd_dkdv_notuner = triton.autotune(configs=[cfg],
                                               key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'])
            bwd_kernel_dk_dv = bwd_dkdv_notuner(bare_bwd_kernel_dk_dv)
            bwd_kernel_dk_dv[grid_dk_dv](
                q, k, v, b, ctx.sm_scale, do,
                dk, dv,
                L, delta,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *b.stride(),
                *do.stride(),
                *dk.stride(),
                *dv.stride(),
                num_head_q=num_head_q,
                num_head_k=num_head_k,
                seqinfo_q0=seqinfo_q0,
                seqinfo_k0=seqinfo_k0,
                varlen_bits=varlen_bits,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqinfo_q1=seqinfo_q1,
                seqinfo_k1=seqinfo_k1,
                hdim_qk=Lk,
                hdim_vo=Lv,
                dropout_p=ctx.dropout_p,
                philox_seed_ptr=philox_seed,
                philox_offset1=philox_offset,
                philox_offset2=0,
                Window_left=window_left,
                Window_right=window_right,
                BLOCK_DMODEL=head_dim_rounded,
                CAUSAL_TYPE=causal_type,
                ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                PADDED_HEAD=padded_head,
                BIAS_TYPE=ctx.bias_type,
                NUM_XCDS=NUM_XCDS,
            )

        def grid_dq(META):
            S = triton.cdiv(max_seqlen_q, META['BLOCK_M'])
            return (S, num_head_q, batch) if META['NUM_XCDS'] == 1 else (num_head_q, S, batch)
        if q.requires_grad:
            bwd_dq_notuner = triton.autotune(configs=[cfg],
                                             key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'])
            bwd_kernel_dq = bwd_dq_notuner(bare_bwd_kernel_dq)
            bwd_kernel_dq[grid_dq](
                q, k, v, b, ctx.sm_scale, do,
                dq, db,
                L,
                delta,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *b.stride(),
                *do.stride(),
                *dq.stride(),
                stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                num_head_q=num_head_q,
                num_head_k=num_head_k,
                seqinfo_q0=seqinfo_q0,
                seqinfo_k0=seqinfo_k0,
                varlen_bits=varlen_bits,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqinfo_q1=seqinfo_q1,
                seqinfo_k1=seqinfo_k1,
                hdim_qk=Lk,
                hdim_vo=Lv,
                dropout_p=ctx.dropout_p,
                philox_seed_ptr=philox_seed,
                philox_offset1=philox_offset,
                philox_offset2=0,
                Window_left=window_left,
                Window_right=window_right,
                BLOCK_DMODEL=head_dim_rounded,
                CAUSAL_TYPE=causal_type,
                ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                PADDED_HEAD=padded_head,
                BIAS_TYPE=ctx.bias_type,
                NUM_XCDS=NUM_XCDS,
            )
        return dq, dk, dv, None if db.numel() == 0 else db, None, None, None, None, None, None, None

varlen_attention = _varlen_attention.apply
