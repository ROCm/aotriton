#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import dropout_mask, dropout_rng, dropout_offsets
from masked_load_store import load_fn

# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M : tl.constexpr, QDIM : tl.constexpr, KDIM : tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)

@triton.jit
def bwd_inner_dk_dv(
    # I/O Tensor
    dk1, dk2,
    dv1, dv2,
    # Problem Description
    q_ptrs1, q_ptrs2, q_stride,
    kt1, kt2, vt1, vt2,
    B_block_ptr,
    do_ptrs1, do_ptrs2, do_stride,
    l_ptrs,
    D_ptrs,
    seqlen_q, seqlen_k, head_dim,
    # Sub-problem range, (lo, hi) specify the range for seqlen_q
    start_k, lo, hi, overflow_size,
    ## Dropout
    ### max_seqlen_k is put in Dropout section because it is not needed by
    ### anything other than dropout
    dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
    # constexpr starts here
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    SPLIT_DMODEL: tl.constexpr,
):
    # initialize offsets
    offs_k = start_k + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, BLOCK_M)
    if SPLIT_DMODEL:
        ld_offs_d1 = None  # head_dim is always >= HALVED_DMODEL, otherwise BLOCK_DMODEL can be halved
        ld_offs_d2 = None if not PADDED_HEAD else tl.arange(HALVED_DMODEL, BLOCK_DMODEL)
    else:
        ld_offs_d1 = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        ld_offs_d2 = 0

    # Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    # DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    q_ptrs1 += lo * q_stride
    do_ptrs1 += lo * do_stride
    if SPLIT_DMODEL:
        q_ptrs2 += lo * q_stride
        do_ptrs2 += lo * do_stride

    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (lo, 0))

    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)

    dV = (QK)^T dO

    dV1 = qk11 dO1 + qk21 dO2 = q1 k1 dO1 + q2 k1 dO2
    dV2 = qk12 dO1 + qk22 dO2 = q1 k2 dO1 + q2 k2 dO2
                                ~~~~~ = 0
    start_k: select k and dV
    start_q: select q and dO
    '''
    # loop over q (seqlen_q, dhead), do (seqlen_q, d_head)
    for start_q in range(lo, hi, BLOCK_M):
        # TODO: Unify the name, the usage of m/n is very confusing
        offs_q_curr = offs_q[:, None] + start_q # (BLOCK_M, 1)
        # -- load q, do --
        # TODO: It is more optimal to do OOB check only in the last iter.
        # (BLOCK_M, BLOCK_DMODEL), offs = (BLOCK_M * iter, 0) = (start_q, 0)
        #
        # This common function can be further split into regular and
        # non-regular version, determined by tl.constexpr, just like the fwd kernel.

        if not FULL_BLOCKS:
            q1 = load_fn(q_ptrs1, offs_q + start_q, ld_offs_d1, seqlen_q, head_dim)
        else:
            q1 = load_fn(q_ptrs1, None, ld_offs_d1, seqlen_q, head_dim)
        # do = tl.load(DO_block_ptr)
        # TODO: pre_load_do
        if not FULL_BLOCKS:
            do1 = load_fn(do_ptrs1, offs_q + start_q, ld_offs_d1, seqlen_q, head_dim)
        else:
            do1 = load_fn(do_ptrs1, None, ld_offs_d1, seqlen_q, head_dim)
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # TODO: These two checks can be optimized to occur on the last iter.
        if not FULL_BLOCKS:
            if overflow_size > 0:
                boundary_n = tl.full((BLOCK_N, ), seqlen_q, dtype=tl.int32)
                mask = offs_q_curr < boundary_n[None, :]
                qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            qk = tl.where(offs_q_curr >= offs_k[None, :], qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            # FIXME: do boundary_check correctly
            # TODO: check q_padded is correct calculated, the condition should be start_q + BLOCK_M
            """
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            """
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        # q.offs = (start_q, 0), k.offs = (0, start_k)
        qk += tl.dot(q1, kt1) # (BLOCK_M, BLOCK_N)
        if SPLIT_DMODEL:
            if not FULL_BLOCKS:
                q2 = load_fn(q_ptrs2, offs_q + start_q, ld_offs_d2, seqlen_q, head_dim)
            else:
                q2 = load_fn(q_ptrs2, None, ld_offs_d2, seqlen_q, head_dim)
            qk += tl.dot(q2, kt2) # (BLOCK_M, BLOCK_N)
        # Check for OOB accesses on D and LSE
        if FULL_BLOCKS:
            Di = tl.load(D_ptrs + offs_q_curr)
            l_i = tl.load(l_ptrs + offs_q_curr)
        else:
            boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
            d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
            d_lse_padding = tl.full((BLOCK_M, ), 0, dtype=tl.float32)
            Di = tl.load(D_ptrs + offs_q_curr,
                         mask=d_lse_ptrs_mask[:, None],
                         other=d_lse_padding[:, None])
            l_i = tl.load(l_ptrs + offs_q_curr,
                          mask=d_lse_ptrs_mask[:,None],
                          other=d_lse_padding[:, None])
        p = tl.math.exp2(qk - l_i) # (BLOCK_M, BLOCK_N)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            # CAVEAT: do NOT update p, ds needs the original p
            if BLOCK_M == 1:
                dv1 += tl.where(keep, p / (1 - dropout_p), 0.0).to(q_ptrs1.dtype.element_ty) * do1
            else:
                pt = tl.trans(tl.where(keep, p / (1 - dropout_p), 0.0)).to(q_ptrs1.dtype.element_ty)
                dv1 += tl.dot(pt, do1)
        else:
            if BLOCK_M == 1:
                dv1 += p.to(q_ptrs1.dtype.element_ty) * do1
            else:
                dv1 += tl.dot(tl.trans(p.to(do1.dtype)), do1)
        if SPLIT_DMODEL:
            if not FULL_BLOCKS:
                do2 = load_fn(do_ptrs2, offs_q + start_q, ld_offs_d2, seqlen_q, head_dim)
            else:
                do2 = load_fn(do_ptrs2, None, ld_offs_d2, seqlen_q, head_dim)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # compute dp = dot(do, vt)
        # dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        # do.shape = (BLOCK_M, BLOCK_DMODEL) vt.shape = (BLOCK_DMODEL, BLOCK_N)
        dp += tl.dot(do1, vt1)
        dp += tl.dot(do2, vt2)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        if SPLIT_DMODEL:
            if ENABLE_DROPOUT:
                philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
                keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
                # CAVEAT: do NOT update p, ds needs the original p
                if BLOCK_M == 1:
                    dv2 += tl.where(keep, p / (1 - dropout_p), 0.0).to(q_ptrs1.dtype.element_ty) * do2
                else:
                    dv2 += tl.dot(pt, do2)
            else:
                if BLOCK_M == 1:
                    dv2 += p.to(q_ptrs1.dtype.element_ty) * do2
                else:
                    dv2 += tl.dot(tl.trans(p.to(do2.dtype)), do2)
        # compute ds = p * (dp - delta[:, None])
        dst = tl.trans((p * (dp - Di)).to(q_ptrs1.dtype.element_ty)) # (BLOCK_M, BLOCK_N)
        # compute dk
        dk1 += tl.dot(dst, q1) # (BLOCK_N, BLOCK_DMODEL)
        if SPLIT_DMODEL:
            dk2 += tl.dot(dst, q2) # (BLOCK_N, BLOCK_DMODEL)

        # update pointers (block_ptr code was left intentionally as comment)
        # Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        # DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0)) # Debug DO accessing problems
        q_ptrs1 += q_stride * BLOCK_M
        do_ptrs1 += do_stride * BLOCK_M
        if SPLIT_DMODEL:
            q_ptrs2 += q_stride * BLOCK_M
            do_ptrs2 += do_stride * BLOCK_M
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_M, 0))
    return dk1, dk2, dv1, dv2
