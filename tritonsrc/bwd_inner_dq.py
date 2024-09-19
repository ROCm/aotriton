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
def bwd_inner_dq(
    q, kt_ptrs, k_stride, vt_ptrs, v_stride, B_block_ptr,
    sm_scale, do,
    dq, DB_block_ptr, store_db,
    l_ptrs,
    D_ptrs,
    seqlen_q,
    seqlen_k,
    start_m,
    head_dim,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    max_seqlen_k,  # It's put after philox because it is not needed by anything other than dropout
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)

    # Check for OOB accesses on D and LSE
    overflow_size_q = start_m + BLOCK_M - seqlen_q
    if overflow_size_q > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size_q, dtype=tl.int32)
        d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        d_lse_padding = tl.full((BLOCK_M, ), 0, dtype=tl.float32)
        Di = tl.load(D_ptrs + offs_m, mask=d_lse_ptrs_mask, other=d_lse_padding)
        l_i = tl.load(l_ptrs + offs_m, mask=d_lse_ptrs_mask, other=d_lse_padding)
    else:
        Di = tl.load(D_ptrs + offs_m)
        l_i = tl.load(l_ptrs + offs_m)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = min(start_m + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (lo, 0))

    qk_scale = sm_scale * 1.44269504089
    q = (q* qk_scale).to(q.type.element_ty)

    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)
    '''
    for start_n in range(lo, hi, BLOCK_N):
        if start_n + BLOCK_N > hi:
            k_padded = True
        else:
            k_padded = False
        # -- load k, v --
        # shape = (BLOCK_DMODEL, BLOCK_N), offs = (0, BLOCK_N * iter) = (0, start_n)
        # kt = tl.load(K_block_ptr)
        # vt = tl.load(V_block_ptr)
        if k_padded:
            kt = load_fn(kt_ptrs, ld_offs_d, offs_n + start_n, head_dim, seqlen_k)
        else:
            kt = load_fn(kt_ptrs, ld_offs_d, None, head_dim, seqlen_k)

        # TODO: pre_load_vt
        if k_padded:
            vt = load_fn(vt_ptrs, ld_offs_d, offs_n + start_n, head_dim, seqlen_k)
        else:
            vt = load_fn(vt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
        # -- compute qk ----
        # q.offs = (start_m, 0), k.offs = (0, start_n)
        qk = dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float("-inf"))
        overflow_size_k = start_n + BLOCK_N - seqlen_k
        boundary_n = tl.full((BLOCK_M, ), seqlen_k, dtype=tl.int32)
        size_n = start_n + tl.arange(0, BLOCK_N)
        mask = size_n[None, :] < boundary_n[:, None]
        qk = tl.where(mask, qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            '''
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            '''
            # FIXME: Must use boundary_check uncondtionally.
            # The optimized tl.load above causes nan for some reason
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * max_seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        if BLOCK_M == 1:
            dq += tl.view(kt, [BLOCK_DMODEL]) * ds.to(q.type.element_ty)
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), kt.shape = (BLOCK_DMODEL, BLOCK_N)
            dq += tl.dot(ds.to(q.type.element_ty), tl.trans(kt)) # (BLOCK_M, BLOCK_DMODEL)
        if BIAS_TYPE == 1:
            if store_db:
                tl.store(DB_block_ptr, ds.to(DB_block_ptr.type.element_ty), boundary_check=(0,1))
        # update pointers
        # Keep the block ptr as comment
        # K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        # V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        kt_ptrs += BLOCK_N * k_stride
        vt_ptrs += BLOCK_N * v_stride
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))
    return (dq * sm_scale).to(dq.type.element_ty)
