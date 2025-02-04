#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import fast_dropout_mask
from masked_load_store import load_fn
from triton.language.extra import libdevice
from composed_tensors import (
    composed_offs_1d,
    composed_advance,
    composed_load,
    composed_dot_both,
    composed_dot_rhs,
    composed_mul_lhs,
    composed_mul_acc,
)

# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M : tl.constexpr, QDIM : tl.constexpr, KDIM : tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)

@triton.jit
def bwd_inner_dq(
    # I/O Tensor
    dq0, dq1, dq2,
    qk_scale, bias_scale,
    DB_block_ptr, store_db,
    # Problem Description
    q0, q1, q2,
    kt_ptrs0, kt_ptrs1, kt_ptrs2,
    k_stride,
    vt_ptrs0, vt_ptrs1, vt_ptrs2,
    v_stride, B_block_ptr,
    do0, do1, do2,
    Di, l_i,
    seqlen_q, seqlen_k, head_dim,
    # Sub-problem range, (lo, hi) specify the range for seqlen_q
    start_q, lo, hi,
    ## Dropout
    ### max_seqlen_k is put in Dropout section because it is not needed by
    ### anything other than dropout
    idropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
    # constexpr starts here
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL0: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
    BLOCK_DMODEL2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # DEBUG_RIGHT: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # initialize offsets
    offs_q = start_q + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_N)
    kt_ptrs0, kt_ptrs1, kt_ptrs2 = composed_advance(kt_ptrs0, kt_ptrs1, kt_ptrs2,
                                                    lo * k_stride,
                                                    BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    vt_ptrs0, vt_ptrs1, vt_ptrs2 = composed_advance(vt_ptrs0, vt_ptrs1, vt_ptrs2,
                                                    lo * v_stride,
                                                    BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (0, lo))
        DB_block_ptr = tl.advance(DB_block_ptr, (0, lo))

    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)
    '''
    for start_k in range(lo, hi, BLOCK_N):
        offs_k_curr = offs_k[None, :] + start_k # (1, BLOCK_N)
        # -- load k, v --
        # shape = (BLOCK_DMODEL, BLOCK_N), offs = (0, BLOCK_N * iter) = (0, start_k)
        # kt = tl.load(K_block_ptr)
        # vt = tl.load(V_block_ptr)
        k_offs_n = start_k + tl.arange(0, BLOCK_N)
        PADDED_SEQ : tl.constexpr = not FULL_BLOCKS

        kt0, kt1, kt2 = composed_load(kt_ptrs0, kt_ptrs1, kt_ptrs2,
                                      k_offs_n,
                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                      seqlen_k, head_dim,
                                      other=0.0,
                                      PADDED_ROW=PADDED_SEQ,
                                      PADDED_COL=PADDED_HEAD,
                                      TRANSPOSED=True)
        # TODO: pre_load_vt
        vt0, vt1, vt2 = composed_load(vt_ptrs0, vt_ptrs1, vt_ptrs2,
                                      k_offs_n,
                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                      seqlen_k, head_dim,
                                      other=0.0,
                                      PADDED_ROW=PADDED_SEQ,
                                      PADDED_COL=PADDED_HEAD,
                                      TRANSPOSED=True)
        # -- compute qk ----
        # q.offs = (start_m, 0), k.offs = (0, start_k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = composed_dot_both(q0, q1, q2,
                               kt0, kt1, kt2,
                               qk,
                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        if not FULL_BLOCKS:
            k_boundary = tl.full((BLOCK_M, ), seqlen_k, dtype=tl.int32)
            mask = offs_k_curr < k_boundary[:, None]
            qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            # qk = tl.where(offs_q[:, None] >= (offs_k[None, :] + start_k), qk, float("-inf"))
            qk = tl.where(offs_q[:, None] >= offs_k_curr, qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            # FIXME: Must use boundary_check uncondtionally.
            # The optimized tl.load above causes nan for some reason
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        # FIXME: Potential bug https://github.com/ROCm/aotriton/issues/54
        p = tl.math.exp2(qk_scale * qk - l_i[:, None])

        if not FULL_BLOCKS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp = composed_dot_both(do0, do1, do2,
                               vt0, vt1, vt2,
                               dp,
                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
            keep = fast_dropout_mask(philox_seed, philox_offset, idropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            dp = tl.where(keep, dp * dropout_scale, 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        if BLOCK_M == 1:
            dq0, dq1, dq2 = composed_mul_acc(tl.view(kt0, [BLOCK_DMODEL0]),
                                             tl.view(kt1, [BLOCK_DMODEL1]),
                                             tl.view(kt2, [BLOCK_DMODEL2]),
                                             ds.to(q0.type.element_ty),
                                             dq0, dq1, dq2,
                                             BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), kt.shape = (BLOCK_DMODEL, BLOCK_N)
            dq0, dq1, dq2 = composed_dot_rhs(ds.to(q0.type.element_ty),
                                             tl.trans(kt0),
                                             tl.trans(kt1),
                                             tl.trans(kt2),
                                             dq0, dq1, dq2,
                                             BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        if BIAS_TYPE == 1:
            if store_db:
                tl.store(DB_block_ptr, ds.to(DB_block_ptr.type.element_ty), boundary_check=(0,1))
        # update pointers
        # Keep the block ptr as comment
        # K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        # V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        kt_ptrs0, kt_ptrs1, kt_ptrs2 = composed_advance(kt_ptrs0, kt_ptrs1, kt_ptrs2,
                                                        BLOCK_N * k_stride,
                                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        vt_ptrs0, vt_ptrs1, vt_ptrs2 = composed_advance(vt_ptrs0, vt_ptrs1, vt_ptrs2,
                                                        BLOCK_N * v_stride,
                                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))
    return dq0, dq1, dq2
