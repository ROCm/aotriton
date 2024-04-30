#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""
import triton
import triton.language as tl
from fwd_kernel import dropout_mask, dropout_rng, dropout_offsets

# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M : tl.constexpr, QDIM : tl.constexpr, KDIM : tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)

# TODO: Remove Unused 'Out' Argument from kernels below
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, B, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    max_seqlens_q, max_seqlens_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_N
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    # TODO: Support varlen here
    seqlen_q = max_seqlens_q
    seqlen_k = max_seqlens_k
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_M)
    # Initialize pointers to Q, K, V
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.
    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_h * stride_kh + off_z * stride_kz
    KT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    if start_m + BLOCK_N > seqlen_k:
        k_padded = True
    else:
        k_padded = False
    v_offset = off_h * stride_vh + off_z * stride_vz
    VT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, start_m),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    do_offset = off_h * stride_oh + off_z * stride_oz
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    off_zh = off_z * num_h + off_h * 1
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h * stride_bh + off_z * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(0, start_m),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    qk_scale = sm_scale * 1.44269504089
    # load k and v: they will stay in SRAM throughout
    # (BLOCK_DMODEL, BLOCK_N)
    if PADDED_HEAD:
        kt = tl.load(KT_block_ptr, boundary_check=(1,0), padding_option="zero")
    else:
        kt = tl.load(KT_block_ptr, boundary_check=(1,), padding_option="zero")
    kt = (kt * qk_scale).to(KT_block_ptr.type.element_ty)
    # (BLOCK_DMODEL, BLOCK_N)
    if PADDED_HEAD:
        vt = tl.load(VT_block_ptr, boundary_check=(1,0), padding_option="zero")
    else:
        vt = tl.load(VT_block_ptr, boundary_check=(1,), padding_option="zero")
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    lo = (start_m // BLOCK_M) * BLOCK_M if CAUSAL else 0
    hi = seqlen_q
    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
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
    start_m: select k and dV
    start_n: select q and dO
    '''
    # loop over q (seqlen_q, dhead), do (seqlen_q, d_head)
    for start_n in range(lo, hi, BLOCK_M):
        if lo + BLOCK_M > seqlen_q:
            q_padded = True
        else:
            q_padded = False
        offs_m_curr = offs_n[:, None] + start_n # (BLOCK_M, 1)
        # -- load q, do --
        # TODO: It is more optimal to do OOB check only in the last iter.
        # (BLOCK_M, BLOCK_DMODEL), offs = (BLOCK_M * iter, 0) = (start_n, 0)
        if PADDED_HEAD:
            q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        # do: (BLOCK_M, BLOCK_DMODEL)
        if PADDED_HEAD:
            do = tl.load(DO_block_ptr, boundary_check=(0,1), padding_option="zero")
        else:
            do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # TODO: These two checks can be optimized to occur on the last iter.
        overflow_size = start_n + BLOCK_M - seqlen_q
        if overflow_size > 0:
            boundary_n = tl.full((BLOCK_N, ), seqlen_q, dtype=tl.int32)
            mask = offs_m_curr < boundary_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            qk = tl.where(offs_m_curr >= offs_m[None, :], qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            # FIXME: do boundary_check correctly
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
        # q.offs = (start_n, 0), k.offs = (0, start_m)
        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt) # (BLOCK_M, BLOCK_N)
        # Check for OOB accesses on D and LSE
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        d_lse_padding = tl.full((BLOCK_M, ), 0, dtype=tl.float32)
        Di = tl.load(D_ptrs + offs_m_curr,
                     mask=d_lse_ptrs_mask[:, None],
                     other=d_lse_padding[:, None])
        l_i = tl.load(l_ptrs + offs_m_curr,
                      mask=d_lse_ptrs_mask[:,None],
                      other=d_lse_padding[:, None])
        p = tl.math.exp2(qk - l_i) # (BLOCK_M, BLOCK_N)
        # -- compute dv ----
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_n * seqlen_k + start_m
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            # CAVEAT: do NOT update p, ds needs the original p
            if BLOCK_M == 1:
                dv += tl.where(keep, p / (1 - dropout_p), 0.0).to(Q.dtype.element_ty) * do
            else:
                dv += tl.dot(tl.trans(tl.where(keep, p / (1 - dropout_p), 0.0)).to(Q.dtype.element_ty), do)
        else:
            if BLOCK_M == 1:
                dv += p.to(Q.dtype.element_ty) * do
            else:
                # dv += tl.dot(tl.trans(p.to(do.dtype)), do)
                dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # compute dp = dot(do, vt)
        # dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        # do.shape = (BLOCK_M, BLOCK_DMODEL) vt.shape = (BLOCK_DMODEL, BLOCK_N)
        dp += tl.dot(do, vt)
        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di) # (BLOCK_M, BLOCK_N)
        # compute dk
        if BLOCK_M == 1:
            dk += ds.to(Q.dtype.element_ty) * q
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), q.shape = (BLOCK_M, BLOCK_DMODEL)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q) # (BLOCK_N, BLOCK_DMODEL)
        # update pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0)) # Debug DO accessing problems
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_M, 0))
    # initialize pointers to output
    dk_offset = off_h * stride_dkh + off_z * stride_dkz
    DK_block_ptr = tl.make_block_ptr(
        base=DK + dk_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_dkn, stride_dkk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    dv_offset = off_h * stride_dvh + off_z * stride_dvz
    DV_block_ptr = tl.make_block_ptr(
        base=DV + dv_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_dvk, stride_dvn),
        offsets=(start_m, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DK_block_ptr, (dk * sm_scale).to(DK.type.element_ty), boundary_check=(0,1))
    tl.store(DV_block_ptr, dv.to(DV.type.element_ty), boundary_check=(0,1))

@triton.jit
def bwd_kernel_dq(
    Q, K, V, B, sm_scale, Out, DO,
    DQ, DB,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    max_seqlens_q, max_seqlens_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    # TODO: Support varlen here
    seqlen_q = max_seqlens_q
    seqlen_k = max_seqlens_k
    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointers to Q, K, V
    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if start_m + BLOCK_M > seqlen_q:
        q_padded = True
    else:
        q_padded = False
    k_offset = off_h * stride_kh + off_z * stride_kz
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_h * stride_vh + off_z * stride_vz
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    do_offset = off_h * stride_oh + off_z * stride_oz
    DO_block_ptr = tl.make_block_ptr(
        base=DO + do_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    off_zh = off_z * num_h + off_h * 1
    if BIAS_TYPE == 0:
        B_block_ptr = 0
        DB_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h * stride_bh + off_z * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
        if (stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0:
            store_db = False
        else:
            store_db = True
        # Still have to make one even if no_db = False
        # due to a limit of Triton: runtime branches must have identical data types.
        DB_block_ptr = tl.make_block_ptr(
                base=DB + off_h * stride_dbh + off_z * stride_dbz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_dbm, stride_dbn),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    qk_scale = sm_scale * 1.44269504089
    # load q and do: they will stay in SRAM throughout
    if PADDED_HEAD:
        q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    if PADDED_HEAD:
        do = tl.load(DO_block_ptr, boundary_check=(0,1), padding_option="zero")
    else:
        do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
    # Check for OOB accesses on D and LSE
    overflow_size_q = start_m + BLOCK_M - seqlen_q
    boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size_q, dtype=tl.int32)
    d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
    d_lse_padding = tl.full((BLOCK_M, ), 0, dtype=tl.float32)
    Di = tl.load(D_ptrs + offs_m, mask=d_lse_ptrs_mask, other=d_lse_padding)
    l_i = tl.load(l_ptrs + offs_m, mask=d_lse_ptrs_mask, other=d_lse_padding)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v
    lo = 0
    hi = min(start_m + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
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
        if PADDED_HEAD:
            kt = tl.load(K_block_ptr, boundary_check=(1,0), padding_option="zero")
            vt = tl.load(V_block_ptr, boundary_check=(1,0), padding_option="zero")
        else:
            kt = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
            vt = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
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
            philox_offset = batch_philox_offset + start_m * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        if BIAS_TYPE == 1:
            if store_db:
                tl.store(DB_block_ptr, ds.to(DB.type.element_ty), boundary_check=(0,1))
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.
        if BLOCK_M == 1:
            dq += tl.view(kt, [BLOCK_DMODEL]) * ds.to(Q.type.element_ty)
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), kt.shape = (BLOCK_DMODEL, BLOCK_N)
            dq += tl.dot(ds.to(Q.type.element_ty), tl.trans(kt)) # (BLOCK_M, BLOCK_DMODEL)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))
    # initialize pointers to output
    dq_offset = off_h * stride_dqh + off_z * stride_dqz
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ + dq_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DQ_block_ptr, (dq * sm_scale).to(DQ_block_ptr.type.element_ty), boundary_check=(0,1))

