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
from bwd_kernel_common import bwd_kernel_dk_dv_common, bwd_kernel_dq_db_common
from masked_load_store import load_fn, mstore2d

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
    num_head_q : 'i32',
    num_head_k : 'i32',
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens : 'i32',   # set num_seqlens to zero to ignore cu_seqlens_q/k
    max_seqlen_q : 'i32', # and use max_seqlen_q/k for all seqlen_q/k
    max_seqlen_k : 'i32',
    head_dim : 'i32',
    dropout_p,
    philox_seed_ptr,
    philox_offset1 : '*u32',
    philox_offset2 : 'u32',
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_m = tl.program_id(0) * BLOCK_N  # start_m is a misused name. For dkdv it partitions seqlen_k
    off_h_k = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index, for varlen it indicates index in cu_seqlens_q/k
    num_z = tl.num_programs(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_m + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)

    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    batch_index = off_z

    if num_seqlens > 0:
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_m >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        batch_index = 0

    if num_seqlens < 0:  # for padded seqlen
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_m >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # Varlen, but padded to Rank 4 tensor
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z

    # Initialize pointers to Q, K, V
    # Q is consumed depending on block ID. Every block uses
    # previous block offset by BLOCK_M x D_HEAD.

    # Note: Q pointers are deferred to later place.
    #       GQA needs loop through off_h_q = i * off_h_k + off_h_k
    # q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm

    # Q_block_ptr = tl.make_block_ptr(
    #     base=Q,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_qm, stride_qk),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    k_offset = off_h_k * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K += k_offset
    kt_ptrs = K + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # kt_offs_n = None if start_m + BLOCK_N <= seqlen_k else start_m + tl.arange(0, BLOCK_N)
    if start_m + BLOCK_N <= seqlen_k:
        kt = load_fn(kt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
    else:
        kt = load_fn(kt_ptrs, ld_offs_d, offs_n, head_dim, seqlen_k)
    # KT_block_ptr = tl.make_block_ptr(
    #     base=K + k_offset,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_kk, stride_kn),
    #     offsets=(0, start_m),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1)
    # )
    v_offset = off_h_k * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V += v_offset
    # VT_block_ptr = tl.make_block_ptr(
    #     base=V,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_vn, stride_vk),
    #     offsets=(0, start_m),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1)
    # )
    # vt = tl.load(VT_block_ptr)
    vt_ptrs = V + offs_d[:, None] * stride_vn + offs_n[None, :] * stride_vk
    if start_m + BLOCK_N <= seqlen_k:
        vt = load_fn(vt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
    else:
        vt = load_fn(vt_ptrs, ld_offs_d, offs_n, head_dim, seqlen_k)
    # tl.device_print('vt', vt)
    # DO_block_ptr = tl.make_block_ptr(
    #     base=DO,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_om, stride_ok),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        # CAVEAT: bias is incompatible with GQA
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h_k * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(0, start_m),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    dk_offset = off_h_k * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
    DK += dk_offset
    dv_offset = off_h_k * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk
    DV += dv_offset

    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089
    kt = (kt * qk_scale).to(kt.type.element_ty)
    group_size = num_head_q // num_head_k
    for off_h_q in range(off_h_k * group_size, off_h_k * group_size + group_size):
        off_zh = off_z * num_head_q + off_h_q * 1
        # This lower loop bound is because of the causal mask. We create a lower triangular
        # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
        # be ignored in the GEMM.
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
        else:
            batch_philox_offset = 0
        # pointer to row-wise quantities in value-like data
        # Shape (batch, num_heads, max_seqlen_q)
        # In varlen cases, batch == len(cu_seqlens_q) - 1).
        # Hence off_z plays the same role in varlen/non-varlen
        D_ptrs = D + off_zh * max_seqlen_q
        l_ptrs = L + off_zh * max_seqlen_q

        q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
        do_ptrs = DO + do_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

        dk, dv = bwd_kernel_dk_dv_common(
            dk, dv,
            q_ptrs, stride_qm, kt, vt, B_block_ptr,
            do_ptrs, stride_om,
            l_ptrs,
            D_ptrs,
            seqlen_q, seqlen_k,
            start_m,
            head_dim,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            max_seqlen_k,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE)
    dk = (dk * sm_scale).to(kt.type.element_ty)
    dv = dv.to(vt.type.element_ty)
    mstore2d(dk,
             BLOCK_N,
             BLOCK_DMODEL,
             o_base=DK,
             o_start_row=start_m,
             o_start_col=0,
             o_rows=seqlen_k,
             o_cols=head_dim,
             stride_row=stride_dkn,
             stride_col=stride_dkk)
    mstore2d(dv,
             BLOCK_N,
             BLOCK_DMODEL,
             o_base=DV,
             o_start_row=start_m,
             o_start_col=0,
             o_rows=seqlen_k,
             o_cols=head_dim,
             stride_row=stride_dvk,
             stride_col=stride_dvn)

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
    num_head_q : 'i32',
    num_head_k : 'i32',
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,   # set num_seqlens to zero to ignore cu_seqlens_q/k
    max_seqlen_q, # and use max_seqlen_q/k for all seqlen_q/k
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed_ptr,
    philox_offset1 : '*u32',
    philox_offset2 : 'u32',
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_m = tl.program_id(0) * BLOCK_M
    off_h_q = tl.program_id(1) # head index
    off_h_k = off_h_q if num_head_q == num_head_k else off_h_q // (num_head_q // num_head_k)
    off_z = tl.program_id(2) # batch index
    num_z = tl.num_programs(2)
    off_zh = off_z * num_head_q + off_h_q * 1
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)

    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    batch_index = off_z

    if num_seqlens > 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0

    if num_seqlens < 0:  # for padded seqlen
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        # Varlen, but padded to Rank 4 tensor
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z
    # Initialize pointers to Q, K, V
    q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q += q_offset
    # Q_block_ptr = tl.make_block_ptr(
    #     base=Q,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_qm, stride_qk),
    #     offsets=(start_m, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    if start_m + BLOCK_M <= seqlen_q:
        q = load_fn(q_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        q = load_fn(q_ptrs, offs_m, ld_offs_d, seqlen_q, head_dim)
    k_offset = off_h_k * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K += k_offset
    kt_ptrs = K + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # K_block_ptr = tl.make_block_ptr(
    #     base=K,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_kk, stride_kn),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1)
    # )
    v_offset = off_h_k * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V += v_offset
    vt_ptrs = V + offs_d[:, None] * stride_vn + offs_n[None, :] * stride_vk
    # V_block_ptr = tl.make_block_ptr(
    #     base=V,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_vn, stride_vk),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1)
    # )
    do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    DO += do_offset
    # DO_block_ptr = tl.make_block_ptr(
    #     base=DO,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_om, stride_ok),
    #     offsets=(start_m, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    do_ptrs = DO + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    if start_m + BLOCK_M <= seqlen_q:
        do = load_fn(do_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        do = load_fn(do_ptrs, offs_m, ld_offs_d, seqlen_q, head_dim)
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * max_seqlen_q
    l_ptrs = L + off_zh * max_seqlen_q
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
    else:
        batch_philox_offset = 0

    # initialize pointers to output
    dq_offset = batch_index * stride_dqz + off_h_q * stride_dqh + cu_seqlens_q_start * stride_dqm
    DQ += dq_offset
    # DQ_block_ptr = tl.make_block_ptr(
    #     base=DQ,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_dqm, stride_dqk),
    #     offsets=(start_m, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    dq_ptrs = DQ + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk
    if start_m + BLOCK_M <= seqlen_q:
        dq = load_fn(dq_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        dq = load_fn(dq_ptrs, offs_m, ld_offs_d, seqlen_q, head_dim)
    store_db = True
    if BIAS_TYPE == 0:
        B_block_ptr = 0
        DB_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h_q * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
        if (stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0:
            store_db = False
        # Still have to make one even if no_db = False
        # due to a limit of Triton: runtime branches must have identical data types.
        DB_block_ptr = tl.make_block_ptr(
                base=DB + off_h_q * stride_dbh + batch_index * stride_dbz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_dbm, stride_dbn),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    dq = bwd_kernel_dq_db_common(
        q, kt_ptrs, stride_kn, vt_ptrs, stride_vk, B_block_ptr,
        sm_scale, do,
        dq, DB_block_ptr, store_db,
        l_ptrs,
        D_ptrs,
        seqlen_q, seqlen_k,
        start_m,
        head_dim,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        max_seqlen_k,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD,
        BIAS_TYPE)
    mstore2d(dq,
             BLOCK_M,
             BLOCK_DMODEL,
             o_base=DQ,
             o_start_row=start_m,
             o_start_col=0,
             o_rows=seqlen_q,
             o_cols=head_dim,
             stride_row=stride_dqm,
             stride_col=stride_dqk)
    # tl.device_print('dq_offset ', dq_offset)
    # tl.device_print('stride_dqm ', stride_dqm)
    # tl.device_print('stride_dqk ', stride_dqk)
    # tl.device_print('head_dim ', head_dim)
    # tl.device_print('dq_ptrs ', dq_ptrs)
    # tl.device_print('dq_masks ', dq_masks)
