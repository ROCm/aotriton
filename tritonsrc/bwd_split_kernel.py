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
from dropout import dropout_mask, dropout_rng, dropout_offsets
from bwd_kernel_common import bwd_kernel_dk_dv_common, bwd_kernel_dq_db_common

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
    # This lower loop bound is because of the causal mask. We create a lower triangular
    # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
    # be ignored in the GEMM.
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k

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

    bwd_kernel_dk_dv_common(
        Q_block_ptr, KT_block_ptr, VT_block_ptr, B_block_ptr,
        sm_scale, DO_block_ptr,
        DK_block_ptr, DV_block_ptr,
        l_ptrs,
        D_ptrs,
        seqlen_q, seqlen_k,
        start_m,
        head_dim,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD,
        BIAS_TYPE)

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
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_zh * seqlen_q
    l_ptrs = L + off_zh * seqlen_q
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k

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
    store_db = True
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

    bwd_kernel_dq_db_common(
        Q_block_ptr, K_block_ptr, V_block_ptr, B_block_ptr,
        sm_scale, DO_block_ptr,
        DQ_block_ptr, DB_block_ptr, store_db,
        l_ptrs,
        D_ptrs,
        seqlen_q, seqlen_k,
        start_m,
        head_dim,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        BLOCK_M,
        BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT,
        PADDED_HEAD,
        BIAS_TYPE)
