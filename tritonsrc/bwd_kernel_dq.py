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
from bwd_inner_dq import bwd_inner_dq
from masked_load_store import load_fn, mstore2d

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
    start_q = tl.program_id(0) * BLOCK_M
    off_h_q = tl.program_id(1) # head index
    off_h_k = off_h_q if num_head_q == num_head_k else off_h_q // (num_head_q // num_head_k)
    off_z = tl.program_id(2) # batch index
    num_z = tl.num_programs(2)
    off_zh = off_z * num_head_q + off_h_q * 1
    offs_q = start_q + tl.arange(0, BLOCK_M)
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
        if start_q >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0

    if num_seqlens < 0:  # for padded seqlen
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_q >= seqlen_q:
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
    #     offsets=(start_q, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    q_ptrs = Q + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
    if start_q + BLOCK_M <= seqlen_q:
        q = load_fn(q_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        q = load_fn(q_ptrs, offs_q, ld_offs_d, seqlen_q, head_dim)
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
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
    #     offsets=(start_q, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    do_ptrs = DO + offs_q[:, None] * stride_om + offs_d[None, :] * stride_ok
    if start_q + BLOCK_M <= seqlen_q:
        do = load_fn(do_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        do = load_fn(do_ptrs, offs_q, ld_offs_d, seqlen_q, head_dim)
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
    #     offsets=(start_q, 0),
    #     block_shape=(BLOCK_M, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    store_db = True
    if BIAS_TYPE == 0:
        B_block_ptr = 0
        DB_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h_q * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_q, 0),
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
                offsets=(start_q, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    k_lo = 0  # reserved for windowed attention
    k_hi = min(start_q + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k
    real_seqlen_k = k_hi - k_lo  # seqlen_q after considering causal (and windowed in the future)
    n_blocks = tl.cdiv(k_hi - k_lo, BLOCK_N)
    n_extra_tokens = 0
    if real_seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - real_seqlen_k
    elif real_seqlen_k % BLOCK_N:
        n_extra_tokens = real_seqlen_k % BLOCK_N
    is_irregular_k = n_extra_tokens != 0
    n_full_blocks = (k_hi - k_lo) // BLOCK_N
    leading_masked_blocks = 0  # TODO: Windowed attention
    trailing_masked_blocks = 0
    # For causal masks, actually it is easier to calculate the full blocks and
    # then derive trailing_masked_blocks. However this algorithm won't work for
    # windowed masks. Therefore we still derive n_full_blocks from
    # trailing_masked_blocks for long term stability.
    if CAUSAL:
        # TODO: Botton right variant
        # Top left variant
        mask_top_edge = min(start_q, seqlen_k)
        n_full_blocks = (mask_top_edge - k_lo) // BLOCK_N
        trailing_masked_blocks = n_blocks - n_full_blocks
    else:
        trailing_masked_blocks = 1 if is_irregular_k else 0

    # Check for OOB accesses on D and LSE
    q_boundary = tl.full((BLOCK_M, ), seqlen_q, dtype=tl.int32)
    d_lse_ptrs_mask = offs_q < q_boundary
    Di = tl.load(D_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)
    l_i = tl.load(l_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)

    dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if n_full_blocks > 0:
        lo = 0
        hi = n_full_blocks * BLOCK_N
        dq = bwd_inner_dq(
            dq, qk_scale, bias_scale,
            DB_block_ptr, store_db,
            q, kt_ptrs, stride_kn, vt_ptrs, stride_vk, B_block_ptr,
            do,
            Di, l_i,
            seqlen_q, seqlen_k, head_dim,
            start_q, lo, hi,
            dropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            True,  # FULL_BLOCKS
            False,  # CAUSAL has zero effect for full blocks
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE)
    # Keep using "trailing_masked_blocks" for windowed attention
    if trailing_masked_blocks > 0:
        lo = n_full_blocks * BLOCK_N
        hi = k_hi
        tl.debug_barrier()
        dq = bwd_inner_dq(
            dq, qk_scale, bias_scale,
            DB_block_ptr, store_db,
            q, kt_ptrs, stride_kn, vt_ptrs, stride_vk, B_block_ptr,
            do,
            Di, l_i,
            seqlen_q, seqlen_k, head_dim,
            start_q, lo, hi,
            dropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            False,  # FULL_BLOCKS
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE)
    dq = (dq * sm_scale).to(dq.type.element_ty)
    mstore2d(dq,
             BLOCK_M,
             BLOCK_DMODEL,
             o_base=DQ,
             o_start_row=start_q,
             o_start_col=0,
             o_rows=seqlen_q,
             o_cols=head_dim,
             stride_row=stride_dqm,
             stride_col=stride_dqk)
