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
from bwd_inner_fuse import bwd_inner_dk_dv_fuse, bwd_inner_dq
from masked_load_store import load_fn, mstore2d

# TODO: Remove Unused 'Out' Argument from kernels below
@triton.jit
def bwd_fuse_kernel(
    # I/O tensors
    Q, K, V, B, sm_scale, 
    Out, DO,
    DK, DV, DQ, DB,
    L,
    # Strides  
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    # Problem size
    num_head_q: tl.constexpr,
    num_head_k: tl.constexpr, 
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens: tl.constexpr,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    head_dim: tl.constexpr,
    # Dropout
    dropout_p: tl.constexpr,
    philox_seed_ptr,
    philox_offset1,
    philox_offset2: tl.constexpr,
    # Constants
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_k = tl.program_id(0) * BLOCK_N1  # start_k partitions seqlen_k
    off_h_k = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index, for varlen it indicates index in cu_seqlens_q/k
    num_z = tl.num_programs(2)
    offs_m = tl.arange(0, BLOCK_M1)
    offs_n = start_k + tl.arange(0, BLOCK_N1)
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
        if start_k >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        batch_index = 0

    if num_seqlens < 0:  # for padded seqlen
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_k >= seqlen_k:
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
    # previous block offset by BLOCK_M1 x D_HEAD.

    k_offset = off_h_k * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K_dkv = K + k_offset
    kt_ptrs = K_dkv + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # kt_offs_n = None if start_k + BLOCK_N1 <= seqlen_k else start_k + tl.arange(0, BLOCK_N1)
    if start_k + BLOCK_N1 <= seqlen_k:
        kt = load_fn(kt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
    else:
        kt = load_fn(kt_ptrs, ld_offs_d, offs_n, head_dim, seqlen_k)

    v_offset = off_h_k * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V_dkv = V + v_offset
    vt_ptrs = V_dkv + offs_d[:, None] * stride_vn + offs_n[None, :] * stride_vk
    if start_k + BLOCK_N1 <= seqlen_k:
        vt = load_fn(vt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
    else:
        vt = load_fn(vt_ptrs, ld_offs_d, offs_n, head_dim, seqlen_k)

    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        # CAVEAT: bias is incompatible with GQA
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h_k * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(0, start_k),
                block_shape=(BLOCK_M1, BLOCK_N1),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    dk_offset = off_h_k * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
    DK += dk_offset
    dv_offset = off_h_k * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk
    DV += dv_offset

    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
    group_size = num_head_q // num_head_k

    q_lo = start_k if CAUSAL else 0
    q_hi = seqlen_q
    real_seqlen_q = q_hi - q_lo  # seqlen_q after considering causal (and windowed in the future)
    n_blocks = tl.cdiv(q_hi - q_lo, BLOCK_M1)
    n_extra_tokens = 0
    if real_seqlen_q < BLOCK_M1:
        n_extra_tokens = BLOCK_M1 - real_seqlen_q
    elif real_seqlen_q % BLOCK_M1:
        n_extra_tokens = real_seqlen_q % BLOCK_M1
    is_irregular_q = n_extra_tokens != 0
    leading_masked_blocks = 0
    trailing_masked_blocks = 0
    if CAUSAL:
        # leading masked blocks comes from the diagnoal cutting line from causal masks
        # can be larger than one if BLOCK_N1 > BLOCK_M1
        leading_masked_blocks = tl.cdiv(BLOCK_N1, BLOCK_M1)
        # trailing masked blocks comes from extra tokens
        # Note trailing block may overlap with leading block
        trailing_masked_blocks = 1 if is_irregular_q else 0
    else:
        leading_masked_blocks = 0
        trailing_masked_blocks = 1 if is_irregular_q else 0
    n_full_blocks = n_blocks - leading_masked_blocks - trailing_masked_blocks

    dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
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
        l_ptrs = L + off_zh * max_seqlen_q

        q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
        do_ptrs = DO + do_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        o_ptrs = Out + do_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

        # dkdk kernel is a little tricky, its masked blocks can be found in both ends
        # leading masked: by causal
        # trailing: by irregular seqlen_q

        # For initialization
        lo = 0
        hi = 0

        if leading_masked_blocks > 0:
            lo = q_lo
            hi = lo + leading_masked_blocks * BLOCK_M1
            # TODO: overflow_size maybe larger than on block (BLOCK_M1)
            #       In this case the bwd_inner_dk_dv can be further optimized
            overflow_size = 0 if hi < q_hi else hi - q_hi
            dk, dv = bwd_inner_dk_dv_fuse(
                dk, dv, qk_scale, bias_scale,
                q_ptrs, stride_qm, kt, vt, B_block_ptr,
                do_ptrs, o_ptrs, stride_om,
                l_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                dropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M1,
                BLOCK_DMODEL,
                BLOCK_N1,
                False,  # FULL_BLOCKS
                CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)
            tl.debug_barrier()

        if n_full_blocks > 0:
            lo = q_lo + leading_masked_blocks * BLOCK_M1
            hi = lo + n_full_blocks * BLOCK_M1            
            dk, dv = bwd_inner_dk_dv_fuse(
                dk, dv, qk_scale, bias_scale,
                q_ptrs, stride_qm, kt, vt, B_block_ptr,
                do_ptrs, o_ptrs, stride_om,
                l_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, 0,
                dropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M1,
                BLOCK_DMODEL,
                BLOCK_N1,
                True,  # FULL_BLOCKS
                False,  # CAUSAL has zero effect for full blocks
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)

        # use n_full_blocks to confirm the trailing masked blocks is not overlapping with leading masked_blocks
        if n_full_blocks >= 0 and trailing_masked_blocks > 0:
            tl.debug_barrier()
            lo = q_lo + leading_masked_blocks * BLOCK_M1 + n_full_blocks * BLOCK_M1
            hi = q_hi
            overflow_size = lo + trailing_masked_blocks * BLOCK_M1 - q_hi
            dk, dv = bwd_inner_dk_dv_fuse(
                dk, dv, qk_scale, bias_scale,
                q_ptrs, stride_qm, kt, vt, B_block_ptr,
                do_ptrs, o_ptrs, stride_om,
                l_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                dropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M1,
                BLOCK_DMODEL,
                BLOCK_N1,
                False,  # FULL_BLOCKS
                CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)
    
    dk = (dk * sm_scale).to(kt.type.element_ty)
    dv = dv.to(vt.type.element_ty)
    mstore2d(dk,
            BLOCK_N1, 
            BLOCK_DMODEL,
            DK,
            start_k, 0,
            seqlen_k, head_dim,
            stride_dkn, stride_dkk)
    mstore2d(dv,
            BLOCK_N1,
            BLOCK_DMODEL, 
            DV,
            start_k, 0,
            seqlen_k, head_dim,
            stride_dvk, stride_dvn)
    # dq computation
    start_q = start_k
    if start_q >= seqlen_q:
        return
    off_h_q = tl.program_id(1) # head index
    off_h_k = off_h_q if num_head_q == num_head_k else off_h_q // (num_head_q // num_head_k)
    off_z = tl.program_id(2) # batch index
    num_z = tl.num_programs(2)
    off_zh = off_z * num_head_q + off_h_q * 1
    offs_q = start_q + tl.arange(0, BLOCK_M2)
    offs_n = tl.arange(0, BLOCK_N2)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    
    off_m = tl.program_id(0) * BLOCK_M2
    off_h = off_h_q # head index
    off_z = tl.program_id(2) # batch index
    num_h = tl.num_programs(1)
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_ok),
        offsets=(off_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )
    # load
    o = tl.load(O_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

    # Initialize pointers to Q, K, V
    q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q += q_offset
    # Q_block_ptr = tl.make_block_ptr(
    #     base=Q,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_qm, stride_qk),
    #     offsets=(start_q, 0),
    #     block_shape=(BLOCK_M2, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    q_ptrs = Q + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qk
    if start_q + BLOCK_M2 <= seqlen_q:
        q = load_fn(q_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        q = load_fn(q_ptrs, offs_q, ld_offs_d, seqlen_q, head_dim)
    k_offset = off_h_k * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K += k_offset
    kt_ptrs = K + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # K_block_ptr = tl.make_block_ptr(
    #     base=K,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_kk, stride_kn),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N2),
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
    #     block_shape=(BLOCK_DMODEL, BLOCK_N2),
    #     order=(0, 1)
    # )
    do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    DO += do_offset
    # DO_block_ptr = tl.make_block_ptr(
    #     base=DO,
    #     shape=(seqlen_q, head_dim),
    #     strides=(stride_om, stride_ok),
    #     offsets=(start_q, 0),
    #     block_shape=(BLOCK_M2, BLOCK_DMODEL),
    #     order=(1, 0)
    # )
    do_ptrs = DO + offs_q[:, None] * stride_om + offs_d[None, :] * stride_ok
    if start_q + BLOCK_M2 <= seqlen_q:
        do = load_fn(do_ptrs, None, ld_offs_d, seqlen_q, head_dim)
    else:
        do = load_fn(do_ptrs, offs_q, ld_offs_d, seqlen_q, head_dim)
    Di = tl.sum(o * do, axis=1)
        
    
    # pointer to row-wise quantities in value-like data
    # D_ptrs = D + off_zh * max_seqlen_q
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
    #     block_shape=(BLOCK_M2, BLOCK_DMODEL),
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
                block_shape=(BLOCK_M2, BLOCK_N2),
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
                block_shape=(BLOCK_M2, BLOCK_N2),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    k_lo = 0  # reserved for windowed attention
    k_hi = min(start_q + BLOCK_M2, seqlen_k) if CAUSAL else seqlen_k
    real_seqlen_k = k_hi - k_lo  # seqlen_q after considering causal (and windowed in the future)
    n_blocks = tl.cdiv(k_hi - k_lo, BLOCK_N2)
    n_extra_tokens = 0
    if real_seqlen_k < BLOCK_N2:
        n_extra_tokens = BLOCK_N2 - real_seqlen_k
    elif real_seqlen_k % BLOCK_N2:
        n_extra_tokens = real_seqlen_k % BLOCK_N2
    is_irregular_k = n_extra_tokens != 0
    n_full_blocks = (k_hi - k_lo) // BLOCK_N2
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
        n_full_blocks = (mask_top_edge - k_lo) // BLOCK_N2
        trailing_masked_blocks = n_blocks - n_full_blocks
    else:
        trailing_masked_blocks = 1 if is_irregular_k else 0

    # Check for OOB accesses on D and LSE
    q_boundary = tl.full((BLOCK_M2, ), seqlen_q, dtype=tl.int32)
    d_lse_ptrs_mask = offs_q < q_boundary
    # Di = tl.load(D_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)
    
    l_i = tl.load(l_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)

    dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
    if n_full_blocks > 0:
        lo = 0
        hi = n_full_blocks * BLOCK_N2
        dq = bwd_inner_dq(
            dq, qk_scale, bias_scale,
            DB_block_ptr, store_db,
            q, kt_ptrs, stride_kn, vt_ptrs, stride_vk, B_block_ptr,
            do,
            Di, l_i,
            seqlen_q, seqlen_k, head_dim,
            start_q, lo, hi,
            dropout_p, dropout_scale, philox_seed, batch_philox_offset, max_seqlen_k,
            BLOCK_M2,
            BLOCK_DMODEL,
            BLOCK_N2,
            True,  # FULL_BLOCKS
            False,  # CAUSAL has zero effect for full blocks
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE)
    # Keep using "trailing_masked_blocks" for windowed attention
    if trailing_masked_blocks > 0:
        lo = n_full_blocks * BLOCK_N2
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
            BLOCK_M2,
            BLOCK_DMODEL,
            BLOCK_N2,
            False,  # FULL_BLOCKS
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE)
    dq = (dq * sm_scale).to(dq.type.element_ty)
    mstore2d(dq,
             BLOCK_M2,
             BLOCK_DMODEL,
             o_base=DQ,
             o_start_row=start_q,
             o_start_col=0,
             o_rows=seqlen_q,
             o_cols=head_dim,
             stride_row=stride_dqm,
             stride_col=stride_dqk)

