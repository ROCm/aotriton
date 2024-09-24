#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from bwd_inner_dk_dv import bwd_inner_dk_dv
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
    SPLIT_DMODEL: tl.constexpr,
):
    HALVED_DMODEL : tl.constexpr = BLOCK_DMODEL // 2
    REG_DMODEL : tl.constexpr = HALVED_DMODEL if SPLIT_DMODEL else BLOCK_DMODEL
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_k = tl.program_id(0) * BLOCK_N  # start_k partitions seqlen_k
    off_h_k = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index, for varlen it indicates index in cu_seqlens_q/k
    num_z = tl.num_programs(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_k + tl.arange(0, BLOCK_N)
    if SPLIT_DMODEL:
        offs_d1 = tl.arange(0, HALVED_DMODEL)
        offs_d2 = tl.arange(HALVED_DMODEL, BLOCK_DMODEL)
        ld_offs_d1 = None  # head_dim is always >= HALVED_DMODEL, otherwise BLOCK_DMODEL can be halved
        ld_offs_d2 = None if not PADDED_HEAD else tl.arange(HALVED_DMODEL, BLOCK_DMODEL)
    else:
        offs_d1 = tl.arange(0, BLOCK_DMODEL)
        offs_d2 = 0
        ld_offs_d1 = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        ld_offs_d2 = 0  # 0 for "Unused", None for "No checking"

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

    # Still need early exit in GPU kernel to support varlen
    if CAUSAL:
        # TODO: bottom right causal and windowed
        if start_k > seqlen_q:
            return

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
    kt_ptrs1 = K + offs_d1[:, None] * stride_kk + offs_n[None, :] * stride_kn
    if SPLIT_DMODEL:
        kt_ptrs2 = K + offs_d2[:, None] * stride_kk + offs_n[None, :] * stride_kn
    else:
        kt_ptrs2 = 0
    # kt_offs_n = None if start_k + BLOCK_N <= seqlen_k else start_k + tl.arange(0, BLOCK_N)
    if SPLIT_DMODEL:
        if start_k + BLOCK_N <= seqlen_k:
            kt1 = load_fn(kt_ptrs1, ld_offs_d1, None, head_dim, seqlen_k)
            kt2 = load_fn(kt_ptrs2, ld_offs_d2, None, head_dim, seqlen_k)
        else:
            kt1 = load_fn(kt_ptrs1, ld_offs_d1, offs_n, head_dim, seqlen_k)
            kt2 = load_fn(kt_ptrs2, ld_offs_d2, offs_n, head_dim, seqlen_k)
    else:
        if start_k + BLOCK_N <= seqlen_k:
            kt1 = load_fn(kt_ptrs1, ld_offs_d1, None, head_dim, seqlen_k)
        else:
            kt1 = load_fn(kt_ptrs1, ld_offs_d1, offs_n, head_dim, seqlen_k)
        kt2 = 0
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
    vt_ptrs1 = V + offs_d1[:, None] * stride_vn + offs_n[None, :] * stride_vk
    if SPLIT_DMODEL:
        vt_ptrs2 = V + offs_d2[:, None] * stride_vn + offs_n[None, :] * stride_vk
    else:
        vt_ptrs2 = 0
    if SPLIT_DMODEL:
        if start_k + BLOCK_N <= seqlen_k:
            vt1 = load_fn(vt_ptrs1, ld_offs_d1, None, head_dim, seqlen_k)
            vt2 = load_fn(vt_ptrs2, ld_offs_d2, None, head_dim, seqlen_k)
        else:
            vt1 = load_fn(vt_ptrs1, ld_offs_d1, offs_n, head_dim, seqlen_k)
            vt2 = load_fn(vt_ptrs2, ld_offs_d2, offs_n, head_dim, seqlen_k)
    else:
        if start_k + BLOCK_N <= seqlen_k:
            vt1 = load_fn(vt_ptrs1, ld_offs_d1, None, head_dim, seqlen_k)
        else:
            vt1 = load_fn(vt_ptrs1, ld_offs_d1, offs_n, head_dim, seqlen_k)
        vt2 = 0
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
                offsets=(0, start_k),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    dk_offset = off_h_k * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
    DK += dk_offset
    dv_offset = off_h_k * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk
    DV += dv_offset

    if SPLIT_DMODEL:
        dv1 = tl.zeros([BLOCK_N, HALVED_DMODEL], dtype=tl.float32)
        dv2 = tl.zeros([BLOCK_N, HALVED_DMODEL], dtype=tl.float32)
        dk1 = tl.zeros([BLOCK_N, HALVED_DMODEL], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, HALVED_DMODEL], dtype=tl.float32)
    else:
        dv1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv2 = 0
        dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk2 = 0
    qk_scale = sm_scale * 1.44269504089
    kt1 = (kt1 * qk_scale).to(kt1.type.element_ty)
    if SPLIT_DMODEL:
        kt2 = (kt2 * qk_scale).to(kt2.type.element_ty)
    group_size = num_head_q // num_head_k

    q_lo = start_k if CAUSAL else 0
    q_hi = seqlen_q
    real_seqlen_q = q_hi - q_lo  # seqlen_q after considering causal (and windowed in the future)
    n_blocks = tl.cdiv(q_hi - q_lo, BLOCK_M)
    n_extra_tokens = 0
    if real_seqlen_q < BLOCK_M:
        n_extra_tokens = BLOCK_M - real_seqlen_q
    elif real_seqlen_q % BLOCK_M:
        n_extra_tokens = real_seqlen_q % BLOCK_M
    is_irregular_q = n_extra_tokens != 0
    leading_masked_blocks = 0
    trailing_masked_blocks = 0
    if CAUSAL:
        # leading masked blocks comes from the diagnoal cutting line from causal masks
        # can be larger than one if BLOCK_N > BLOCK_M
        leading_masked_blocks = tl.cdiv(BLOCK_N, BLOCK_M)
        # trailing masked blocks comes from extra tokens
        # Note trailing block may overlap with leading block
        trailing_masked_blocks = 1 if is_irregular_q else 0
    else:
        leading_masked_blocks = 0
        trailing_masked_blocks = 1 if is_irregular_q else 0
    n_full_blocks = n_blocks - leading_masked_blocks - trailing_masked_blocks

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
        q_ptrs1 = Q + q_offset + offs_m[:, None] * stride_qm + offs_d1[None, :] * stride_qk
        if SPLIT_DMODEL:
            q_ptrs2 = Q + q_offset + offs_m[:, None] * stride_qm + offs_d2[None, :] * stride_qk
        else:
            q_ptrs2 = 0
        do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
        do_ptrs1 = DO + do_offset + offs_m[:, None] * stride_om + offs_d1[None, :] * stride_ok
        if SPLIT_DMODEL:
            do_ptrs2 = DO + do_offset + offs_m[:, None] * stride_om + offs_d2[None, :] * stride_ok
        else:
            do_ptrs2 = 0

        # dkdk kernel is a little tricky, its masked blocks can be found in both ends
        # leading masked: by causal
        # trailing: by irregular seqlen_q

        # For initialization
        lo = 0
        hi = 0

        if leading_masked_blocks > 0:
            lo = q_lo
            hi = lo + leading_masked_blocks * BLOCK_M
            # TODO: overflow_size maybe larger than on block (BLOCK_M)
            #       In this case the bwd_inner_dk_dv can be further optimized
            overflow_size = 0 if hi < q_hi else hi - q_hi
            dk1, dk2, dv1, dv2 = bwd_inner_dk_dv(
                dk1, dk2,
                dv1, dv2,
                q_ptrs1, q_ptrs2, stride_qm,
                kt1, kt2, vt1, vt2,
                B_block_ptr,
                do_ptrs1, do_ptrs2, stride_om,
                l_ptrs,
                D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                False,  # FULL_BLOCKS
                CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE,
                SPLIT_DMODEL,
                )
            tl.debug_barrier()

        if n_full_blocks > 0:
            lo = q_lo + leading_masked_blocks * BLOCK_M
            hi = lo + n_full_blocks * BLOCK_M
            dk1, dk2, dv1, dv2 = bwd_inner_dk_dv(
                dk1, dk2,
                dv1, dv2,
                q_ptrs1, q_ptrs2, stride_qm,
                kt1, kt2, vt1, vt2,
                B_block_ptr,
                do_ptrs1, do_ptrs2, stride_om,
                l_ptrs,
                D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, 0,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                True,  # FULL_BLOCKS
                False,  # CAUSAL has zero effect for full blocks
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE,
                SPLIT_DMODEL,
                )

        # use n_full_blocks to confirm the trailing masked blocks is not overlapping with leading masked_blocks
        if n_full_blocks >= 0 and trailing_masked_blocks > 0:
            tl.debug_barrier()
            lo = q_lo + leading_masked_blocks * BLOCK_M + n_full_blocks * BLOCK_M
            hi = q_hi
            overflow_size = lo + trailing_masked_blocks * BLOCK_M - q_hi
            dk1, dk2, dv1, dv2 = bwd_inner_dk_dv(
                dk1, dk2,
                dv1, dv2,
                q_ptrs1, q_ptrs2, stride_qm,
                kt1, kt2, vt1, vt2,
                B_block_ptr,
                do_ptrs1, do_ptrs2, stride_om,
                l_ptrs,
                D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M,
                BLOCK_DMODEL,
                BLOCK_N,
                False,  # FULL_BLOCKS
                CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE,
                SPLIT_DMODEL,
                )
    dk1 = (dk1 * sm_scale).to(kt1.type.element_ty)
    dv1 = dv1.to(vt1.type.element_ty)
    if SPLIT_DMODEL:
        dk2 = (dk2 * sm_scale).to(kt2.type.element_ty)
        dv2 = dv2.to(vt2.type.element_ty)
    mstore2d(dk1,
             BLOCK_N,
             REG_DMODEL,
             o_base=DK,
             o_start_row=start_k,
             o_start_col=0,
             o_rows=seqlen_k,
             o_cols=head_dim,
             stride_row=stride_dkn,
             stride_col=stride_dkk)
    if SPLIT_DMODEL:
        mstore2d(dk2,
                 BLOCK_N,
                 REG_DMODEL,
                 o_base=DK,
                 o_start_row=start_k,
                 o_start_col=HALVED_DMODEL,
                 o_rows=seqlen_k,
                 o_cols=head_dim,
                 stride_row=stride_dkn,
                 stride_col=stride_dkk)
    mstore2d(dv1,
             BLOCK_N,
             REG_DMODEL,
             o_base=DV,
             o_start_row=start_k,
             o_start_col=0,
             o_rows=seqlen_k,
             o_cols=head_dim,
             stride_row=stride_dvk,
             stride_col=stride_dvn)
    if SPLIT_DMODEL:
        mstore2d(dv2,
                 BLOCK_N,
                 REG_DMODEL,
                 o_base=DV,
                 o_start_row=start_k,
                 o_start_col=HALVED_DMODEL,
                 o_rows=seqlen_k,
                 o_cols=head_dim,
                 stride_row=stride_dvk,
                 stride_col=stride_dvn)
