#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
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
from bwd_inner_dk_dv import bwd_inner_dk_dv
from dropout import PHILOX_RN_PER_OFFSET
from masked_load_store import load_fn, mstore2d
from composed_tensors import (
    composed_offs_1d,
    composed_zeros_2d,
    composed_ptrs,
    composed_load,
    composed_advance,
    composed_to,
    composed_store,
    composed_mul_lhs,
    composed_dot_both,
    composed_dot_rhs,
    composed_mul_lhs,
    composed_mul_acc,
)

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
    dropout_p : tl.float32,
    philox_seed_ptr,
    philox_offset1 : '*u64',
    philox_offset2 : 'u64',
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    tl.static_assert(BLOCK_DMODEL > 0, 'BLOCK_DMODEL must be greater than 0')
    BLOCK_DMODEL_R0 : tl.constexpr = BLOCK_DMODEL
    BLOCK_DMODEL0 : tl.constexpr = 2 ** (BLOCK_DMODEL_R0.bit_length() - 1)
    BLOCK_DMODEL_R1 : tl.constexpr = BLOCK_DMODEL_R0 - BLOCK_DMODEL0
    BLOCK_DMODEL1 : tl.constexpr = 2 ** (BLOCK_DMODEL_R1.bit_length() - 1) if BLOCK_DMODEL_R1 > 0 else 0
    BLOCK_DMODEL_R2 : tl.constexpr = BLOCK_DMODEL_R1 - BLOCK_DMODEL1
    BLOCK_DMODEL2 : tl.constexpr = 2 ** (BLOCK_DMODEL_R2.bit_length() - 1) if BLOCK_DMODEL_R2 > 0 else 0
    BLOCK_DMODEL_R3 : tl.constexpr = BLOCK_DMODEL_R2 - BLOCK_DMODEL2

    tl.static_assert(BLOCK_DMODEL_R3 == 0, f'BLOCK_DMODEL = {BLOCK_DMODEL} = 0b{BLOCK_DMODEL:b} cannot be factored into <= 3 power of two values')
    tl.static_assert(BLOCK_DMODEL1 > 0 or BLOCK_DMODEL2 == 0, 'Only trailing BLOCK_DMODELx can be 0')

    idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)
    philox_seed = 0
    philox_offset_base = philox_offset2
    philox_offset_stride = tl.cdiv(max_seqlen_k, PHILOX_RN_PER_OFFSET)
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    start_k = tl.program_id(0) * BLOCK_N  # start_k partitions seqlen_k
    off_h_k = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index, for varlen it indicates index in cu_seqlens_q/k
    num_z = tl.num_programs(2)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_k + tl.arange(0, BLOCK_N)
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

    k_ptrs0, k_ptrs1, k_ptrs2 = composed_ptrs(K,
                                              stride_kz, stride_kh, stride_kn, stride_kk,
                                              batch_index, off_h_k, cu_seqlens_k_start + offs_n,
                                              BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                              TRANSPOSED=True)
    # kt_offs_n = None if start_k + BLOCK_N <= seqlen_k else start_k + tl.arange(0, BLOCK_N)
    v_ptrs0, v_ptrs1, v_ptrs2 = composed_ptrs(V,
                                              stride_vz, stride_vh, stride_vk, stride_vn,
                                              batch_index, off_h_k, cu_seqlens_k_start + offs_n,
                                              BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                              TRANSPOSED=True)

    if start_k + BLOCK_N <= seqlen_k:
        kt0, kt1, kt2 = composed_load(k_ptrs0, k_ptrs1, k_ptrs2,
                                      offs_n,
                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                      seqlen_k, head_dim,
                                      other=0.0,
                                      PADDED_ROW=False,
                                      PADDED_COL=PADDED_HEAD,
                                      TRANSPOSED=True)
        vt0, vt1, vt2 = composed_load(v_ptrs0, v_ptrs1, v_ptrs2,
                                      offs_n,
                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                      seqlen_k, head_dim,
                                      other=0.0,
                                      PADDED_ROW=False,
                                      PADDED_COL=PADDED_HEAD,
                                      TRANSPOSED=True)
    else:
        kt0, kt1, kt2 = composed_load(k_ptrs0, k_ptrs1, k_ptrs2,
                                      offs_n,
                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                      seqlen_k, head_dim,
                                      other=0.0,
                                      PADDED_ROW=True,
                                      PADDED_COL=PADDED_HEAD,
                                      TRANSPOSED=True)
        vt0, vt1, vt2 = composed_load(v_ptrs0, v_ptrs1, v_ptrs2,
                                      offs_n,
                                      BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                      seqlen_k, head_dim,
                                      other=0.0,
                                      PADDED_ROW=True,
                                      PADDED_COL=PADDED_HEAD,
                                      TRANSPOSED=True)
    # KT_block_ptr = tl.make_block_ptr(
    #     base=K + k_offset,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_kk, stride_kn),
    #     offsets=(0, start_m),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1)
    # )

    # VT_block_ptr = tl.make_block_ptr(
    #     base=V,
    #     shape=(head_dim, seqlen_k),
    #     strides=(stride_vn, stride_vk),
    #     offsets=(0, start_m),
    #     block_shape=(BLOCK_DMODEL, BLOCK_N),
    #     order=(0, 1)
    # )
    # vt = tl.load(VT_block_ptr)
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

    dv0, dv1, dv2 = composed_zeros_2d(BLOCK_N, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    dk0, dk1, dk2 = composed_zeros_2d(BLOCK_N, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
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

    dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
    for off_h_q in range(off_h_k * group_size, off_h_k * group_size + group_size):
        off_zh = off_z * num_head_q + off_h_q * 1
        # This lower loop bound is because of the causal mask. We create a lower triangular
        # result. The upper triangular is -inf (becomes 0 when we do e^x). As such, it can
        # be ignored in the GEMM.
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * philox_offset_stride
        else:
            batch_philox_offset = 0
        # pointer to row-wise quantities in value-like data
        # Shape (batch, num_heads, max_seqlen_q)
        # In varlen cases, batch == len(cu_seqlens_q) - 1).
        # Hence off_z plays the same role in varlen/non-varlen
        D_ptrs = D + off_zh * max_seqlen_q
        l_ptrs = L + off_zh * max_seqlen_q

        q_ptrs0, q_ptrs1, q_ptrs2 = composed_ptrs(Q,
                                                  stride_qz, stride_qh, stride_qm, stride_qk,
                                                  batch_index, off_h_q, cu_seqlens_q_start + offs_m,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

        do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(DO,
                                                  stride_oz, stride_oh, stride_om, stride_ok,
                                                  batch_index, off_h_q, cu_seqlens_q_start + offs_m,
                                                  BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

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
            dk0, dk1, dk2, dv0, dv1, dv2 = bwd_inner_dk_dv(
                dk0, dk1, dk2,
                dv0, dv1, dv2,
                qk_scale, bias_scale,
                q_ptrs0, q_ptrs1, q_ptrs2,
                stride_qm,
                kt0, kt1, kt2, vt0, vt1, vt2,
                B_block_ptr,
                do_ptrs0, do_ptrs1, do_ptrs2,
                stride_om,
                l_ptrs,
                D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                BLOCK_M,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_N,
                False,  # FULL_BLOCKS
                CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)
            tl.debug_barrier()

        if n_full_blocks > 0:
            lo = q_lo + leading_masked_blocks * BLOCK_M
            hi = lo + n_full_blocks * BLOCK_M
            dk0, dk1, dk2, dv0, dv1, dv2 = bwd_inner_dk_dv(
                dk0, dk1, dk2,
                dv0, dv1, dv2,
                qk_scale, bias_scale,
                q_ptrs0, q_ptrs1, q_ptrs2,
                stride_qm,
                kt0, kt1, kt2, vt0, vt1, vt2,
                B_block_ptr,
                do_ptrs0, do_ptrs1, do_ptrs2,
                stride_om,
                l_ptrs,
                D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, 0,
                idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                BLOCK_M,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_N,
                True,  # FULL_BLOCKS
                False,  # CAUSAL has zero effect for full blocks
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)

        # use n_full_blocks to confirm the trailing masked blocks is not overlapping with leading masked_blocks
        if n_full_blocks >= 0 and trailing_masked_blocks > 0:
            tl.debug_barrier()
            lo = q_lo + leading_masked_blocks * BLOCK_M + n_full_blocks * BLOCK_M
            hi = q_hi
            overflow_size = lo + trailing_masked_blocks * BLOCK_M - q_hi
            dk0, dk1, dk2, dv0, dv1, dv2 = bwd_inner_dk_dv(
                dk0, dk1, dk2,
                dv0, dv1, dv2,
                qk_scale, bias_scale,
                q_ptrs0, q_ptrs1, q_ptrs2,
                stride_qm,
                kt0, kt1, kt2, vt0, vt1, vt2,
                B_block_ptr,
                do_ptrs0, do_ptrs1, do_ptrs2,
                stride_om,
                l_ptrs,
                D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                BLOCK_M,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_N,
                False,  # FULL_BLOCKS
                CAUSAL,
                ENABLE_DROPOUT,
                PADDED_HEAD,
                BIAS_TYPE)

    dk0, dk1, dk2 = composed_mul_lhs(dk0, dk1, dk2,
                                     sm_scale,
                                     BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    dk0, dk1, dk2 = composed_to(dk0, dk1, dk2, kt0.type.element_ty)
    dv0, dv1, dv2 = composed_to(dv0, dv1, dv2, vt0.type.element_ty)

    composed_store(dk0, dk1, dk2,
                   BLOCK_N,
                   BLOCK_DMODEL0,
                   BLOCK_DMODEL1,
                   BLOCK_DMODEL2,
                   o_base=DK,
                   o_start_row=start_k,
                   o_start_col=0,
                   o_rows=seqlen_k,
                   o_cols=head_dim,
                   stride_row=stride_dkn,
                   stride_col=stride_dkk)

    composed_store(dv0, dv1, dv2,
                   BLOCK_N,
                   BLOCK_DMODEL0,
                   BLOCK_DMODEL1,
                   BLOCK_DMODEL2,
                   o_base=DV,
                   o_start_row=start_k,
                   o_start_col=0,
                   o_rows=seqlen_k,
                   o_cols=head_dim,
                   stride_row=stride_dvk,
                   stride_col=stride_dvn)

