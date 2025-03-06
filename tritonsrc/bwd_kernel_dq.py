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
from fwd_kernel_inner import (
    IS_JIT_COMPILING,
    constexpr_or_f32,
    constexpr_or_i32,
)
from bwd_inner_dq import bwd_inner_dq
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
    composed_mul_acc,
)

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
    dropout_p : tl.float32,
    philox_seed_ptr,
    philox_offset1 : '*u64',
    philox_offset2 : 'u64',
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL_TYPE: tl.constexpr,  # 0: disable, 1: top-left aligned, 2: bottom-right aligned
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    PERSISTENT_TYPE: tl.constexpr,  # 0: disable, 1: fixed, 2: dynamic
    persistent_atomic_counter,
    Num_CU: constexpr_or_i32,
    GRID_CU_MULTIP: tl.constexpr,
    Batch: constexpr_or_i32,
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
    
    # Adaptor code for AOTriton, to minimize main body code change
    ## tl.constexpr to variable
    IS_CAUSAL : tl.constexpr = CAUSAL_TYPE != 0
    IS_CAUSAL_BOTTOM_RIGHT : tl.constexpr = CAUSAL_TYPE == 2
    USE_BIAS : tl.constexpr = BIAS_TYPE == 1
    PERSISTENT : tl.constexpr = (PERSISTENT_TYPE > 0)
    PERSISTENT_DYNAMIC : tl.constexpr = (PERSISTENT_TYPE == 2)
    tl.static_assert(BIAS_TYPE == 0 or BIAS_TYPE == 1, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    BATCH = tl.num_programs(2)

    # Initialize philox for dropout
    philox_seed = 0
    philox_offset_base = philox_offset2
    philox_offset_stride = tl.cdiv(max_seqlen_k, PHILOX_RN_PER_OFFSET)
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)

    tile_id = 0
    num_tiles_total = 1
    num_tiles_per_head = 1
    num_tiles_per_sample = 1
    Num_WG = Num_CU * GRID_CU_MULTIP  # number of workgroups launched
    unsupported_by_persistent = num_seqlens != 0
    if PERSISTENT and not unsupported_by_persistent:  # Only enable for non-varlen
        # if persistent, kernel loops over multiple tiles
        num_tiles_per_head = tl.cdiv(max_seqlen_q, BLOCK_M)  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * num_head_q  # times the number of heads
        num_tiles_total = num_tiles_per_sample * Batch  # times the number of samples
        if PERSISTENT_DYNAMIC:
            tile_id = persistent_atomic_counter.atomic_add(1)  # returns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)

    continue_condition : tl.int1 = True  # as we can't have return statements inside while loop in Triton

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT
        if PERSISTENT_DYNAMIC and not unsupported_by_persistent:
            # tile id basically tells us the Q block we are handling
            off_z = tile_id // num_tiles_per_sample  # at which batch sample are we
            off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head  # at which head are we inside the sample
            start_q = tile_id % num_tiles_per_sample % num_tiles_per_head * BLOCK_M  # at which tile are we inside the head
        else:
            start_q = tl.program_id(0) * BLOCK_M
            off_h_q = tl.program_id(1) # head index
            off_z = tl.program_id(2) # batch index

        # Get MQA/GQA head offset
        off_h_k = off_h_q if num_head_q == num_head_k else off_h_q // (num_head_q // num_head_k)
        off_zh = off_z * num_head_q + off_h_q * 1
        offs_q = start_q + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        
        # Initialize sequence length variables
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
                continue_condition = False
            else:
                cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
                cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
                seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
                batch_index = 0

        elif num_seqlens < 0:  # for padded seqlen
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            if start_q >= seqlen_q:
                continue_condition = False
            else:
                cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
                cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
                seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
                # Varlen, but padded to Rank 4 tensor
                cu_seqlens_q_start = 0
                cu_seqlens_k_start = 0
                batch_index = off_z

        if continue_condition:
            # Initialize pointers to Q, K, V
            # Q_block_ptr = tl.make_block_ptr(
            #     base=Q,
            #     shape=(seqlen_q, head_dim),
            #     strides=(stride_qm, stride_qk),
            #     offsets=(start_q, 0),
            #     block_shape=(BLOCK_M, BLOCK_DMODEL),
            #     order=(1, 0)
            # )
            q_ptrs0, q_ptrs1, q_ptrs2 = composed_ptrs(Q,
                                                    stride_qz, stride_qh, stride_qm, stride_qk,
                                                    batch_index, off_h_q, cu_seqlens_q_start + offs_q,
                                                    BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
            if start_q + BLOCK_M <= seqlen_q:
                q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                        offs_q,
                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                        seqlen_q, head_dim,
                                        other=0.0,
                                        PADDED_ROW=False,
                                        PADDED_COL=PADDED_HEAD,
                                        TRANSPOSED=False)
            else:
                q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                        offs_q,
                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                        seqlen_q, head_dim,
                                        other=0.0,
                                        PADDED_ROW=True,
                                        PADDED_COL=PADDED_HEAD,
                                        TRANSPOSED=False)
            qk_scale = sm_scale * 1.44269504089
            bias_scale = 1.0 / sm_scale
            
            # Initialize pointers to K and V
            kt_ptrs0, kt_ptrs1, kt_ptrs2 = composed_ptrs(K,
                                                        stride_kz, stride_kh, stride_kn, stride_kk,
                                                        batch_index, off_h_k, cu_seqlens_k_start + offs_n,
                                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                        TRANSPOSED=True)
            # K_block_ptr = tl.make_block_ptr(
            #     base=K,
            #     shape=(head_dim, seqlen_k),
            #     strides=(stride_kk, stride_kn),
            #     offsets=(0, 0),
            #     block_shape=(BLOCK_DMODEL, BLOCK_N),
            #     order=(0, 1)
            # )

            vt_ptrs0, vt_ptrs1, vt_ptrs2 = composed_ptrs(V,
                                                        stride_vz, stride_vh, stride_vk, stride_vn,
                                                        batch_index, off_h_k, cu_seqlens_k_start + offs_n,
                                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                        TRANSPOSED=True)

            do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(DO,
                                                        stride_oz, stride_oh, stride_om, stride_ok,
                                                        batch_index, off_h_q, cu_seqlens_q_start + offs_q,
                                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
            if start_q + BLOCK_M <= seqlen_q:
                do0, do1, do2 = composed_load(do_ptrs0, do_ptrs1, do_ptrs2,
                                            offs_q,
                                            BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                            seqlen_q, head_dim,
                                            other=0.0,
                                            PADDED_ROW=False,
                                            PADDED_COL=PADDED_HEAD,
                                            TRANSPOSED=False)
            else:
                do0, do1, do2 = composed_load(do_ptrs0, do_ptrs1, do_ptrs2,
                                            offs_q,
                                            BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                            seqlen_q, head_dim,
                                            other=0.0,
                                            PADDED_ROW=True,
                                            PADDED_COL=PADDED_HEAD,
                                            TRANSPOSED=False)
            
            # Pointer to row-wise quantities in value-like data
            D_ptrs = D + off_zh * max_seqlen_q
            l_ptrs = L + off_zh * max_seqlen_q
            if ENABLE_DROPOUT:
                batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * philox_offset_stride
            else:
                batch_philox_offset = 0

            # Initialize pointers to output
            dq_offset = batch_index * stride_dqz + off_h_q * stride_dqh + cu_seqlens_q_start * stride_dqm
            DQ_ptr = DQ + dq_offset
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
                B_ptr = 0
                DB_ptr = 0
            elif BIAS_TYPE == 1:
                # B_block_ptr = tl.make_block_ptr(
                #         base=B + off_h_q * stride_bh + batch_index * stride_bz,
                #         shape=(seqlen_q, seqlen_k),
                #         strides=(stride_bm, stride_bn),
                #         offsets=(start_q, 0),
                #         block_shape=(BLOCK_M, BLOCK_N),
                #         order=(1, 0)
                #         )
                B_ptr = B + off_h_q * stride_bh + batch_index * stride_bz
                if (stride_dbz == 0 and stride_dbh == 0) and stride_dbm == 0:
                    store_db = False
                # Still have to make one even if no_db = False
                # due to a limit of Triton: runtime branches must have identical data types.
                # DB_block_ptr = tl.make_block_ptr(
                #         base=DB + off_h_q * stride_dbh + batch_index * stride_dbz,
                #         shape=(seqlen_q, seqlen_k),
                #         strides=(stride_dbm, stride_dbn),
                #         offsets=(start_q, 0),
                #         block_shape=(BLOCK_M, BLOCK_N),
                #         order=(1, 0)
                #         )
                DB_ptr = DB + off_h_q * stride_dbh + batch_index * stride_dbz
            else:
                tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

            # Calculate causal boundaries
            k_lo = 0  # reserved for windowed attention
            k_hi = min(start_q + BLOCK_M, seqlen_k) if IS_CAUSAL else seqlen_k
            if IS_CAUSAL_BOTTOM_RIGHT:
                causal_offset = seqlen_q - seqlen_k if seqlen_q > seqlen_k else 0
                k_hi = min(start_q + BLOCK_M - causal_offset, seqlen_k) if IS_CAUSAL else seqlen_k

            real_seqlen_k = k_hi - k_lo  # seqlen_q after considering causal (and windowed in the future)
            if real_seqlen_k <= 0:
                dq0, dq1, dq2 = composed_zeros_2d(BLOCK_M, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                dq0, dq1, dq2 = composed_mul_lhs(dq0, dq1, dq2,
                                                sm_scale,
                                                BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                dq0, dq1, dq2 = composed_to(dq0, dq1, dq2, dq0.type.element_ty)
                composed_store(dq0, dq1, dq2,
                               BLOCK_M,
                               BLOCK_DMODEL0,
                               BLOCK_DMODEL1,
                               BLOCK_DMODEL2,
                               o_base=DQ_ptr,
                               o_start_row=start_q,
                               o_start_col=0,
                               o_rows=seqlen_q,
                               o_cols=head_dim,
                               stride_row=stride_dqm,
                               stride_col=stride_dqk)
                continue_condition = False
            else:
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
                if IS_CAUSAL:
                    if IS_CAUSAL_BOTTOM_RIGHT:
                        # Bottom right variant
                        if start_q >= causal_offset:
                            mask_top_edge = min(start_q - causal_offset, seqlen_k)
                        else:
                            mask_top_edge = 0
                    else:
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

                idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)
                dropout_scale = 1.0 / (1.0 - dropout_p) if ENABLE_DROPOUT else 1.0
                dq0, dq1, dq2 = composed_zeros_2d(BLOCK_M, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                n_full_blocks = n_blocks - leading_masked_blocks - trailing_masked_blocks
                if n_full_blocks > 0:
                    lo = 0
                    hi = n_full_blocks * BLOCK_N
                    dq0, dq1, dq2 = bwd_inner_dq(
                        dq0, dq1, dq2,
                        qk_scale, bias_scale,
                        DB_ptr, store_db,
                        q0, q1, q2,
                        kt_ptrs0, kt_ptrs1, kt_ptrs2,
                        stride_kn,
                        vt_ptrs0, vt_ptrs1, vt_ptrs2,
                        stride_vk,
                        stride_bn, stride_bm, stride_dbn, stride_dbm,
                        B_ptr,
                        do0, do1, do2,
                        Di, l_i,
                        seqlen_q, seqlen_k, head_dim,
                        start_q, lo, hi,
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
                # Keep using "trailing_masked_blocks" for windowed attention
                if trailing_masked_blocks > 0:
                    lo = n_full_blocks * BLOCK_N
                    hi = k_hi
                    tl.debug_barrier()
                    dq0, dq1, dq2 = bwd_inner_dq(
                        dq0, dq1, dq2,
                        qk_scale, bias_scale,
                        DB_ptr, store_db,
                        q0, q1, q2,
                        kt_ptrs0, kt_ptrs1, kt_ptrs2,
                        stride_kn,
                        vt_ptrs0, vt_ptrs1, vt_ptrs2,
                        stride_vk,
                        stride_bn, stride_bm, stride_dbn, stride_dbm,
                        B_ptr,
                        do0, do1, do2,
                        Di, l_i,
                        seqlen_q, seqlen_k, head_dim,
                        start_q, lo, hi,
                        idropout_p, dropout_scale, philox_seed, batch_philox_offset, philox_offset_stride,
                        BLOCK_M,
                        BLOCK_DMODEL0,
                        BLOCK_DMODEL1,
                        BLOCK_DMODEL2,
                        BLOCK_N,
                        False,  # FULL_BLOCKS
                        IS_CAUSAL,
                        ENABLE_DROPOUT,
                        PADDED_HEAD,
                        BIAS_TYPE)
                
                # Apply scale and store result
                dq0, dq1, dq2 = composed_mul_lhs(dq0, dq1, dq2,
                                                sm_scale,
                                                BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                dq0, dq1, dq2 = composed_to(dq0, dq1, dq2, dq0.type.element_ty)
                composed_store(dq0, dq1, dq2,
                               BLOCK_M,
                               BLOCK_DMODEL0,
                               BLOCK_DMODEL1,
                               BLOCK_DMODEL2,
                               o_base=DQ_ptr,
                               o_start_row=start_q,
                               o_start_col=0,
                               o_rows=seqlen_q,
                               o_cols=head_dim,
                               stride_row=stride_dqm,
                               stride_col=stride_dqk)

        if not PERSISTENT or unsupported_by_persistent:
            tile_id = num_tiles_total  # break after single tile
        else:
            if PERSISTENT_DYNAMIC:
                tile_id = persistent_atomic_counter.atomic_add(1)
            else:
                tile_id += Num_WG
