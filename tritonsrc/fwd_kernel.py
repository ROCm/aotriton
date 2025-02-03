#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm
See https://tridao.me/publications/flash2/flash2.pdf

Credits:
AMD Triton kernels team
OpenAI kernel team
"""

import triton
import triton.language as tl
from fwd_kernel_inner import (
    _attn_fwd_inner,
    IS_JIT_COMPILING,
    constexpr_or_f32,
    constexpr_or_i32,
)
from masked_load_store import mstore2d
from composed_tensors import (
    composed_offs_1d,
    composed_zeros_2d,
    composed_ptrs,
    composed_load,
    composed_advance,
    composed_to,
    composed_store,
    composed_mul_lhs,
    composed_casual_mask,
)
import os

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def attn_fwd(
        # Basic SDPA
        Q, K, V, B, A, Sm_scale : constexpr_or_f32, L, Out,
        Q_descale, K_descale, P_scale, P_descale, V_descale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        stride_bz, stride_bh, stride_bm, stride_bn,
        stride_az, stride_ah,
        # MQA/GQA
        Num_head_q : constexpr_or_i32,
        Num_head_k : constexpr_or_i32,
        # Varlen
        Num_seqlens : constexpr_or_i32,
        cu_seqlens_q,
        cu_seqlens_k,
        Max_seqlen_q : constexpr_or_i32,
        Max_seqlen_k : constexpr_or_i32,
        # Head Dimensions
        BLOCK_DMODEL: tl.constexpr,
        Head_dim : constexpr_or_i32,
        PADDED_HEAD: tl.constexpr,
        # dropout and PRNG
        ENABLE_DROPOUT: tl.constexpr,
        dropout_p,
        philox_seed_ptr : '*u64',
        philox_offset1 : '*u32',
        philox_offset2 : tl.int32,  # TODO: move to tl.int64
        philox_seed_output : '*u64',
        philox_offset_output : '*u64',
        RETURN_ENCODED_SOFTMAX: tl.constexpr,
        encoded_softmax,
        # causal, (Planned Feature) windowed attention
        CAUSAL_TYPE: tl.constexpr,
        # bias
        BIAS_TYPE: tl.constexpr,
        # alibi
        USE_ALIBI: tl.constexpr,
        # INT8
        INT8: tl.constexpr,
        INT8_KV: tl.constexpr,
        USE_P_SCALE: tl.constexpr,
        # Persistent related arguments
        PERSISTENT_TYPE: tl.constexpr,  # 0: disable, 1: fixed, 2: dynamic
        persistent_atomic_counter,
        Num_CU : constexpr_or_i32,
        GRID_CU_MULTIP: tl.constexpr,
        Batch : constexpr_or_i32,
        # Performance
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PRE_LOAD_V: tl.constexpr,
        ):
    # TODO: Put this decomposition into a @triton.jit function when tuple support is more complete
    tl.static_assert(BLOCK_DMODEL > 0, 'BLOCK_DMODEL must be greater than 0')
    BLOCK_DMODEL_R0 : tl.constexpr = BLOCK_DMODEL
    BLOCK_DMODEL0 : tl.constexpr = 2 ** (BLOCK_DMODEL_R0.bit_length() - 1)
    BLOCK_DMODEL_R1 : tl.constexpr = BLOCK_DMODEL_R0 - BLOCK_DMODEL0
    BLOCK_DMODEL1 : tl.constexpr = 2 ** (BLOCK_DMODEL_R1.bit_length() - 1) if BLOCK_DMODEL_R1 > 0 else 0
    BLOCK_DMODEL_R2 : tl.constexpr = BLOCK_DMODEL_R1 - BLOCK_DMODEL1
    BLOCK_DMODEL2 : tl.constexpr = 2 ** (BLOCK_DMODEL_R2.bit_length() - 1) if BLOCK_DMODEL_R2 > 0 else 0
    BLOCK_DMODEL_R3 : tl.constexpr = BLOCK_DMODEL_R2 - BLOCK_DMODEL2

    # tl.static_print('BLOCK_DMODEL0', BLOCK_DMODEL0)
    # tl.static_print('BLOCK_DMODEL1', BLOCK_DMODEL1)
    # tl.static_print('BLOCK_DMODEL2', BLOCK_DMODEL2)

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
    L_not_null = L.cast(dtype=tl.uint64, bitcast=True) != 0  # Allows null L for training=False
    INT8_GEMM: tl.constexpr = INT8 and (not INT8_KV)

    ## philox
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
        if (tl.program_id(0) == 0 and tl.program_id(1) == 0) and tl.program_id(2) == 0:
            if philox_seed_output.cast(dtype=tl.uint64, bitcast=True) != 0:
                tl.store(philox_seed_output, philox_seed)
            if philox_offset_output.cast(dtype=tl.uint64, bitcast=True) != 0:
                tl.store(philox_offset_output,
                         philox_offset_base.to(dtype=philox_seed_output.type.element_ty))

    if PERSISTENT:  # if persistent, kernel loops over multiple tiles
        Num_WG = Num_CU * GRID_CU_MULTIP  # number of workgroups launched
        num_tiles_per_head = tl.cdiv(Max_seqlen_q, BLOCK_M)  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * Num_head_q  # times the number of heads
        num_tiles_total = num_tiles_per_sample * Batch  # times the number of samples
        if PERSISTENT_DYNAMIC:
            tile_id = persistent_atomic_counter.atomic_add(1)  # retuns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)
    else:  # standard, kernel processes only one tile
        tile_id = 0
        num_tiles_total = 1

    continue_condition : tl.int1 = True  # as we can't have return statements inside while loop in Triton

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT
        if PERSISTENT_DYNAMIC:
            # tile id basically tells us the Q block we are handling
            off_z = tile_id // num_tiles_per_sample  # at which batch sample are we
            off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head  # at which head are we inside the sample
            start_m = tile_id % num_tiles_per_sample % num_tiles_per_head  # at which tile are we inside the head
        else:
            start_m = tl.program_id(0)
            off_h_q = tl.program_id(1)
            off_z = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        if Num_seqlens > 0:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0). Some seqlens might be too
            # small for all start_m so for those we return early.
            if start_m * BLOCK_M >= seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            batch_index = 0  # FILEPR
        elif Num_seqlens < 0: # FILEPR
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            if start_m * BLOCK_M >= seqlen_q:
                continue_condition = False
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
            # Varlen, but padded to Rank 4 tensor
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            batch_index = off_z
        else:
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            seqlen_q = Max_seqlen_q
            seqlen_k = Max_seqlen_k
            batch_index = off_z

        if continue_condition:
            # Now we compute whether we need to exit early due to causal masking.
            # This is because for seqlen_q > seqlen_k, M rows of the attn scores
            # are completely masked, resulting in 0s written to the output, and
            # inf written to LSE. We don't need to do any GEMMs in this case.
            # This block of code determines what N is, and if this WG is operating
            # on those M rows.
            o_base = Out + batch_index * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
            if IS_CAUSAL:
                # If seqlen_q == seqlen_k, the attn scores are a square matrix.
                # If seqlen_q != seqlen_k, attn scores are rectangular which means

                if CAUSAL_TYPE == 2:
                    # bottom right aligned causal mask, and ends at either
                    # the top edge (seqlen_q < seqlen_k) or left edge.
                    # This captures the decrease in n_blocks if we have a rectangular attn matrix
                    n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
                else:
                    # top left aligned version
                    n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M, BLOCK_N)

                # This is what adjusts the block_max for the current WG, only
                # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
                n_blocks = min(n_blocks, n_blocks_seqlen)
                # If we have no blocks after adjusting for seqlen deltas, this WG is part of
                # the blocks that are all 0. We exit early.
                if n_blocks <= 0:
                    acc0, acc1, acc2 = composed_zeros_2d(BLOCK_M, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2, dtype=Out.type.element_ty)
                    # We still need to write 0s to the result
                    composed_store(acc0, acc1, acc2,
                                   BLOCK_M,
                                   BLOCK_DMODEL0,
                                   BLOCK_DMODEL1,
                                   BLOCK_DMODEL2,
                                   o_base=o_base,
                                   o_start_row=start_m * BLOCK_M,
                                   o_start_col=0,
                                   o_rows=seqlen_q,
                                   o_cols=BLOCK_DMODEL,
                                   stride_row=stride_om,
                                   stride_col=stride_on)
                    # The tensor allocated for L is based on Max_seqlen_q as that is
                    # statically known.
                    if L_not_null:
                        l_ptrs = L + off_z * Num_head_q * Max_seqlen_q + off_h_q * Max_seqlen_q + offs_m
                        # We store inf to LSE, not -inf because in the bwd pass, we subtract this
                        # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
                        l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
                        l_ptrs_mask = offs_m < Max_seqlen_q
                        tl.store(l_ptrs, l, mask=l_ptrs_mask)
                    # TODO: Should dropout and return encoded softmax be handled here too?
                    continue_condition = False
                    # return

            if continue_condition:
                # MQA / GQA has different K and V head offsets.
                # Note: using If-then-else may trigger a compiler bug...
                off_h_k = off_h_q // (Num_head_q // Num_head_k)

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N

                # Compute pointers for all the tensors used in this kernel.
                q_ptrs0, q_ptrs1, q_ptrs2 = composed_ptrs(Q,
                                                          stride_qz, stride_qh, stride_qm, stride_qk,
                                                          batch_index, off_h_q, cu_seqlens_q_start + offs_m,
                                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                k_ptrs0, k_ptrs1, k_ptrs2 = composed_ptrs(K,
                                                          stride_kz, stride_kh, stride_kn, stride_kk,
                                                          batch_index, off_h_k, cu_seqlens_k_start + offs_n,
                                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                                          TRANSPOSED=True)
                v_ptrs0, v_ptrs1, v_ptrs2 = composed_ptrs(V,
                                                          stride_vz, stride_vh, stride_vk, stride_vn,
                                                          batch_index, off_h_k, cu_seqlens_k_start + offs_n,
                                                          BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                # Compute pointers for all the scale tensors used in this kernel.

                if INT8:
                    k_descale_ptrs = K_descale + off_h_k
                    v_descale_ptrs = V_descale + off_h_k
                    if not INT8_KV:
                        q_descale_ptrs = Q_descale + off_h_q
                    if USE_P_SCALE:
                        p_scale_ptrs = P_scale + off_h_q
                        p_descale_ptrs = P_descale + off_h_q

                if USE_BIAS:
                    # Note: this might get large enough to overflow on some configs
                    bias_offset = B + batch_index * stride_bz + off_h_q * stride_bh  # FILEPR
                    bias_ptrs = bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
                else:
                    bias_ptrs = None

                if USE_ALIBI:
                    a_offset = off_z * stride_az + off_h_q * stride_ah
                    alibi_slope = tl.load(alibi_slopes + a_offset)
                else:
                    alibi_slope = None

                off_zh = off_z * Num_head_q + off_h_q
                if ENABLE_DROPOUT:
                    batch_philox_offset = philox_offset_base + off_zh * Max_seqlen_q * Max_seqlen_k  # FILEPR
                else:
                    batch_philox_offset = 0
                # We can ask to return the dropout mask without actually doing any dropout. In
                # this case, we return an invalid pointer so indicate the mask is not valid.
                if RETURN_ENCODED_SOFTMAX:
                    encoded_sm_base = encoded_softmax + off_zh * Max_seqlen_q * Max_seqlen_k  # FILEPR
                else:
                    encoded_sm_base = None
                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], -3.40282e+38, dtype=tl.float32)  # FILEPR
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc0, acc1, acc2 = composed_zeros_2d(BLOCK_M, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                Qk_scale : constexpr_or_f32 = Sm_scale * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                if (start_m + 1) * BLOCK_M > seqlen_q:
                    q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                               offs_m,
                                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                               seqlen_q, Head_dim,
                                               other=0.0,
                                               PADDED_ROW=True,
                                               PADDED_COL=PADDED_HEAD,
                                               TRANSPOSED=False)
                else:
                    q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                               offs_m,
                                               BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                               seqlen_q, Head_dim,
                                               other=0.0,
                                               PADDED_ROW=False,
                                               PADDED_COL=PADDED_HEAD,
                                               TRANSPOSED=False)

                if INT8:
                    k_descale = tl.load(k_descale_ptrs)
                    v_descale = tl.load(v_descale_ptrs)
                    if not INT8_KV:
                        q_descale = tl.load(q_descale_ptrs)
                    else:
                        q_descale = None
                    if USE_P_SCALE:
                        p_scale = tl.load(p_scale_ptrs)
                        p_descale = tl.load(p_descale_ptrs)
                    else:
                        p_scale = None
                        p_descale = None
                else:
                    q_descale = None
                    k_descale = None
                    v_descale = None
                    p_scale = None
                    p_descale = None
                # Here we compute how many full and masked blocks we have.
                padded_block_k = n_extra_tokens != 0
                is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
                if IS_CAUSAL:
                    # There are always at least BLOCK_M // BLOCK_N masked blocks.
                    # Additionally there might be one more due to dissimilar seqlens.
                    masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
                else:
                    # Padding on Q does not need to be masked in the FA loop.
                    masked_blocks = padded_block_k
                # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
                # In this case we might exceed n_blocks so pick the min.
                masked_blocks = min(masked_blocks, n_blocks)
                n_full_blocks = n_blocks - masked_blocks
                block_min = 0
                block_max = n_blocks * BLOCK_N
                # Compute for full blocks. Here we set causal to false regardless of its actual
                # value because there is no masking. Similarly we do not need padding.
                if n_full_blocks > 0:
                    block_max = (n_blocks - masked_blocks) * BLOCK_N
                    acc0, acc1, acc2, l_i, m_i = _attn_fwd_inner(
                            # Inputs
                            acc0, acc1, acc2,
                            l_i, m_i, Qk_scale,
                            q0, q1, q2,
                            k_ptrs0, k_ptrs1, k_ptrs2,
                            v_ptrs0, v_ptrs1, v_ptrs2,
                            bias_ptrs,
                            stride_kn, stride_vk, stride_bn,
                            # Task positions
                            start_m, block_min, block_max,
                            seqlen_k, seqlen_q, Head_dim,
                            # Dropout
                            dropout_p, philox_seed, batch_philox_offset, Max_seqlen_k,
                            encoded_sm_base,
                            # offs_n_causal, masked_blocks, n_extra_tokens, _
                            0, 0, 0,
                            # Alibi
                            alibi_slope,
                            # INT8
                            q_descale, k_descale, v_descale, p_scale,
                            # constexpr
                            IS_CAUSAL=False,
                            BLOCK_M=BLOCK_M,
                            BLOCK_DMODEL0=BLOCK_DMODEL0,
                            BLOCK_DMODEL1=BLOCK_DMODEL1,
                            BLOCK_DMODEL2=BLOCK_DMODEL2,
                            BLOCK_N=BLOCK_N,
                            OFFS_M=offs_m,
                            OFFS_N=offs_n,
                            PRE_LOAD_V=PRE_LOAD_V,
                            MASK_STEPS=False,
                            ENABLE_DROPOUT=ENABLE_DROPOUT,
                            RETURN_ENCODED_SOFTMAX=RETURN_ENCODED_SOFTMAX,
                            PADDED_HEAD=PADDED_HEAD,
                            INT8_GEMM=INT8_GEMM,
                            INT8_KV=INT8_KV,
                            USE_P_SCALE=USE_P_SCALE,
                            )
                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
                # Remaining blocks, if any, are full / not masked.
                if (masked_blocks > 0):
                    if IS_CAUSAL:
                        offs_n_causal = offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL_BOTTOM_RIGHT else offs_n
                    else:
                        offs_n_causal = 0
                    k_ptrs0, k_ptrs1, k_ptrs2 = composed_advance(k_ptrs0, k_ptrs1, k_ptrs2,
                                                                 n_full_blocks * BLOCK_N * stride_kn,
                                                                 BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
                                                                )
                    v_ptrs0, v_ptrs1, v_ptrs2 = composed_advance(v_ptrs0, v_ptrs1, v_ptrs2,
                                                                 n_full_blocks * BLOCK_N * stride_vk,
                                                                 BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
                                                                )
                    if USE_BIAS:
                        bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
                    acc0, acc1, acc2, l_i, m_i = _attn_fwd_inner(
                            # Inputs
                            acc0, acc1, acc2,
                            l_i, m_i, Qk_scale,
                            q0, q1, q2,
                            k_ptrs0, k_ptrs1, k_ptrs2,
                            v_ptrs0, v_ptrs1, v_ptrs2,
                            bias_ptrs,
                            stride_kn, stride_vk, stride_bn,
                            # Task positions
                            start_m, block_min, block_max,
                            seqlen_k, seqlen_q, Head_dim,
                            # Dropout
                            dropout_p, philox_seed, batch_philox_offset, Max_seqlen_k,
                            encoded_sm_base,
                            # CAUSAL: offs_n_causal, masked_blocks, n_extra_tokens, _
                            offs_n_causal, masked_blocks, n_extra_tokens,
                            # Alibi
                            alibi_slope,
                            # INT8
                            q_descale, k_descale, v_descale, p_scale,
                            # constexpr
                            IS_CAUSAL=IS_CAUSAL,
                            BLOCK_M=BLOCK_M,
                            BLOCK_DMODEL0=BLOCK_DMODEL0,
                            BLOCK_DMODEL1=BLOCK_DMODEL1,
                            BLOCK_DMODEL2=BLOCK_DMODEL2,
                            BLOCK_N=BLOCK_N,
                            OFFS_M=offs_m,
                            OFFS_N=offs_n,
                            PRE_LOAD_V=PRE_LOAD_V,
                            MASK_STEPS=True,
                            ENABLE_DROPOUT=ENABLE_DROPOUT,
                            RETURN_ENCODED_SOFTMAX=RETURN_ENCODED_SOFTMAX,
                            PADDED_HEAD=PADDED_HEAD,
                            INT8_GEMM=INT8_GEMM,
                            INT8_KV=INT8_KV,
                            USE_P_SCALE=USE_P_SCALE,
                            )

                if INT8 and not INT8_KV:
                    if USE_P_SCALE:
                        acc *= p_descale
                    acc *= v_descale

                # epilogue
                # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
                l_recip = 1 / l_i[:, None]
                if ENABLE_DROPOUT:  # Should make dropout faster?
                    l_recip *= 1.0 / (1 - dropout_p)
                # tl.device_print('l_i', l_i)
                # tl.device_print('acc0 no recip', acc0)
                acc0, acc1, acc2 = composed_mul_lhs(acc0, acc1, acc2,
                                                    l_recip,
                                                    BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
                # tl.device_print('acc0 after recip', acc0)
                # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
                # then we have one block with a row of all NaNs which come from computing
                # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
                # and store 0s where there are NaNs as these rows should've been zeroed out.
                end_m_idx = (start_m + 1) * BLOCK_M
                start_m_idx = start_m * BLOCK_M
                causal_start_idx = seqlen_q - seqlen_k if IS_CAUSAL_BOTTOM_RIGHT else 0
                acc0, acc1, acc2 = composed_to(acc0, acc1, acc2, Out.type.element_ty)
                if IS_CAUSAL:
                    if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
                        mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
                        z = 0.0
                        acc0, acc1, acc2 = composed_casual_mask(acc0, acc1, acc2,
                                                                mask_m_offsets, causal_start_idx,
                                                                z.to(acc0.type.element_ty),
                                                                BLOCK_DMODEL0,
                                                                BLOCK_DMODEL1,
                                                                BLOCK_DMODEL2)

                overflow_size = end_m_idx - seqlen_q
                if L_not_null:
                    # write back LSE
                    l_ptrs = L + off_z * Num_head_q * Max_seqlen_q + off_h_q * Max_seqlen_q + offs_m
                    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
                    # This is only true for the last M block. For others, overflow_size will be -ve
                    if overflow_size > 0:
                        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
                        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                    else:
                        tl.store(l_ptrs, m_i + tl.math.log2(l_i))
                # tl.device_print('final acc0', acc0)
                # tl.device_print('acc1', acc1)
                # tl.device_print('acc2', acc2)
                # write back O
                composed_store(acc0, acc1, acc2,
                               BLOCK_M,
                               BLOCK_DMODEL0,
                               BLOCK_DMODEL1,
                               BLOCK_DMODEL2,
                               o_base=o_base,
                               o_start_row=start_m * BLOCK_M,
                               o_start_col=0,
                               o_rows=seqlen_q,
                               o_cols=Head_dim,
                               stride_row=stride_om,
                               stride_col=stride_on)

        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = persistent_atomic_counter.atomic_add(1)
            else:
                tile_id += Num_WG
        else:
            tile_id = num_tiles_total  # break after single tile
