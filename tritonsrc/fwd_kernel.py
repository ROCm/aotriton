#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
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
from fwd_kernel_inner import attn_fwd_inner
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
)

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def attn_fwd(
        Q, K, V, B, sm_scale, L, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_bz, stride_bh, stride_bm, stride_bn,
        stride_oz, stride_oh, stride_om, stride_on,
        num_head_q : 'i32',
        num_head_k : 'i32',
        cu_seqlens_q,
        cu_seqlens_k,
        num_seqlens : 'i32',
        max_seqlen_q : 'i32',
        max_seqlen_k : 'i32',
        head_dim : 'i32',
        dropout_p,
        philox_seed_ptr,
        philox_offset1 : '*u32',
        philox_offset2 : 'i32',
        philox_seed_output : '*u64',
        philox_offset_output : '*u64',
        encoded_softmax,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        pre_load_v: tl.constexpr,
        ENABLE_DROPOUT: tl.constexpr,
        RETURN_ENCODED_SOFTMAX: tl.constexpr,
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
    # tl.static_print(f'{BLOCK_DMODEL=}')
    # tl.static_print(f'{BLOCK_DMODEL0=}')
    # tl.static_print(f'{BLOCK_DMODEL1=}')
    # tl.static_print(f'{BLOCK_DMODEL2=}')
    # tl.static_print(f'{BLOCK_DMODEL_R0=}')
    # tl.static_print(f'{BLOCK_DMODEL_R1=}')
    # tl.static_print(f'{BLOCK_DMODEL_R2=}')
    # tl.static_print(f'{BLOCK_DMODEL_R3=}')
    tl.static_assert(BLOCK_DMODEL_R3 == 0, f'BLOCK_DMODEL = {BLOCK_DMODEL} = 0b{BLOCK_DMODEL:b} cannot be factored into <= 3 power of two values')
    # The supported D0D1D2 pattern is xxx/xx0/x00, x means non-zero power of two, 0 means 0
    # Truth Table of supported BLOCK_DMODELs value
    #
    # |        | D2 > 0 | D2 = 0 |
    # |--------|--------|--------|
    # | D1 = 0 |   F    |    T   |
    # | D1 > 0 |   T    |    T   |
    tl.static_assert(BLOCK_DMODEL1 > 0 or BLOCK_DMODEL2 == 0, 'Only trailing BLOCK_DMODELx can be 0')
    # lower case pre_load_v for backward compatibility, minimize changes to
    # other files. Will be fixed in a separate PR
    PRE_LOAD_V : tl.constexpr = pre_load_v
    # No ALIBI interface for now
    USE_ALIBI : tl.constexpr = False
    # alibi_slopes = None
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # offs_d = tl.arange(0, BLOCK_DMODEL)
    # offs_d = composed_offs_1d(BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)
    # ^ NOT WORKING 
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
    if num_seqlens > 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0
    elif num_seqlens == 0:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        batch_index = off_z
    else: # < 0 for padded seqlen
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M >= seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        # Varlen, but padded to Rank 4 tensor
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if CAUSAL:  # Head Dimension Triptych TODO
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # This captures the decrease in n_blocks if we have a rectangular attn matrix

        # bottom right aligned version. Ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)

        # top left aligned version. Ends at either
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M, BLOCK_N)

        # This is what adjusts the block_max for the current WG, only
        # if CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = Out + batch_index * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
            acc0, acc1, acc2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc0, acc1, acc2, mask=o_ptrs_mask)
            # The tensor allocated for L is based on max_seqlen_q as that is
            # statically known.
            L_ptr_base = L + (off_z * num_head_q + off_h_q)  * max_seqlen_q
            l_ptrs = L_ptr_base + offs_m
            # We store inf to LSE, not -inf because in the bwd pass, we subtract this
            # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
            l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
            l_ptrs_mask = offs_m < max_seqlen_q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    # group_size = num_head_q // num_head_k
    # if group_size != 1:
    #     off_h_k = off_h_q // group_size
    # else:
    #     off_h_k = off_h_q
    off_h_k = off_h_q // (num_head_q // num_head_k)

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # Compute pointers for all the tensors used in this kernel.
    # q_offset = Q + batch_index * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    # q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # k_offset = K + batch_index * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    # k_ptrs0, k_ptrs1, k_ptrs2 = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # v_offset = V + batch_index * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    # v_ptrs0, v_ptrs1, v_ptrs2 = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
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
    if BIAS_TYPE == 0:
        bias_ptrs = None
    elif BIAS_TYPE == 1:  # Head Dimension Triptych TODO
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = B + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

    if USE_ALIBI:
        a_offset = batch_index * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    off_zh = off_z * num_head_q + off_h_q
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
    else:
        batch_philox_offset = 0
    # We can ask to return the dropout mask without actually doing any dropout. In
    # this case, we return an invalid pointer so indicate the mask is not valid.
    if RETURN_ENCODED_SOFTMAX:
        encoded_sm_base = encoded_softmax + off_zh * max_seqlen_q * max_seqlen_k
        # encoded_sm_ptrs = encoded_sm_base + offs_m[:, None] * max_seqlen_k + offs_n[None, :]
    else:
        encoded_sm_base = None
        # encoded_sm_ptrs = None
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], -3.40282e+38, dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc0, acc1, acc2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    acc0, acc1, acc2 = composed_zeros_2d(BLOCK_M, BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

    # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale  # TODO: legacy code to remove
    # # Q is loaded once at the beginning and shared by all N blocks.
    # q_ptrs_mask = offs_m[:, None] < seqlen_q
    # if PADDED_HEAD:
    #     q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < head_dim)

    # Head Dimension Triptych WIP
    # q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    if (start_m + 1) * BLOCK_M > seqlen_q:
        q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                   offs_m,
                                   BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                   seqlen_q, head_dim,
                                   other=0.0,
                                   PADDED_ROW=True,
                                   PADDED_COL=PADDED_HEAD,
                                   TRANSPOSED=False)
    else:
        q0, q1, q2 = composed_load(q_ptrs0, q_ptrs1, q_ptrs2,
                                   offs_m,
                                   BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                   seqlen_q, head_dim,
                                   other=0.0,
                                   PADDED_ROW=False,
                                   PADDED_COL=PADDED_HEAD,
                                   TRANSPOSED=False)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc0, acc1, acc2, l_i, m_i = attn_fwd_inner(
                acc0, acc1, acc2,
                l_i, m_i, qk_scale, bias_scale,
                q0, q1, q2,
                k_ptrs0, k_ptrs1, k_ptrs2,
                v_ptrs0, v_ptrs1, v_ptrs2,
                bias_ptrs,
                stride_kn, stride_vk, stride_bn,
                seqlen_q, seqlen_k, head_dim,
                start_m, block_min, block_max,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                encoded_sm_base,
                # offs_n_causal, masked_blocks, n_extra_tokens
                0, 0, 0,
                alibi_slope,
                # CAUSAL, ....
                False,
                BLOCK_M,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_N,
                # _, MASK_STEPS, ...
                PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if masked_blocks > 0:
        if CAUSAL:
            # Bottom right variant
            # offs_n_causal = offs_n + (seqlen_q - seqlen_k)
            # Top left variant
            offs_n_causal = offs_n
        else:
            offs_n_causal = 0
        # k_ptrs0, k_ptrs1, k_ptrs2 += n_full_blocks * BLOCK_N * stride_kn
        # v_ptrs0, v_ptrs1, v_ptrs2 += n_full_blocks * BLOCK_N * stride_vk
        k_ptrs0, k_ptrs1, k_ptrs2 = composed_advance(k_ptrs0, k_ptrs1, k_ptrs2,
                                                     n_full_blocks * BLOCK_N * stride_kn,
                                                     BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
                                                    )
        v_ptrs0, v_ptrs1, v_ptrs2 = composed_advance(v_ptrs0, v_ptrs1, v_ptrs2,
                                                     n_full_blocks * BLOCK_N * stride_vk,
                                                     BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
                                                    )
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        # if RETURN_ENCODED_SOFTMAX:
        #     encoded_sm_base += n_full_blocks * BLOCK_N
            # encoded_sm_ptrs += n_full_blocks * BLOCK_N
        acc0, acc1, acc2, l_i, m_i = attn_fwd_inner(
                acc0, acc1, acc2, l_i, m_i, qk_scale, bias_scale,
                q0, q1, q2,
                k_ptrs0, k_ptrs1, k_ptrs2,
                v_ptrs0, v_ptrs1, v_ptrs2,
                bias_ptrs,
                stride_kn, stride_vk, stride_bn,
                seqlen_q, seqlen_k, head_dim,
                start_m, block_min, block_max,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                encoded_sm_base,
                offs_n_causal, masked_blocks, n_extra_tokens,
                alibi_slope,
                # CAUSAL, ....
                CAUSAL,
                BLOCK_M,
                BLOCK_DMODEL0,
                BLOCK_DMODEL1,
                BLOCK_DMODEL2,
                BLOCK_N,
                # _, MASK_STEPS, ...
                PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD)
    # epilogue
    l_recip = 1.0 / l_i[:, None]
    # acc0, acc1, acc2 = acc0, acc1, acc2 * l_recip
    acc0, acc1, acc2 = composed_mul_lhs(acc0, acc1, acc2,
                                        l_recip,
                                        BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2
                                       )
    if ENABLE_DROPOUT:  # Head Dimension Triptych TODO
        dropout_scale = 1.0 / (1 - dropout_p)
        acc0, acc1, acc2 = acc0, acc1, acc2 * dropout_scale
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    # Bottom right alignment
    # causal_start_idx = seqlen_q - seqlen_k
    # Top left alignment
    causal_start_idx = 0

    # acc0, acc1, acc2 = acc0, acc1, acc2.to(Out.type.element_ty)
    acc0, acc1, acc2 = composed_to(acc0, acc1, acc2, Out.type.element_ty)

    if CAUSAL:  # Head Dimension Triptych TODO
        tl.assume(start_m_idx >= 0)
        tl.assume(end_m_idx >= 0)
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc0, acc1, acc2 = tl.where(out_ptrs_mask, acc0, acc1, acc2, z.to(acc0, acc1, acc2.type.element_ty))
    # FIXME: MQA/GQA L tensor
    # TODO: make writing of L optional
    # write back LSE

    # L's shape: (batch, head, seqlen_q)
    L_ptr_base = L + (off_z * num_head_q + off_h_q) * max_seqlen_q
    l_ptrs = L_ptr_base + offs_m
    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    o_base = Out + batch_index * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    # mstore2d(acc0, acc1, acc2,
    #          BLOCK_M,
    #          BLOCK_DMODEL,
    #          o_base=o_base,
    #          o_start_row=start_m * BLOCK_M,
    #          o_start_col=0,
    #          o_rows=seqlen_q,
    #          o_cols=head_dim,
    #          stride_row=stride_om,
    #          stride_col=stride_on)
    composed_store(acc0, acc1, acc2,
                   BLOCK_M,
                   BLOCK_DMODEL0,
                   BLOCK_DMODEL1,
                   BLOCK_DMODEL2,
                   o_base=o_base,
                   o_start_row=start_m * BLOCK_M,
                   o_start_col=0,
                   o_rows=seqlen_q,
                   o_cols=head_dim,
                   stride_row=stride_om,
                   stride_col=stride_on)
