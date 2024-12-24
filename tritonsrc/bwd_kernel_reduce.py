#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import dropout_mask, dropout_rng, dropout_offsets
from masked_load_store import load_fn, mstore2d_reduce
from triton.language.extra import libdevice

# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M : tl.constexpr, QDIM : tl.constexpr, KDIM : tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)


@triton.jit
def bwd_inner_dq_reduce(
    # I/O Tensor
    dq_main, dq_tail, qk_scale, bias_scale,
    DB_block_ptr, store_db,
    # Problem Description
    q_main, q_tail, 
    kt_ptrs_main, kt_ptrs_tail, k_stride,
    vt_ptrs_main, vt_ptrs_tail, v_stride,
    B_block_ptr,
    do_main, do_tail,
    Di, l_i,
    seqlen_q, seqlen_k, head_dim,
    # Sub-problem range
    start_q, lo, hi,
    dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
    # constexpr
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_MAIN: tl.constexpr,
    BLOCK_DMODEL_TAIL: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    offs_q = start_q + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_N)
    offs_d_main = tl.arange(0, BLOCK_DMODEL_MAIN)
    offs_d_tail = tl.arange(0, BLOCK_DMODEL_TAIL)
    
    kt_ptrs_main += lo * k_stride
    kt_ptrs_tail += lo * k_stride
    vt_ptrs_main += lo * v_stride
    vt_ptrs_tail += lo * v_stride

    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (0, lo))
        DB_block_ptr = tl.advance(DB_block_ptr, (0, lo))

    for start_k in range(lo, hi, BLOCK_N):
        # Load K main and tail
        if not FULL_BLOCKS:
            kt_main = load_fn(kt_ptrs_main, offs_d_main, offs_k + start_k, BLOCK_DMODEL_MAIN, seqlen_k)
            kt_tail = load_fn(kt_ptrs_tail, offs_d_tail, offs_k + start_k, head_dim - BLOCK_DMODEL_MAIN, seqlen_k)
            vt_main = load_fn(vt_ptrs_main, offs_d_main, offs_k + start_k, BLOCK_DMODEL_MAIN, seqlen_k)
            vt_tail = load_fn(vt_ptrs_tail, offs_d_tail, offs_k + start_k, head_dim - BLOCK_DMODEL_MAIN, seqlen_k)
        else:
            kt_main = load_fn(kt_ptrs_main, offs_d_main, None, BLOCK_DMODEL_MAIN, seqlen_k)
            kt_tail = load_fn(kt_ptrs_tail, offs_d_tail, None, head_dim - BLOCK_DMODEL_MAIN, seqlen_k)
            vt_main = load_fn(vt_ptrs_main, offs_d_main, None, BLOCK_DMODEL_MAIN, seqlen_k)
            vt_tail = load_fn(vt_ptrs_tail, offs_d_tail, None, head_dim - BLOCK_DMODEL_MAIN, seqlen_k)

        # Initialize qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        # Compute QK for main and tail
        qk += dot(BLOCK_M, BLOCK_DMODEL_MAIN, BLOCK_DMODEL_MAIN, q_main, kt_main)
        qk += dot(BLOCK_M, BLOCK_DMODEL_TAIL, BLOCK_DMODEL_TAIL, q_tail, kt_tail)

        # Handle masking and bias
        offs_k_curr = offs_k[None, :] + start_k
        if not FULL_BLOCKS:
            k_boundary = tl.full((BLOCK_M, ), seqlen_k, dtype=tl.int32)
            mask = offs_k_curr < k_boundary[:, None]
            qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            qk = tl.where(offs_q[:, None] >= offs_k_curr, qk, float("-inf"))
        
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')

        # Compute attention weights
        p = tl.math.exp2(qk_scale * qk - l_i[:, None])
        if not FULL_BLOCKS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)

        # Compute dP
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += dot(BLOCK_M, BLOCK_DMODEL_MAIN, BLOCK_DMODEL_MAIN, do_main, vt_main)
        dp += dot(BLOCK_M, BLOCK_DMODEL_TAIL, BLOCK_DMODEL_TAIL, do_tail, vt_tail)

        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)

        # Compute dS
        ds = p * (dp - Di[:, None])
                    
        # Compute dQ for main and tail separately
        if BLOCK_M == 1:
            dq_main += tl.view(kt_main, [BLOCK_DMODEL_MAIN]) * ds.to(q_main.dtype)
            dq_tail += tl.view(kt_tail, [BLOCK_DMODEL_TAIL]) * ds.to(q_tail.dtype)
        else:
            dq_main = tl.dot(ds.to(q_main.dtype), tl.trans(kt_main), acc=dq_main)
            dq_tail = tl.dot(ds.to(q_tail.dtype), tl.trans(kt_tail), acc=dq_tail)
        
        if BIAS_TYPE == 1:
            if store_db:
                tl.store(DB_block_ptr, ds.to(DB_block_ptr.type.element_ty), boundary_check=(0,1))
        
        # Update pointers
        kt_ptrs_main += BLOCK_N * k_stride
        kt_ptrs_tail += BLOCK_N * k_stride
        vt_ptrs_main += BLOCK_N * v_stride
        vt_ptrs_tail += BLOCK_N * v_stride
        
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))

    return dq_main, dq_tail


@triton.jit
def bwd_kernel_dq_reduce(
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
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_MAIN: tl.constexpr,
    BLOCK_DMODEL_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # Initialize random seed for dropout
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)

    # Program ID
    start_q = tl.program_id(0) * BLOCK_M
    off_h_q = tl.program_id(1)
    off_h_k = off_h_q
    off_z = tl.program_id(2)
    
    # Calculate batch/head indices
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    off_zh = off_z * num_h + off_h_q

    # Initialize offsets
    offs_q = start_q + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_main = tl.arange(0, BLOCK_DMODEL_MAIN)
    offs_d_tail = tl.arange(0, BLOCK_DMODEL_TAIL)

    # Handle sequence lengths
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

    # Still need early exit in GPU kernel to support varlen
    if CAUSAL:
        # TODO: bottom right causal and windowed
        if start_q > seqlen_k:
            return

    # Initialize pointers to Q, K, V
    q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q += q_offset

    # Load Q main and tail
    q_ptrs_main = Q + offs_q[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL_MAIN)[None, :] * stride_qk
    q_ptrs_tail = Q + offs_q[:, None] * stride_qm + (BLOCK_DMODEL_MAIN + tl.arange(0, BLOCK_DMODEL_TAIL))[None, :] * stride_qk
    
    if start_q + BLOCK_M <= seqlen_q:
        q_main = load_fn(q_ptrs_main, None, offs_d_main, seqlen_q, BLOCK_DMODEL_MAIN)
        q_tail = load_fn(q_ptrs_tail, None, offs_d_tail, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)
    else:
        q_main = load_fn(q_ptrs_main, offs_q, offs_d_main, seqlen_q, BLOCK_DMODEL_MAIN)
        q_tail = load_fn(q_ptrs_tail, offs_q, offs_d_tail, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)

    # Initialize pointers to DO
    do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    DO += do_offset
    
    # Load DO main and tail
    do_ptrs_main = DO + offs_q[:, None] * stride_om + tl.arange(0, BLOCK_DMODEL_MAIN)[None, :] * stride_ok
    do_ptrs_tail = DO + offs_q[:, None] * stride_om + (BLOCK_DMODEL_MAIN + tl.arange(0, BLOCK_DMODEL_TAIL))[None, :] * stride_ok
    
    if start_q + BLOCK_M <= seqlen_q:
        do_main = load_fn(do_ptrs_main, None, offs_d_main, seqlen_q, BLOCK_DMODEL_MAIN)
        do_tail = load_fn(do_ptrs_tail, None, offs_d_tail, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)
    else:
        do_main = load_fn(do_ptrs_main, offs_q, offs_d_main, seqlen_q, BLOCK_DMODEL_MAIN)
        do_tail = load_fn(do_ptrs_tail, offs_q, offs_d_tail, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)

    # Initialize pointers to K and V
    k_offset = off_h_k * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    v_offset = off_h_k * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    K += k_offset
    V += v_offset

    # Setup K and V pointers for main and tail
    kt_ptrs_main = K + tl.arange(0, BLOCK_DMODEL_MAIN)[:, None] * stride_kk + offs_n[None, :] * stride_kn
    kt_ptrs_tail = K + (BLOCK_DMODEL_MAIN + tl.arange(0, BLOCK_DMODEL_TAIL))[:, None] * stride_kk + offs_n[None, :] * stride_kn
    vt_ptrs_main = V + tl.arange(0, BLOCK_DMODEL_MAIN)[:, None] * stride_vn + offs_n[None, :] * stride_vk
    vt_ptrs_tail = V + (BLOCK_DMODEL_MAIN + tl.arange(0, BLOCK_DMODEL_TAIL))[:, None] * stride_vn + offs_n[None, :] * stride_vk

    # Initialize pointers to LSE and D
    D_ptrs = D + off_zh * max_seqlen_q
    l_ptrs = L + off_zh * max_seqlen_q
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
    else:
        batch_philox_offset = 0

    # Check for OOB accesses on D and LSE
    q_boundary = tl.full((BLOCK_M, ), seqlen_q, dtype=tl.int32)
    d_lse_ptrs_mask = offs_q < q_boundary
    Di = tl.load(D_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)
    l_i = tl.load(l_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)

    # Initialize output tensors
    dq_main = tl.zeros([BLOCK_M, BLOCK_DMODEL_MAIN], dtype=tl.float32)
    dq_tail = tl.zeros([BLOCK_M, BLOCK_DMODEL_TAIL], dtype=tl.float32)

    # Setup bias pointers if needed
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

    # Calculate scaling factors
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale

    # Calculate block ranges
    k_lo = 0
    k_hi = min(start_q + BLOCK_M, seqlen_k) if CAUSAL else seqlen_k
    real_seqlen_k = k_hi - k_lo
    
    n_blocks = tl.cdiv(k_hi - k_lo, BLOCK_N)
    n_extra_tokens = 0
    if real_seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - real_seqlen_k
    elif real_seqlen_k % BLOCK_N:
        n_extra_tokens = real_seqlen_k % BLOCK_N
        
    is_irregular_k = n_extra_tokens != 0
    leading_masked_blocks = 0
    trailing_masked_blocks = 0
    if CAUSAL:
        mask_top_edge = start_q
        trailing_masked_blocks = tl.cdiv(k_hi - mask_top_edge // BLOCK_N * BLOCK_N, BLOCK_N)
    else:
        trailing_masked_blocks = 1 if is_irregular_k else 0
                
    # Check for OOB accesses on D and LSE
    q_boundary = tl.full((BLOCK_M, ), seqlen_q, dtype=tl.int32)
    d_lse_ptrs_mask = offs_q < q_boundary
    Di = tl.load(D_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)
    l_i = tl.load(l_ptrs + offs_q, mask=d_lse_ptrs_mask, other=0.0)

    n_full_blocks = n_blocks - leading_masked_blocks - trailing_masked_blocks
    
    # Process blocks
    if n_full_blocks > 0:
        lo = 0
        hi = n_full_blocks * BLOCK_N
        
        dq_main, dq_tail = bwd_inner_dq_reduce(
            dq_main, dq_tail, qk_scale, bias_scale,
            DB_block_ptr, store_db,
            q_main, q_tail,
            kt_ptrs_main, kt_ptrs_tail, stride_kn,
            vt_ptrs_main, vt_ptrs_tail, stride_vk,
            B_block_ptr,
            do_main, do_tail,
            Di, l_i,
            seqlen_q, seqlen_k, head_dim,
            start_q, lo, hi,
            dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_MAIN,
            BLOCK_DMODEL_TAIL,
            True,  # FULL_BLOCKS
            False,  # CAUSAL has zero effect for full blocks
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE
        )

    if trailing_masked_blocks > 0:
        lo = n_full_blocks * BLOCK_N
        hi = k_hi
        tl.debug_barrier()
        dq_main, dq_tail = bwd_inner_dq_reduce(
            dq_main, dq_tail, qk_scale, bias_scale,
            DB_block_ptr, store_db,
            q_main, q_tail,
            kt_ptrs_main, kt_ptrs_tail, stride_kn,
            vt_ptrs_main, vt_ptrs_tail, stride_vk,
            B_block_ptr,
            do_main, do_tail,
            Di, l_i,
            seqlen_q, seqlen_k, head_dim,
            start_q, lo, hi,
            dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_MAIN,
            BLOCK_DMODEL_TAIL,
            False,  # FULL_BLOCKS
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE
        )

    # Scale and store results
    dq_main = (dq_main * sm_scale).to(dq_main.type.element_ty)
    dq_tail = (dq_tail * sm_scale).to(dq_tail.type.element_ty)

    # Store results
    dq_offset = batch_index * stride_dqz + off_h_q * stride_dqh + cu_seqlens_q_start * stride_dqm
    mstore2d_reduce(
        dq_main, dq_tail,
        BLOCK_M,
        DQ + dq_offset,
        start_q, 0,
        seqlen_q, head_dim,
        stride_dqm, stride_dqk,
        BLOCK_DMODEL_MAIN,
        BLOCK_DMODEL_TAIL
    )

@triton.jit
def bwd_inner_dk_dv_reduce(
    dk_main, dk_tail, dv_main, dv_tail, qk_scale, bias_scale,
    q_ptrs_main, q_ptrs_tail, q_stride,
    kt_main, kt_tail, vt_main, vt_tail,
    B_block_ptr,
    do_ptrs_main, do_ptrs_tail, do_stride,
    l_ptrs, D_ptrs,
    seqlen_q, seqlen_k, head_dim,
    start_k, lo, hi, overflow_size,
    dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_MAIN: tl.constexpr,
    BLOCK_DMODEL_TAIL: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    offs_k = start_k + tl.arange(0, BLOCK_N)
    offs_q = tl.arange(0, BLOCK_M)
    
    q_ptrs_main += lo * q_stride
    q_ptrs_tail += lo * q_stride
    do_ptrs_main += lo * do_stride
    do_ptrs_tail += lo * do_stride

    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (lo, 0))
    
    #TODO:MASK
    for start_q in range(lo, hi, BLOCK_M):
        offs_q_curr = offs_q[:, None] + start_q
        if not FULL_BLOCKS:
            q_main = load_fn(q_ptrs_main, offs_q + start_q, None, seqlen_q, head_dim)
            q_tail = load_fn(q_ptrs_tail, offs_q + start_q, None, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)
            do_main = load_fn(do_ptrs_main, offs_q + start_q, None, seqlen_q, head_dim)
            do_tail = load_fn(do_ptrs_tail, offs_q + start_q, None, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)
        else:
            q_main = load_fn(q_ptrs_main, None, None, seqlen_q, head_dim)
            q_tail = load_fn(q_ptrs_tail, None, None, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)
            do_main = load_fn(do_ptrs_main, None, None, seqlen_q, head_dim)
            do_tail = load_fn(do_ptrs_tail, None, None, seqlen_q, head_dim - BLOCK_DMODEL_MAIN)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if not FULL_BLOCKS:
            if overflow_size > 0:
                boundary_n = tl.full((BLOCK_N, ), seqlen_q, dtype=tl.int32)
                mask = offs_q_curr < boundary_n[None, :]
                qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            qk = tl.where(offs_q_curr >= offs_k[None, :], qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            qk += bias * bias_scale
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        #TODO: BLOCK_M = 1
        qk += tl.dot(q_main, kt_main)
        qk += tl.dot(q_tail, kt_tail)
        if FULL_BLOCKS:
            Di = tl.load(D_ptrs + offs_q_curr)
            l_i = tl.load(l_ptrs + offs_q_curr)
        else:
            boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
            d_lse_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
            d_lse_padding = tl.full((BLOCK_M, ), 0, dtype=tl.float32)
            Di = tl.load(D_ptrs + offs_q_curr, 
                        mask=d_lse_ptrs_mask[:, None],
                        other=d_lse_padding[:, None])
            l_i = tl.load(l_ptrs + offs_q_curr,
                         mask=d_lse_ptrs_mask[:,None], 
                         other=d_lse_padding[:, None])
        p = tl.math.exp2(qk_scale * qk - l_i)
        if not FULL_BLOCKS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)
                        
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            p_dropped = tl.where(keep, p / (1.0 - dropout_p), 0.0)
        else:
            p_dropped = p
        if BLOCK_M == 1:
            dv_main += p_dropped.to(do_main.dtype) * do_main
            dv_tail += p_dropped.to(do_tail.dtype) * do_tail
        else:
            dv_main += tl.dot(tl.trans(p_dropped.to(do_main.dtype)), do_main)
            dv_tail += tl.dot(tl.trans(p_dropped.to(do_tail.dtype)), do_tail)
        dp_main = tl.dot(do_main, vt_main)
        dp_tail = tl.dot(do_tail, vt_tail)
        dp = dp_main + dp_tail

        if ENABLE_DROPOUT:
            dp = tl.where(keep, dp / (1.0 - dropout_p), 0)
        ds = p * (dp - Di)
        if BLOCK_M == 1:
            dk_main += ds.to(q_main.dtype) * q_main
            dk_tail += ds.to(q_tail.dtype) * q_tail
        else:
            dk_main += tl.dot(tl.trans(ds.to(q_main.dtype)), q_main)
            dk_tail += tl.dot(tl.trans(ds.to(q_tail.dtype)), q_tail)
        q_ptrs_main += q_stride * BLOCK_M
        q_ptrs_tail += q_stride * BLOCK_M
        do_ptrs_main += do_stride * BLOCK_M
        do_ptrs_tail += do_stride * BLOCK_M
        
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_M, 0))

    return dk_main, dk_tail, dv_main, dv_tail

@triton.jit
def bwd_kernel_dk_dv_reduce(
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
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens : 'i32',
    max_seqlen_q : 'i32',
    max_seqlen_k : 'i32',
    head_dim : 'i32',
    dropout_p,
    philox_seed_ptr,
    philox_offset1 : '*u32',
    philox_offset2 : 'u32',
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_MAIN: tl.constexpr,
    BLOCK_DMODEL_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # Program ID
    start_k = tl.program_id(0) * BLOCK_N
    off_h_k = tl.program_id(1)
    off_z = tl.program_id(2)
    philox_seed = 0
    philox_offset_base = philox_offset2
    if ENABLE_DROPOUT:
        philox_seed = tl.load(philox_seed_ptr)
        philox_offset_base += tl.load(philox_offset1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_k + tl.arange(0, BLOCK_N)
    offs_d_main = tl.arange(0, BLOCK_DMODEL_MAIN)
    offs_d_tail = tl.arange(0, BLOCK_DMODEL_TAIL)
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
    elif num_seqlens < 0:
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        if start_k >= seqlen_k:
            return
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z

    if CAUSAL and start_k > seqlen_q:
        return
    k_offset = off_h_k * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K += k_offset
    
    kt_ptrs_main = K + offs_d_main[:, None] * stride_kk + offs_n[None, :] * stride_kn
    kt_ptrs_tail = K + (BLOCK_DMODEL_MAIN + offs_d_tail[:, None]) * stride_kk + offs_n[None, :] * stride_kn
    if start_k + BLOCK_N <= seqlen_k:
        kt_main = load_fn(kt_ptrs_main, offs_d_main, None, head_dim, seqlen_k)
        kt_tail = load_fn(kt_ptrs_tail, offs_d_tail, None, head_dim-BLOCK_DMODEL_MAIN, seqlen_k)
    else:
        kt_main = load_fn(kt_ptrs_main, offs_d_main, offs_n, head_dim, seqlen_k)
        kt_tail = load_fn(kt_ptrs_tail, offs_d_tail, offs_n, head_dim-BLOCK_DMODEL_MAIN, seqlen_k)

    v_offset = off_h_k * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V += v_offset
    
    vt_ptrs_main = V + offs_d_main[:, None] * stride_vn + offs_n[None, :] * stride_vk
    vt_ptrs_tail = V + (BLOCK_DMODEL_MAIN + offs_d_tail[:, None]) * stride_vn + offs_n[None, :] * stride_vk
    
    if start_k + BLOCK_N <= seqlen_k:
        vt_main = load_fn(vt_ptrs_main, offs_d_main, None, head_dim, seqlen_k)
        vt_tail = load_fn(vt_ptrs_tail, offs_d_tail, None, head_dim-BLOCK_DMODEL_MAIN, seqlen_k)
    else:
        vt_main = load_fn(vt_ptrs_main, offs_d_main, offs_n, head_dim, seqlen_k)
        vt_tail = load_fn(vt_ptrs_tail, offs_d_tail, offs_n, head_dim-BLOCK_DMODEL_MAIN, seqlen_k)
    dv_main = tl.zeros([BLOCK_N, BLOCK_DMODEL_MAIN], dtype=tl.float32)
    dv_tail = tl.zeros([BLOCK_N, BLOCK_DMODEL_TAIL], dtype=tl.float32)
    dk_main = tl.zeros([BLOCK_N, BLOCK_DMODEL_MAIN], dtype=tl.float32)
    dk_tail = tl.zeros([BLOCK_N, BLOCK_DMODEL_TAIL], dtype=tl.float32)
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
            base=B + off_h_k * stride_bh + batch_index * stride_bz,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(0, start_k),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
    qk_scale = sm_scale * 1.44269504089
    bias_scale = 1.0 / sm_scale
    q_lo = start_k if CAUSAL else 0
    q_hi = seqlen_q
    real_seqlen_q = q_hi - q_lo
    n_blocks = tl.cdiv(q_hi - q_lo, BLOCK_M)
    n_extra_tokens = 0
    if real_seqlen_q < BLOCK_M:
        n_extra_tokens = BLOCK_M - real_seqlen_q
    elif real_seqlen_q % BLOCK_M:
        n_extra_tokens = real_seqlen_q % BLOCK_M
    
    is_irregular_q = n_extra_tokens != 0
    leading_masked_blocks = tl.cdiv(BLOCK_N, BLOCK_M) if CAUSAL else 0
    trailing_masked_blocks = 1 if is_irregular_q else 0
    n_full_blocks = n_blocks - leading_masked_blocks - trailing_masked_blocks
    group_size : tl.constexpr = 1
    for off_h_q in range(off_h_k * group_size, off_h_k * group_size + group_size):
        off_zh = off_z * tl.num_programs(1) + off_h_q
        
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
        else:
            batch_philox_offset = 0
        D_ptrs = D + off_zh * max_seqlen_q
        l_ptrs = L + off_zh * max_seqlen_q

        q_offset = off_h_q * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
        do_offset = off_h_q * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
        
        q_ptrs_main = Q + q_offset + offs_m[:, None] * stride_qm + offs_d_main[None, :] * stride_qk
        q_ptrs_tail = Q + q_offset + offs_m[:, None] * stride_qm + (BLOCK_DMODEL_MAIN + offs_d_tail[None, :]) * stride_qk
        
        do_ptrs_main = DO + do_offset + offs_m[:, None] * stride_om + offs_d_main[None, :] * stride_ok
        do_ptrs_tail = DO + do_offset + offs_m[:, None] * stride_om + (BLOCK_DMODEL_MAIN + offs_d_tail[None, :]) * stride_ok
        if leading_masked_blocks > 0:
            lo = q_lo
            hi = lo + leading_masked_blocks * BLOCK_M
            overflow_size = 0 if hi < q_hi else hi - q_hi
            dk_main, dk_tail, dv_main, dv_tail = bwd_inner_dk_dv_reduce(
                dk_main, dk_tail, dv_main, dv_tail, qk_scale, bias_scale,
                q_ptrs_main, q_ptrs_tail, stride_qm, kt_main, kt_tail, vt_main, vt_tail, B_block_ptr,
                do_ptrs_main, do_ptrs_tail, stride_om,
                l_ptrs, D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL_MAIN, BLOCK_DMODEL_TAIL,
                False, CAUSAL, ENABLE_DROPOUT, PADDED_HEAD, BIAS_TYPE
            )
        if n_full_blocks > 0:
            lo = q_lo + leading_masked_blocks * BLOCK_M
            hi = lo + n_full_blocks * BLOCK_M
            dk_main, dk_tail, dv_main, dv_tail = bwd_inner_dk_dv_reduce(
                dk_main, dk_tail, dv_main, dv_tail, qk_scale, bias_scale,
                q_ptrs_main, q_ptrs_tail, stride_qm, kt_main, kt_tail, vt_main, vt_tail, B_block_ptr,
                do_ptrs_main, do_ptrs_tail, stride_om,
                l_ptrs, D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, 0,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL_MAIN, BLOCK_DMODEL_TAIL,
                True, False, ENABLE_DROPOUT, PADDED_HEAD, BIAS_TYPE
            )

        if n_full_blocks >= 0 and trailing_masked_blocks > 0:
            lo = q_lo + leading_masked_blocks * BLOCK_M + n_full_blocks * BLOCK_M
            hi = q_hi
            overflow_size = lo + trailing_masked_blocks * BLOCK_M - q_hi
            dk_main, dk_tail, dv_main, dv_tail = bwd_inner_dk_dv_reduce(
                dk_main, dk_tail, dv_main, dv_tail, qk_scale, bias_scale,
                q_ptrs_main, q_ptrs_tail, stride_qm, kt_main, kt_tail, vt_main, vt_tail, B_block_ptr,
                do_ptrs_main, do_ptrs_tail, stride_om,
                l_ptrs, D_ptrs,
                seqlen_q, seqlen_k, head_dim,
                start_k, lo, hi, overflow_size,
                dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL_MAIN, BLOCK_DMODEL_TAIL,
                False, CAUSAL, ENABLE_DROPOUT, PADDED_HEAD, BIAS_TYPE
            )
    dk_main = (dk_main * sm_scale).to(kt_main.dtype)
    dk_tail = (dk_tail * sm_scale).to(kt_tail.dtype)
    dv_main = dv_main.to(vt_main.dtype)
    dv_tail = dv_tail.to(vt_tail.dtype)
    dk_offset = off_h_k * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
    dv_offset = off_h_k * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk

    mstore2d_reduce(
        dk_main, dk_tail,
        BLOCK_N,
        DK + dk_offset,
        start_k, 0,
        seqlen_k, head_dim,
        stride_dkn, stride_dkk,
        BLOCK_DMODEL_MAIN,
        BLOCK_DMODEL_TAIL
    )

    mstore2d_reduce(
        dv_main, dv_tail,
        BLOCK_N,
        DV + dv_offset,
        start_k, 0,
        seqlen_k, head_dim,
        stride_dvk, stride_dvn,
        BLOCK_DMODEL_MAIN,
        BLOCK_DMODEL_TAIL
    )