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

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]

@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep

@triton.jit
def attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m,
    seqlen_q,
    seqlen_k_low,
    seqlen_k_high,
    k_padded,
    dropout_p,
    dropout_seqlen_k,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    MARGINAL_BLOCK: tl.constexpr,  # MARGINAL_BLOCK = CAUSAL or k_padded
    PADDED_HEAD: tl.constexpr,
):
    lo, hi = seqlen_k_low, seqlen_k_high
    if MARGINAL_BLOCK:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        # -- compute qk ----
        # MARGINAL_BLOCK serves as a compile-time switch for first attn_fwd_inner calls to "solid" blocks
        if MARGINAL_BLOCK and k_padded:
            if PADDED_HEAD:
                k = tl.load(K_block_ptr, boundary_check=(1,0), padding_option="zero")
            else:
                k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            if PADDED_HEAD:
                k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            else:
                k = tl.load(K_block_ptr)
        if pre_load_v:
            if MARGINAL_BLOCK and k_padded:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
            else:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MARGINAL_BLOCK:
            if CAUSAL:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(mask, qk, float("-inf"))
            if k_padded:
                boundary_m = tl.full([BLOCK_M], seqlen_k_high, dtype=tl.int32)
                size_n = start_n + offs_n[None,:]
                mask = size_n < boundary_m[:,None]
                qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # Note about the conflicts of Flash attention algorithm and PyTorch's CUDA implementation
        # PyTorch needs to return softmax(qk) (dropout mask encoded in sign bits)
        # While Flash attention paper computer the dropout AFTER exp2(qk- m_ij)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * dropout_seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, dropout_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty), boundary_check=(0,1))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr,
                     p.to(encoded_softmax_block_ptr.type.element_ty),
                     boundary_check=(0,1))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not pre_load_v:
            if MARGINAL_BLOCK and k_padded:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
            else:
                if PADDED_HEAD:
                    v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
                else:
                    v = tl.load(V_block_ptr)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    seqlen_q,
    seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    if start_m * BLOCK_M + BLOCK_M > seqlen_q:
        q_padded = True
    else:
        q_padded = False
    k_padded = True
    if seqlen_k < BLOCK_N:
        seqlen_k_faligned = 0 # floor aligned
    elif seqlen_k % BLOCK_N:
        extra_tokens_n = seqlen_k % BLOCK_N
        seqlen_k_faligned = seqlen_k - extra_tokens_n
    else:
        k_padded = False
        seqlen_k_faligned = seqlen_k

    q_offset = off_h * stride_qh + off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
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
        shape=(seqlen_k, head_dim),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504089
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    if q_padded:
        if PADDED_HEAD:
            q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        if PADDED_HEAD:
            q = tl.load(Q_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and attn_fwd_inner gets 3 as its STAGE
    off_zh = off_z * num_h + off_h * 1
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_zh * seqlen_q * seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0

    # Stage 1: off-band (for causal) or non-boundary (for irregular seqlen_k) blocks
    if CAUSAL:
        # Causal = True
        seqlen_k_low = 0
        seqlen_k_high = min(seqlen_k_faligned, start_m * BLOCK_M)
    else:
        # Causal = False
        seqlen_k_low = 0
        seqlen_k_high = seqlen_k_faligned
    acc, l_i, m_i = attn_fwd_inner(
        acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
        start_m, seqlen_q, seqlen_k_low, seqlen_k_high, False,
        dropout_p, seqlen_k, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
        False, offs_m, offs_n,
        pre_load_v,
        ENABLE_DROPOUT,
        RETURN_ENCODED_SOFTMAX,
        MARGINAL_BLOCK=False,
        PADDED_HEAD=PADDED_HEAD,
    )
    # Stage 2: on-band or boundary blocks
    if CAUSAL or k_padded:
        seqlen_k_low = seqlen_k_high
        if CAUSAL:
            seqlen_k_high = min(seqlen_k, start_m * BLOCK_M + BLOCK_M)
        else:
            seqlen_k_high = seqlen_k
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_q, seqlen_k_low, seqlen_k_high, k_padded,
            dropout_p, seqlen_k, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            CAUSAL, offs_m, offs_n,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            MARGINAL_BLOCK=True,
            PADDED_HEAD=PADDED_HEAD,
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    m_ptrs = M + off_zh * seqlen_q + offs_m
    # Check for last block_M
    if q_padded:
        overflow_size = (start_m * BLOCK_M + BLOCK_M) - seqlen_q
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        # This is a > check because mask being 0 blocks the store.
        m_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=m_ptrs_mask)
    else:
        tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if q_padded:
        if PADDED_HEAD:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))
    else:
        if PADDED_HEAD:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,))
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
