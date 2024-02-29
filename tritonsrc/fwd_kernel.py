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
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    bias_ptr,
    start_m,
    seqlen_k,
    seqlen_k_faligned,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_BLOCK: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1: # "Solid" blocks of Causal masks
        lo, hi = 0, min(seqlen_k_faligned, start_m * BLOCK_M)
    elif STAGE == 2: # "Semi-solid", or "Transition" block of Causal mask
        # Must use BLOCK_M, because the starting position of semi-solid block
        # is determined by start_m * BLOCK_M
        lo, hi = start_m * BLOCK_M, min(seqlen_k_faligned, start_m * BLOCK_M + BLOCK_M)
        # lo = tl.multiple_of(lo, BLOCK_M)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
        tl.static_assert(PADDED_BLOCK == False, 'STAGE=2 should not be used with PADDED_BLOCK=True')
    # So here, we are computing the elements for that last irregular block.
    # In the loop,  we will mask the elements of BLOCK_N that do not exist.
    elif PADDED_BLOCK:
        lo, hi = seqlen_k_faligned, seqlen_k_faligned + BLOCK_N
        # lo = tl.multiple_of(lo, BLOCK_N)
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
        if bias_ptr is not None:
            if bias_ptr.type.element_ty.is_block():
                bias_ptr = tl.advance(bias_ptr, (0, lo))
            else:
                bias_ptr += lo
    # causal = False
    else:
        lo, hi = 0, seqlen_k_faligned
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        '''
        if STAGE == 1 or STAGE == 3:
            start_n = tl.multiple_of(start_n, BLOCK_N)
        '''
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if (PADDED_BLOCK or STAGE == 2) or PADDED_HEAD:
            k = tl.load(K_block_ptr, boundary_check=(1,0), padding_option="zero")
        else:
            k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        if PRE_LOAD_V:
            if (PADDED_BLOCK or STAGE == 2) or PADDED_HEAD:
                v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
            else:
                v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if STAGE == 2:
            mask = OFFS_M[:, None] >= (start_n + OFFS_N[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        if PADDED_BLOCK:
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.float32)
            size_n = start_n + OFFS_N[None,:]
            mask = size_n < boundary_m[:,None]
            qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        if bias_ptr is not None:
            if PADDED_BLOCK:
                if bias_ptr.type.element_ty.is_block():
                    bias = tl.load(bias_ptr,boundary_check=(1,), padding_option="zero")
                else:
                    size_n = start_n + OFFS_N
                    boundary_n = tl.full([BLOCK_N], seqlen_k, dtype=tl.float32)
                    bias_padding = tl.full([BLOCK_N], 0, dtype=tl.float32)
                    bias = tl.load(bias_ptr, mask=size_n < boundary_n, other=bias_padding)
            else:
                bias = tl.load(bias_ptr)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += (bias * 1.44269504089)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # Note about the conflicts of Flash attention algorithm and PyTorch's CUDA implementation
        # PyTorch needs to return softmax(qk) (dropout mask encoded in sign bits)
        # While Flash attention paper compute the dropout AFTER exp2(qk- m_ij)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty), boundary_check=(0,1))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty), boundary_check=(0,1))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            if (PADDED_BLOCK or STAGE == 2) or PADDED_HEAD:
                v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
            else:
                v = tl.load(V_block_ptr, boundary_check=(1,), padding_option="zero")
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if bias_ptr is not None:
            if bias_ptr.type.element_ty.is_block():
                bias_ptr = tl.advance(bias_ptr, (0, BLOCK_N))
            else:
                bias_ptr += BLOCK_N
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
def attn_fwd(
    Q, K, V, B, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bn,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlens_q, max_seqlens_k,
    head_dim_q, head_dim_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    VARLEN: tl.constexpr,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    PADDED_HEAD: tl.constexpr,  # Cannot be inferred by AOT Compiler
):
    is_mqa = head_dim_q != head_dim_k
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    if VARLEN:
        cu_seqlens_q_this = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_next = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_next - cu_seqlens_q_this
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_this = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_next = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_next - cu_seqlens_k_this
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = max_seqlens_q
        seqlen_k = max_seqlens_k
    if is_mqa:
        off_h_k = off_h_q % head_dim_k
    else:
        off_h_k = off_h_q
    need_padding = False
    if seqlen_k < BLOCK_N:
        need_padding = True
        extra_tokens_n = BLOCK_N - seqlen_k
        seqlen_k_faligned = 0 # floor aligned
    elif seqlen_k % BLOCK_N:
        need_padding = True
        extra_tokens_n = seqlen_k % BLOCK_N
        seqlen_k_faligned = seqlen_k - extra_tokens_n
    else:
        seqlen_k_faligned = seqlen_k

    q_offset = off_z * stride_qz +  off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim_q),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim_k, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, head_dim_k),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    off_zh = off_z * num_h + off_h_q * 1
    if BIAS_TYPE != 0:
        if BIAS_TYPE == 1:
            bias_ptr = B + off_h_q * stride_bh + offs_n
        elif BIAS_TYPE == 2:
            bias_ptr = tl.make_block_ptr(
                base=B + off_h_q * stride_bh,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
    else:
        bias_ptr = None
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504089
    # load q: it will stay in SRAM throughout on NV GPUs but in VGPRs on AMD GPUs
    q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # We can ask to return the dropout mask without actually doing any dropout. In
    # this case, we return an invalid pointer so indicate the mask is not valid.
    # TODO: Fix encoded softmax. It currently uses just h_q in the base offset.
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
    if STAGE & 1:
        # equal to N_CTX if N_CTX is already a multiple of block_M
        if seqlen_k >= BLOCK_N:
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                bias_ptr,
                start_m, seqlen_k, seqlen_k_faligned,
                dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                4 - STAGE, offs_m, offs_n,
                PRE_LOAD_V,
                ENABLE_DROPOUT,
                RETURN_ENCODED_SOFTMAX,
                PADDED_BLOCK=False,
                PADDED_HEAD=PADDED_HEAD,
            )
        tl.debug_barrier()
        if need_padding:
            acc, l_i, m_i = _attn_fwd_inner(
                acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
                bias_ptr,
                start_m, seqlen_k, seqlen_k_faligned,
                dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
                BLOCK_M, BLOCK_DMODEL, BLOCK_N,
                4 - STAGE, offs_m, offs_n,
                PRE_LOAD_V,
                ENABLE_DROPOUT,
                RETURN_ENCODED_SOFTMAX,
                PADDED_BLOCK=True,
                PADDED_HEAD=PADDED_HEAD,
            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            bias_ptr,
            start_m, seqlen_k, seqlen_k_faligned,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            2, offs_m, offs_n,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            PADDED_BLOCK=False,
            PADDED_HEAD=PADDED_HEAD,
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    m_ptrs = M + off_zh * max_seqlens_q + offs_m
    # Check for last block_M
    overflow_size = (start_m * BLOCK_M + BLOCK_M) - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        # This is a > check because mask being 0 blocks the store.
        m_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=m_ptrs_mask)
    else:
        tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim_q),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))  # Don't exceed shape, makes sure padding isn't put in output.
