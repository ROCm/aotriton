# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl

from fwd_kernel_inner import attn_fwd_inner

@triton.jit
def attn_fwd_common(
        Q_block_ptr,
        K_block_ptr,
        V_block_ptr,
        B_block_ptr,
        O_block_ptr,
        M_ptr_base,
        sm_scale,
        start_m,
        seqlen_q,
        seqlen_k,
        seqlen_k_faligned,
        q_padded,
        dropout_p,
        philox_seed,
        batch_philox_offset,
        encoded_softmax_block_ptr,
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
    k_padded = seqlen_k != seqlen_k_faligned
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
        acc, l_i, m_i, q, K_block_ptr, V_block_ptr, B_block_ptr,
        start_m, seqlen_q, q_padded, seqlen_k_low, seqlen_k_high, False,
        dropout_p, seqlen_k, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N,
        False, offs_m, offs_n,
        pre_load_v,
        ENABLE_DROPOUT,
        RETURN_ENCODED_SOFTMAX,
        MARGINAL_BLOCK=False,
        PADDED_HEAD=PADDED_HEAD,
        BIAS_TYPE=BIAS_TYPE,
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
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, B_block_ptr,
            start_m, seqlen_q, q_padded, seqlen_k_low, seqlen_k_high, k_padded,
            dropout_p, seqlen_k, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N,
            CAUSAL, offs_m, offs_n,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            MARGINAL_BLOCK=True,
            PADDED_HEAD=PADDED_HEAD,
            BIAS_TYPE=BIAS_TYPE,
        )
    # epilogue
    # write back m
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # m_ptrs = M + off_zh * seqlen_q + offs_m
    m_ptrs = M_ptr_base + offs_m
    # Check for last block_M
    if q_padded:
        overflow_size = (start_m * BLOCK_M + BLOCK_M) - seqlen_q
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        # This is a > check because mask being 0 blocks the store.
        m_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(m_ptrs, m_i + tl.math.log2(l_i), mask=m_ptrs_mask)
    else:
        tl.store(m_ptrs, m_i + tl.math.log2(l_i))

    if q_padded:
        if PADDED_HEAD:
            tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0,1))
        else:
            tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0,))
    else:
        if PADDED_HEAD:
            tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(1,))
        else:
            tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty))
