# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import dropout_mask, dropout_rng, dropout_offsets

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr, B_block_ptr,
    start_m,
    seqlen_q,
    q_padded,
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
    BIAS_TYPE: tl.constexpr,
):
    lo, hi = seqlen_k_low, seqlen_k_high
    if MARGINAL_BLOCK:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, lo))
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, lo))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        # -- compute qk ----
        # MARGINAL_BLOCK serves as a compile-time switch for first attn_fwd_inner calls to "solid" blocks
        k = tl.load(K_block_ptr, boundary_check=(1,0), padding_option="zero")
        '''
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
        '''
        if pre_load_v:
            v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
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
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            '''
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            '''
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
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
            v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
            '''
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
            '''
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(V_block_ptr.type.element_ty), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i
