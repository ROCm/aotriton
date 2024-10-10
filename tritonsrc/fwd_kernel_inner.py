# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import dropout_mask, dropout_rng, dropout_offsets
from masked_load_store import load_fn, mstore2d
from triton.language.extra import libdevice

@triton.jit
def attn_fwd_inner(
        # Problem Description
        acc, l_i, m_i, qk_scale, bias_scale,
        q, k_ptrs, v_ptrs, bias_ptrs,
        stride_kn, stride_vk, stride_bn,
        seqlen_q, seqlen_k, head_dim,
        # Sub-problem range
        start_m, block_min, block_max,
        # Auxiliary options
        ## Dropout
        dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
        ## Debug Return
        encoded_sm_base,
        ## Irregular support
        offs_n_causal, masked_blocks, n_extra_tokens,
        ## Alibi
        alibi_slope,
        # constexpr starts here
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PRE_LOAD_V: tl.constexpr,
        MASK_STEPS: tl.constexpr,
        ENABLE_DROPOUT: tl.constexpr,
        RETURN_ENCODED_SOFTMAX: tl.constexpr,
        PADDED_HEAD: tl.constexpr,
):
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_d, k_offs_n, head_dim, seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_d, seqlen_k, head_dim)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
                size_n = start_n + offs_n[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = offs_m[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        qk += tl.dot(q, k)
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, offs_m, bias_offs_n, seqlen_q, seqlen_k)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += bias * bias_scale

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, global_m_positions,
                                              global_n_positions)
            qk += alibi_block * bias_scale

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # FIXME: when sm_scale = 0.0 and MASK_STEPS/CAUSAL = True
        #        qk * qk_scale = nan
        p = tl.math.exp2(qk_scale * (qk - m_ij[:, None]))

        # tl.debug_barrier()
        # tl.device_print('m_ij', m_ij)
        # # tl.device_print('log p', qk * qk_scale - m_ij[:, None])
        # # # tl.device_print('p', p)
        # tl.debug_barrier()

        if MASK_STEPS or CAUSAL:
            if qk_scale == 0.0:
                p = tl.where(libdevice.isnan(p), 0.0, p)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * max_seqlen_k + start_n
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                mstore2d(tl.where(keep, p, -p).to(q.type.element_ty),
                         BLOCK_M,
                         BLOCK_N,
                         o_base=encoded_sm_base,
                         o_start_row=start_m * BLOCK_M,
                         o_start_col=start_n,
                         o_rows=seqlen_q,
                         o_cols=seqlen_k,
                         stride_row=max_seqlen_k,
                         stride_col=1)
                # tl.store(encoded_sm_ptrs, tl.where(keep, p, -p).to(q.type.element_ty))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            mstore2d(p.to(q.type.element_ty),
                     BLOCK_M,
                     BLOCK_N,
                     o_base=encoded_sm_base,
                     o_start_row=start_m * BLOCK_M,
                     o_start_col=start_n,
                     o_rows=seqlen_q,
                     o_cols=seqlen_k,
                     stride_row=max_seqlen_k,
                     stride_col=1)
            # tl.store(encoded_sm_ptrs, p.to(q.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(qk_scale * (m_i - m_ij))
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_d, seqlen_k, head_dim)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij

        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        # if RETURN_ENCODED_SOFTMAX:
        #     encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i
