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
from fwd_kernel_common import attn_fwd_common

@triton.jit
def attn_fwd(
    Q, K, V, B, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_on,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,   # set num_seqlens to zero to ignore cu_seqlens_q/k
    max_seqlen_q,  # and use max_seqlen_q/k for all seqlen_q/k 
    max_seqlen_k,
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
    BIAS_TYPE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index
    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    off_zh = off_z * num_h + off_h * 1
    # FIXME: Better pattern for this branch? It's copied into three kernels
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

    if start_m * BLOCK_M + BLOCK_M > seqlen_q:
        q_padded = True
    else:
        q_padded = False
    if seqlen_k < BLOCK_N:
        seqlen_k_faligned = 0 # floor aligned
    elif seqlen_k % BLOCK_N:
        extra_tokens_n = seqlen_k % BLOCK_N
        seqlen_k_faligned = seqlen_k - extra_tokens_n
    else:
        seqlen_k_faligned = seqlen_k

    q_offset = off_h * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_h * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_h * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, head_dim),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    if BIAS_TYPE == 0:
        B_block_ptr = 0
    elif BIAS_TYPE == 1:
        B_block_ptr = tl.make_block_ptr(
                base=B + off_h * stride_bh + batch_index * stride_bz,
                shape=(seqlen_q, seqlen_k),
                strides=(stride_bm, stride_bn),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_zh * max_seqlen_q * max_seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(max_seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0
    # write back O
    o_offset = off_h * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    M_ptr_base = M + off_zh * max_seqlen_q
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_zh * max_seqlen_q * max_seqlen_k
    else:
        batch_philox_offset = 0

    attn_fwd_common(Q_block_ptr,
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
                    max_seqlen_k,
                    encoded_softmax_block_ptr,
                    CAUSAL=CAUSAL,
                    BLOCK_M=BLOCK_M,
                    BLOCK_DMODEL=BLOCK_DMODEL,
                    BLOCK_N=BLOCK_N,
                    pre_load_v=pre_load_v,
                    ENABLE_DROPOUT=ENABLE_DROPOUT,
                    RETURN_ENCODED_SOFTMAX=RETURN_ENCODED_SOFTMAX,
                    PADDED_HEAD=PADDED_HEAD,
                    BIAS_TYPE=BIAS_TYPE)
