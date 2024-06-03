# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from fwd_kernel_common import attn_fwd_common

@triton.jit
def attn_fwd_varlen(
    Q, K, V, B, sm_scale, M, Out,
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vk, stride_vh, stride_vn,
    stride_bm, stride_bh, stride_bn,
    stride_om, stride_oh, stride_on,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,  # Needed for dropout, otherwise philox_offset 
    max_seqlen_k,  # will be overwhelmingly complicated
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
    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
    seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
    # We have a one-size-fits-all grid in id(0). Some seqlens might be too
    # small for all start_m so for those we return early.
    if start_m * BLOCK_M > seqlen_q:
        return
    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
    seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start

    num_h = tl.num_programs(1)
    num_z = tl.num_programs(2)
    off_zh = off_z * num_h + off_h * 1
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

    q_offset = off_h * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_h * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(head_dim, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_h * stride_vh + cu_seqlens_k_start * stride_vn
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
    else:
        tl.static_assert(False, f'bias is unsupported in varlen kernel')
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_h * cu_seqlens_q_start * max_seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0
    # write back O
    o_offset = off_h * stride_oh + cu_seqlens_q_start * stride_om
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, head_dim),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    M_ptr_base = M + off_zh * cu_seqlens_q_start
    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_h * cu_seqlens_q_start * max_seqlen_k
    else:
        batch_philox_offset = 0

    attn_fwd_common(Q_block_ptr,
                    K_block_ptr,
                    V_block_ptr,
                    B_block_ptr,
                    O_block_ptr,
                    M_ptr_base,
                    sm_scale,
                    dropout_p,
                    start_m,
                    seqlen_q,
                    seqlen_k,
                    seqlen_k_faligned,
                    q_padded,
                    philox_seed,
                    batch_philox_offset,
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
