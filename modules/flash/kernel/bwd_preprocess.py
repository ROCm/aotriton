#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
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
from varlen_bits import (
    decode_addressing,
    lse_token_pitch,
    lse_row_addressing,
)
from composed_tensors import (
    composed_ptrs,
    composed_load,
    composed_inner_product_fp32,
)


@triton.jit
def bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    # Exactly the Q-side subsequence of bwd_kernel_dk_dv's argument order. The
    # ATI linker topologically sorts every sub-kernel's list into one union, so
    # a kernel whose order is not a subsequence of the key kernel's is a cycle,
    # not merely untidy.
    seqinfo_q0,
    varlen_bits,
    max_seqlen_q,
    seqinfo_q1,
    hdim_vo,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    # TODO: Put this decomposition into a @triton.jit function when tuple support is more complete
    tl.static_assert(D_HEAD > 0, 'D_HEAD must be greater than 0')
    D_HEAD_R0 : tl.constexpr = D_HEAD
    D_HEAD0 : tl.constexpr = 2 ** (D_HEAD_R0.bit_length() - 1)
    D_HEAD_R1 : tl.constexpr = D_HEAD_R0 - D_HEAD0
    D_HEAD1 : tl.constexpr = 2 ** (D_HEAD_R1.bit_length() - 1) if D_HEAD_R1 > 0 else 0
    D_HEAD_R2 : tl.constexpr = D_HEAD_R1 - D_HEAD1
    D_HEAD2 : tl.constexpr = 2 ** (D_HEAD_R2.bit_length() - 1) if D_HEAD_R2 > 0 else 0
    D_HEAD_R3 : tl.constexpr = D_HEAD_R2 - D_HEAD2

    tl.static_assert(D_HEAD_R3 == 0, f'D_HEAD = {D_HEAD} = 0b{D_HEAD:b} cannot be factored into <= 3 power of two values')
    tl.static_assert(D_HEAD1 > 0 or D_HEAD2 == 0, 'Only trailing D_HEADx can be 0')

    off_m = tl.program_id(0) * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # sequence index z, decoded by varlen_bits
    num_h = tl.num_programs(1)
    num_seq = tl.num_programs(2)  # = N

    seqlen_q, q_row_off, batch_index = decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, off_z)
    # At function top level, not inside a loop, so Triton accepts the bare
    # return. Inert for dense: the grid is cdiv(max_seqlen_q, BLOCK_M).
    if off_m >= seqlen_q:
        return
    lse_tokens = lse_token_pitch(varlen_bits, max_seqlen_q,
                                 seqinfo_q0, seqinfo_q1, num_seq)

    # BLOCK POINTERS ARE KEPT FOR DOCUMENTATION PURPOSE
    # o_offset = off_h * stride_oh + off_z * stride_oz
    # O_block_ptr = tl.make_block_ptr(
    #     base=Out + o_offset,
    #     shape=(seqlen_q, hdim_vo),
    #     strides=(stride_om, stride_on),
    #     offsets=(off_m, 0),
    #     block_shape=(BLOCK_M, D_HEAD),
    #     order=(1, 0)
    # )
    o_ptrs0, o_ptrs1, o_ptrs2 = composed_ptrs(Out,
                                              stride_oz, stride_oh, stride_om, stride_on,
                                              batch_index, off_h, q_row_off + offs_m,
                                              D_HEAD0, D_HEAD1, D_HEAD2)
    # do_offset = off_h * stride_doh + off_z * stride_doz
    # DO_block_ptr = tl.make_block_ptr(
    #     base=DO + do_offset,
    #     shape=(seqlen_q, hdim_vo),
    #     strides=(stride_dom, stride_don),
    #     offsets=(off_m, 0),
    #     block_shape=(BLOCK_M, D_HEAD),
    #     order=(1, 0)
    # )
    do_ptrs0, do_ptrs1, do_ptrs2 = composed_ptrs(DO,
                                                 stride_doz, stride_doh, stride_dom, stride_don,
                                                 batch_index, off_h, q_row_off + offs_m,
                                                 D_HEAD0, D_HEAD1, D_HEAD2)

    o0, o1, o2 = composed_load(o_ptrs0, o_ptrs1, o_ptrs2,
                               offs_m,
                               D_HEAD0, D_HEAD1, D_HEAD2,
                               seqlen_q, hdim_vo,
                               other=0.0,
                               PADDED_ROW=True,
                               PADDED_COL=PADDED_HEAD,
                               TRANSPOSED=False)
    do0, do1, do2 = composed_load(do_ptrs0, do_ptrs1, do_ptrs2,
                                  offs_m,
                                  D_HEAD0, D_HEAD1, D_HEAD2,
                                  seqlen_q, hdim_vo,
                                  other=0.0,
                                  PADDED_ROW=True,
                                  PADDED_COL=PADDED_HEAD,
                                  TRANSPOSED=False)
    # # load
    # # o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # # do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # o = tl.load(O_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)
    # do = tl.load(DO_block_ptr, boundary_check=(0,1), padding_option="zero").to(tl.float32)

    # compute
    # delta = tl.sum(o * do, axis=1)
    delta = composed_inner_product_fp32(o0, o1, o2,
                                        do0, do1, do2,
                                        D_HEAD0, D_HEAD1, D_HEAD2,
                                        axis=1)

    # write-back. Delta is a row-wise fp32 side output and shares LSE's layout
    # by construction, so it goes through the same addressing -- pitch included.
    lse_base, lse_pitch = lse_row_addressing(varlen_bits, batch_index, off_h,
                                             num_h, lse_tokens, q_row_off)
    # Check for OOB accesses
    delta_ptrs = Delta + lse_base + (off_m + tl.arange(0, BLOCK_M)) * lse_pitch
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow, dtype=tl.int32)
        mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=mask)
    else:
        tl.store(delta_ptrs, delta)
