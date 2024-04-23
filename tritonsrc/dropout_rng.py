#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from fwd_kernel import dropout_rng

@triton.jit
def debug_fill_dropout_rng(R,
                           stride_rz, stride_rh, stride_rm, stride_rn,
                           seqlen_q, seqlen_k,
                           philox_seed,
                           philox_offset_base,
                           BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr,
                           ):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index
    d_offset = off_h * stride_rh + off_z * stride_rz
    num_h = tl.num_programs(1)
    off_zh = off_z * num_h + off_h * 1
    batch_philox_offset = philox_offset_base + off_zh * seqlen_q * seqlen_k
    R_block_ptr = tl.make_block_ptr(
        base=R + d_offset,
        shape=(seqlen_q, seqlen_k),
        strides=(stride_rm, stride_rn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    for start_n in range(0, seqlen_k, BLOCK_N):
        philox_offset = batch_philox_offset + start_m * BLOCK_M * seqlen_k + start_n
        rng = dropout_rng(philox_seed, philox_offset, BLOCK_M, BLOCK_N, seqlen_k)
        tl.store(R_block_ptr, rng.to(R_block_ptr.type.element_ty), boundary_check=(0,1))
        R_block_ptr = tl.advance(R_block_ptr, (0, BLOCK_N))
