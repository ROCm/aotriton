#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import fast_philox

@triton.jit
def debug_fill_dropout_rng(R,
                           stride_rz, stride_rh, stride_rm, stride_rn,
                           seqlen_q, seqlen_k,
                           philox_seed : tl.uint64,
                           philox_offset_base : tl.uint64,
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
    # Minimal SCALE that SCALE * MAX_I32 > 0.5. Tested with
    #   >>> scale = np.array([2.32830657e-10], dtype=np.float32); (scale * max_i32) > 0.5
    #   array([False])
    #   >>> scale = np.array([2.32830658e-10], dtype=np.float32); (scale * max_i32) > 0.5
    #   array([ True])
    #   >>> (np.array([2.328306571e-10], dtype=np.float32) - np.array([2.32830657e-10], dtype=np.float32)) == 0.0
    #   array([ True])  # The final `1` exceeds the precision limit of fp32
    #   >>> (np.array([2.32830658e-10], dtype=np.float32) - np.array([2.328306579e-10], dtype=np.float32)) == 0.0
    #   array([ True])  # The final digits exceeds the precision limit of fp32
    # Note1: Do not use tl.uint_to_uniform_float, it translates max_u32 to 0.0
    # Note2: Do not pick SCALE * MAX_I32 < 0.5 (== is not possible due to fp
    #        errors), because it cannot generate 1.0 otherwise
    SCALE : tl.constexpr = 2.32830658e-10
    for start_n in range(0, seqlen_k, BLOCK_N):
        philox_offset = batch_philox_offset + start_m * BLOCK_M * seqlen_k + start_n
        rng = fast_philox(philox_seed, philox_offset, BLOCK_M, BLOCK_N, seqlen_k)
        if R.type.element_ty == tl.float32:
            # Attept to translate [-MAX_I32-1, MAX_I32] to [0.0, 1.0]
            # 0 should be translated to 0.0
            rng = (rng * SCALE + 0.5).to(tl.float32)
            tl.clamp(rng, 0.0, 1.0) # In case |rng| >= MAX_I32
            tl.store(R_block_ptr,
                     rng,
                     boundary_check=(0,1))
        elif R.type.element_ty == tl.int32:
            tl.store(R_block_ptr, rng, boundary_check=(0,1))
        else:
            tl.static_assert(False, 'R.type.element_ty must be either float32 or int32')
        R_block_ptr = tl.advance(R_block_ptr, (0, BLOCK_N))

@triton.jit
def debug_fill_dropout_rng_tensor(R,
                                  stride_rz, stride_rh, stride_rm, stride_rn,
                                  seqlen_q, seqlen_k,
                                  philox_seed_ptr: '*u64',
                                  philox_offset1 : '*u64',
                                  philox_offset2 : tl.uint64,
                                  BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr,
                                  ):
    philox_seed = tl.load(philox_seed_ptr)
    philox_offset_base = philox_offset2
    philox_offset_base += tl.load(philox_offset1)
    debug_fill_dropout_rng(R,
                           stride_rz, stride_rh, stride_rm, stride_rn,
                           seqlen_q, seqlen_k,
                           philox_seed,
                           philox_offset_base,
                           BLOCK_M,
                           BLOCK_N,
                           )
