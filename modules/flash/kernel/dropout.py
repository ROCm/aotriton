# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl

PHILOX_RN_PER_OFFSET = tl.constexpr(4)

@triton.jit
def fast_dropout_offsets(philox_seed, philox_offset, M, N, stride):
    ms = tl.arange(0, M)
    ns = tl.arange(0, N)
    return philox_offset + ms[:, None] * stride + ns[None, :]

@triton.jit
def fast_philox(philox_seed, philox_offset, M : tl.constexpr, N : tl.constexpr, stride):
    rng_offsets = fast_dropout_offsets(philox_seed, philox_offset, M, N, stride)
    # Get 4 uint64 blocks
    r0, r1, r2, r3 = tl.randint4x(philox_seed, rng_offsets)
    if False:
        tl.static_assert(r0.dtype == tl.uint64,
                        "fast_philox expects tl.randint4x returns uint32. "
                        "The behavior has been changed in https://github.com/triton-lang/triton/pull/6832")
        tl.static_assert(PHILOX_RN_PER_OFFSET == 8)
        # Cast them to 8 int32 blocks
        # NOTE: dead branch (`if False`). Its nested tl.join has the lane
        # ordering the live branch below is written to avoid, and would need
        # the same swap plus a decision on what its 8-randoms-per-offset
        # consumer expects. Left alone deliberately: it is unreachable, so
        # there is nothing to check a change against.
        r64 = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(M, N * 4)  # 4x uint64 blocks
        r64_hi = ((r64 >> 32) & 0xffffffff).to(tl.uint32)
        r64_lo = (r64 & 0xffffffff).to(tl.uint32)
        # Cast to signed integer due to PyTorch limit:
        #   "compare_cuda" not implemented for 'UInt32'
        r32 = tl.join(r64_hi, r64_lo).reshape(M, N * 8).to(tl.int32, bitcast=True)
    else:
        tl.static_assert(r0.dtype == tl.uint32,
                        "fast_philox expects tl.randint4x returns uint32. "
                        "The behavior has been changed in https://github.com/triton-lang/triton/pull/6832")
        tl.static_assert(PHILOX_RN_PER_OFFSET == 4)
        # Contract: the four randoms tl.randint4x draws from one Philox offset
        # occupy that offset's four consecutive key columns in generation
        # order -- for offset n, column 4*n+i holds r<i>. That is what lets an
        # independent implementation of the same mapping reproduce the mask bit
        # for bit. Inside AOTriton the agreement is free, because every
        # consumer -- the forward kernel, both backward kernels, and the debug
        # encoded-softmax kernel -- reaches the mask through this one function.
        #
        # The pairing below is therefore (r0,r2),(r1,r3), and must stay that
        # way. tl.join appends a MINOR axis, so the OUTER join of a nest owns
        # the last (fastest-varying) axis and the inner joins own the one
        # before it: join(join(a,b), join(c,d)) has shape (..., 2, 2) indexed
        # [inner, outer] and flattens row-major to [a, c, b, d] -- the outer
        # selector alternating first. The natural-looking (r0,r1),(r2,r3) would
        # lay the columns out as [r0, r2, r1, r3] and break the contract above;
        # swapping the middle two inputs cancels that and yields the required
        # [r0, r1, r2, r3].
        #
        # Getting this wrong only permutes four columns, so the mask's
        # distribution, its per-offset keep count and every statistical
        # property survive intact -- and no test that takes its reference mask
        # from this kernel can see the difference. Check the column order
        # directly.
        r32 = tl.join(tl.join(r0, r2), tl.join(r1, r3)).reshape(M, N * 4).to(tl.int32, bitcast=True)
    return r32

@triton.jit
def fast_dropout_mask(philox_seed,
                      dropout_p : tl.int32,
                      offset_base,
                      offset_x,
                      offset_y,
                      M : tl.constexpr,
                      N : tl.constexpr,
                      stride):
    tl.static_assert(N % PHILOX_RN_PER_OFFSET == 0, "fast_dropout_mask only supports N % 8 == 0")
    tl.static_assert(philox_seed.dtype == tl.uint64, "fast_dropout_mask only accepts uint64 philox_seed")
    tl.static_assert(offset_base.dtype == tl.uint64, "fast_dropout_mask only accepts uint64 philox_offset")
    tl.static_assert(dropout_p.dtype == tl.int32, "fast_dropout_mask only accepts int32 dropout_p")
    # Derive of BLOCK_N indepedent int4x offsets algorithm
    # Old offset algorithm:
    #   size = M * N
    #   offset = base + x * stride + y
    # Or,
    #   offsets = base[x:x+M, y:y+N]
    #
    # New algorithm can generate PHILOX_RN_PER_OFFSET u32 PRNGs from one offset
    # So the demands of offsets is much smaller:
    #   offsets = base0[x:x+M, y1:y1+N//PHILOX_RN_PER_OFFSET]
    # Here y1 = y // PHILOX_RN_PER_OFFSET, base0 = base but stride0 = cdiv(stride, PHILOX_RN_PER_OFFSET)
    #
    # Note: the caller of fast_dropout_mask must ensure:
    #  1. stepping of y is N
    #  2. y % PHILOX_RN_PER_OFFSET == 0
    philox_offset = offset_base + offset_x * stride + offset_y // PHILOX_RN_PER_OFFSET
    rng_output = fast_philox(philox_seed, philox_offset, M, N // PHILOX_RN_PER_OFFSET, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep
