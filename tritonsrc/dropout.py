import triton
import triton.language as tl

@triton.jit
def fast_dropout_offsets(philox_seed, philox_offset, M, N, stride):
    ms = tl.arange(0, M)
    ns = tl.arange(0, N // 8) * 8
    return philox_offset + ms[:, None] * stride + ns[None, :]

@triton.jit
def fast_philox(philox_seed, philox_offset, M : tl.constexpr, N : tl.constexpr, stride):
    rng_offsets = fast_dropout_offsets(philox_seed, philox_offset, M, N, stride)
    # Get 4 uint64 blocks
    r0, r1, r2, r3 = tl.randint4x(philox_seed, rng_offsets)
    tl.static_assert(r0.dtype == tl.uint64, "fast_philox expects tl.randint4x returns uint64")
    # Cast them to 8 uint32 blocks
    r64 = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(M, N // 2)
    r64_hi = ((r64 >> 32) & 0xffffffff).to(tl.uint32)
    r64_lo = (r64 & 0xffffffff).to(tl.uint32)
    # Cast to signed integer due to Torch limit:
    #   "compare_cuda" not implemented for 'UInt32'
    r32 = tl.join(r64_hi, r64_lo).reshape(M, N).to(tl.int32, bitcast=True)
    return r32

@triton.jit
def fast_dropout_mask(philox_seed,
                      philox_offset,
                      dropout_p : tl.int32,
                      M : tl.constexpr,
                      N : tl.constexpr,
                      stride):
    tl.static_assert(philox_seed.dtype == tl.uint64, "fast_dropout_mask only accepts uint64 philox_seed")
    tl.static_assert(philox_offset.dtype == tl.uint64, "fast_dropout_mask only accepts uint64 philox_offset")
    tl.static_assert(dropout_p.dtype == tl.int32, "fast_dropout_mask only accepts int32 dropout_p")
    rng_output = fast_philox(philox_seed, philox_offset, M, N, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep
