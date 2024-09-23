import triton
import triton.language as tl

# 16
@triton.jit
def unrolled0_fbt_ptrs(m_ptrs,
                       k_ptrs,
                       stride_k,
                       lo : tl.constexpr,
                       hi : tl.constexpr,
                       BLOCK_K : tl.constexpr):
    # tl.static_print(f'unrolled0_fbt_ptrs {lo=} {hi=}')
    tl.static_assert(hi - lo == BLOCK_K)
    return m_ptrs + (k_ptrs + lo) * stride_k

# 32
@triton.jit
def unrolled1_fbt_ptrs(m_ptrs,
                       k_ptrs,
                       stride_k,
                       lo : tl.constexpr,
                       hi : tl.constexpr,
                       BLOCK_K : tl.constexpr):
    # tl.static_print(f'unrolled1_fbt_ptrs {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    half1st = unrolled0_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo, lo + half, BLOCK_K)
    half2nd = unrolled0_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo + half, hi, BLOCK_K)
    return tl.join(half1st, half2nd)

# 64
@triton.jit
def unrolled2_fbt_ptrs(m_ptrs,
                       k_ptrs,
                       stride_k,
                       lo : tl.constexpr,
                       hi : tl.constexpr,
                       BLOCK_K : tl.constexpr):
    # tl.static_print(f'unrolled2_fbt_ptrs {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    half1st = unrolled1_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo, lo + half, BLOCK_K)
    half2nd = unrolled1_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo + half, hi, BLOCK_K)
    return tl.join(half1st, half2nd)

# 128
@triton.jit
def unrolled3_fbt_ptrs(m_ptrs,
                       k_ptrs,
                       stride_k,
                       lo : tl.constexpr,
                       hi : tl.constexpr,
                       BLOCK_K : tl.constexpr):
    # tl.static_print(f'unrolled3_fbt_ptrs {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    half1st = unrolled2_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo, lo + half, BLOCK_K)
    half2nd = unrolled2_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo + half, hi, BLOCK_K)
    return tl.join(half1st, half2nd)

# 256
@triton.jit
def unrolled4_fbt_ptrs(m_ptrs,
                       k_ptrs,
                       stride_k,
                       lo : tl.constexpr,
                       hi : tl.constexpr,
                       BLOCK_K : tl.constexpr):
    # tl.static_print(f'unrolled4_fbt_ptrs {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    half1st = unrolled3_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo, lo + half, BLOCK_K)
    half2nd = unrolled3_fbt_ptrs(m_ptrs, k_ptrs, stride_k, lo + half, hi, BLOCK_K)
    return tl.join(half1st, half2nd)

@triton.jit
def fbt_ptrs(m_ptrs,
             k_ptrs,
             stride_k,
             k_lo : tl.constexpr,
             k_hi : tl.constexpr,
             BLOCK_K : tl.constexpr):
    N_BLOCKS : tl.constexpr = (k_hi - k_lo) // BLOCK_K
    if N_BLOCKS == 1:
        return unrolled0_fbt_ptrs(m_ptrs, k_ptrs, stride_k, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 2:
        return unrolled1_fbt_ptrs(m_ptrs, k_ptrs, stride_k, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 4:
        return unrolled2_fbt_ptrs(m_ptrs, k_ptrs, stride_k, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 8:
        return unrolled3_fbt_ptrs(m_ptrs, k_ptrs, stride_k, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 16:
        return unrolled4_fbt_ptrs(m_ptrs, k_ptrs, stride_k, k_lo, k_hi, BLOCK_K)
    else:
        tl.static_assert(False, f'Unsupport N_BLOCKS {N_BLOCKS} from ({k_hi=} - {k_lo=})/{BLOCK_K=}')

'''
Full Binary Tree load

Load (M, K) matrix
'''

# @triton.jit
# def fbt_ptrs(m_ptrs,
#              k_ptrs,
#              stride_k,
#              K : tl.constexpr,
#              BLOCK_K : tl.constexpr):
#     return fbt_ptrs_auto(m_ptrs, k_ptrs, stride_k, 0, K, BLOCK_K)
