import triton
import triton.language as tl

# 16
@triton.jit
def dot_asym_ptr2nd_0(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr):
    # tl.static_print(f'dot_asym_ptr2nd_0 {lo=} {hi=}')
    tl.static_assert(hi - lo == BLOCK_K)
    # TODO: masked load
    b = tl.load(bm_ptrs + (bk_ptrs + lo) * stride_bk)
    acc = tl.dot(a, b, acc)
    return acc

# 32
@triton.jit
def dot_asym_ptr2nd_1(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr):
    # tl.static_print(f'dot_asym_ptr2nd_1 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    alo, ahi = a.split()
    acc = dot_asym_ptr2nd_0(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K)
    acc = dot_asym_ptr2nd_0(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K)
    return acc

# 64
@triton.jit
def dot_asym_ptr2nd_2(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr):
    # tl.static_print(f'dot_asym_ptr2nd_2 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    alo, ahi = a.split()
    acc = dot_asym_ptr2nd_1(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K)
    acc = dot_asym_ptr2nd_1(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K)
    return acc

# 128
@triton.jit
def dot_asym_ptr2nd_3(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr):
    # tl.static_print(f'dot_asym_ptr2nd_3 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    alo, ahi = a.split()
    acc = dot_asym_ptr2nd_2(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K)
    acc = dot_asym_ptr2nd_2(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K)
    return acc

# 256
@triton.jit
def dot_asym_ptr2nd_4(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr):
    # tl.static_print(f'dot_asym_ptr2nd_4 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    alo, ahi = a.split()
    acc = dot_asym_ptr2nd_3(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K)
    acc = dot_asym_ptr2nd_3(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K)
    return acc

@triton.jit
def dot_asym_ptr2nd_auto(acc,
                         a,
                         bm_ptrs, bk_ptrs,
                         stride_bk,
                         k_lo : tl.constexpr,
                         k_hi : tl.constexpr,
                         BLOCK_K : tl.constexpr,
                         ):
    N_BLOCKS : tl.constexpr = (k_hi - k_lo) // BLOCK_K
    if N_BLOCKS == 1:
        return dot_asym_ptr2nd_0(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 2:
        return dot_asym_ptr2nd_1(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 4:
        return dot_asym_ptr2nd_2(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 8:
        return dot_asym_ptr2nd_3(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K)
    elif N_BLOCKS == 16:
        return dot_asym_ptr2nd_4(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K)
    else:
        tl.static_assert(False, f'Unsupport N_BLOCKS {N_BLOCKS} from ({hi=} - {lo=})/{BLOCK_K=}')

'''
Full Binary Tree load

Load (M, K) matrix
'''
@triton.jit
def interleaved_dot_asym_ptr2nd(acc,
                                a,
                                bm_ptrs, bk_ptrs,
                                stride_bk,
                                K : tl.constexpr,
                                BLOCK_K : tl.constexpr):
    return dot_asym_ptr2nd_auto(acc, a, bm_ptrs, bk_ptrs, stride_bk, 0, K, BLOCK_K)