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
                      BLOCK_K : tl.constexpr,
                      STACKED_A : tl.constexpr,
                      ):
    # tl.static_print(f'dot_asym_ptr2nd_0 {lo=} {hi=}')
    tl.static_assert(hi - lo == BLOCK_K)
    # TODO: masked load
    b = tl.load(bm_ptrs + (bk_ptrs + lo) * stride_bk)
    # tl.static_print(f'dot_asym_ptr2nd_0 a.shape = {a.shape[0]} {a.shape[1]}')
    # tl.static_print(f'dot_asym_ptr2nd_0 b.shape = {b.shape[0]} {b.shape[1]}')
    # tl.static_print(f'dot_asym_ptr2nd_0 acc.shape = {acc.shape[0]} {acc.shape[1]}')
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
                      BLOCK_K : tl.constexpr,
                      STACKED_A : tl.constexpr,
                      ):
    # tl.static_print(f'dot_asym_ptr2nd_1 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    if STACKED_A:
        alo, ahi = a.split()
        acc = dot_asym_ptr2nd_0(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acc = dot_asym_ptr2nd_0(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    else:
        acclo, acchi = acc.split()
        acclo = dot_asym_ptr2nd_0(acclo, a, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acchi = dot_asym_ptr2nd_0(acchi, a, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    # tl.static_print(f'dot_asym_ptr2nd_1 ret acc.shape = {acc.shape[0]} {acc.shape[1]}')
    return acc

# 64
@triton.jit
def dot_asym_ptr2nd_2(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr,
                      STACKED_A : tl.constexpr,
                      ):
    # tl.static_print(f'dot_asym_ptr2nd_2 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    if STACKED_A:
        alo, ahi = a.split()
        acc = dot_asym_ptr2nd_1(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acc = dot_asym_ptr2nd_1(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    else:
        acclo, acchi = acc.split()
        acclo = dot_asym_ptr2nd_1(acclo, a, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acchi = dot_asym_ptr2nd_1(acchi, a, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    # tl.static_print(f'dot_asym_ptr2nd_2 ret acc.shape = {acc.shape[0]} {acc.shape[1]}')
    return acc

# 128
@triton.jit
def dot_asym_ptr2nd_3(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr,
                      STACKED_A : tl.constexpr,
                      ):
    # tl.static_print(f'dot_asym_ptr2nd_3 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    if STACKED_A:
        alo, ahi = a.split()
        acc = dot_asym_ptr2nd_2(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acc = dot_asym_ptr2nd_2(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    else:
        acclo, acchi = acc.split()
        acclo = dot_asym_ptr2nd_2(acclo, a, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acchi = dot_asym_ptr2nd_2(acchi, a, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    # tl.static_print(f'dot_asym_ptr2nd_3 ret acc.shape = {acc.shape[0]} {acc.shape[1]}')
    return acc

# 256
@triton.jit
def dot_asym_ptr2nd_4(acc,
                      a,
                      bm_ptrs, bk_ptrs,
                      stride_bk,
                      lo : tl.constexpr,
                      hi : tl.constexpr,
                      BLOCK_K : tl.constexpr,
                      STACKED_A : tl.constexpr,
                      ):
    # tl.static_print(f'dot_asym_ptr2nd_4 {lo=} {hi=}')
    tl.static_assert(hi - lo >= BLOCK_K)
    half : tl.constexpr = (hi - lo) // 2
    if STACKED_A:
        alo, ahi = a.split()
        acc = dot_asym_ptr2nd_3(acc, alo, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acc = dot_asym_ptr2nd_3(acc, ahi, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    else:
        acclo, acchi = acc.split()
        acclo = dot_asym_ptr2nd_3(acchi, a, bm_ptrs, bk_ptrs, stride_bk, lo, lo + half, BLOCK_K, STACKED_A)
        acchi = dot_asym_ptr2nd_3(acclo, a, bm_ptrs, bk_ptrs, stride_bk, lo + half, hi, BLOCK_K, STACKED_A)
    return acc

@triton.jit
def dot_asym_ptr2nd_block0(acc,
                           a_full,
                           bm_ptrs, bk_ptrs,
                           stride_bk,
                           M : tl.constexpr,
                           k_lo : tl.constexpr,
                           k_hi : tl.constexpr,
                           BLOCK_K : tl.constexpr,
                           STACKED_A : tl.constexpr,
                           ):
    a = a_full
    return dot_asym_ptr2nd_0(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, False)

@triton.jit
def dot_asym_ptr2nd_block1(acc,
                           a_full,
                           bm_ptrs, bk_ptrs,
                           stride_bk,
                           M : tl.constexpr,
                           k_lo : tl.constexpr,
                           k_hi : tl.constexpr,
                           BLOCK_K : tl.constexpr,
                           STACKED_A : tl.constexpr,
                           ):
    # tl.static_print(f'M = {M}')
    # tl.static_print(f'BLOCK_K = {BLOCK_K}')
    if STACKED_A:
        a = a_full.reshape(M, 2, BLOCK_K).trans(0, 2, 1)
        return dot_asym_ptr2nd_1(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
    else:
        acc_stack = acc.reshape(M, 2, BLOCK_K).trans(0, 2, 1)
        dot_asym_ptr2nd_1(acc_stack, a_full, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
        return acc

@triton.jit
def dot_asym_ptr2nd_block2(acc,
                           a_full,
                           bm_ptrs, bk_ptrs,
                           stride_bk,
                           M : tl.constexpr,
                           k_lo : tl.constexpr,
                           k_hi : tl.constexpr,
                           BLOCK_K : tl.constexpr,
                           STACKED_A : tl.constexpr,
                           ):
    if STACKED_A:
        a = a_full.reshape(M, 2, 2, BLOCK_K).trans(0, 3, 2, 1)
        return dot_asym_ptr2nd_2(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
    else:
        acc_stack = acc.reshape(M, 2, 2, BLOCK_K).trans(0, 3, 2, 1)
        dot_asym_ptr2nd_2(acc_stack, a_full, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
        return acc

@triton.jit
def dot_asym_ptr2nd_block3(acc,
                           a_full,
                           bm_ptrs, bk_ptrs,
                           stride_bk,
                           M : tl.constexpr,
                           k_lo : tl.constexpr,
                           k_hi : tl.constexpr,
                           BLOCK_K : tl.constexpr,
                           STACKED_A : tl.constexpr,
                           ):
    if STACKED_A:
        a = a_full.reshape(M, 2, 2, 2, BLOCK_K).trans(0, 4, 3, 2, 1)
        return dot_asym_ptr2nd_3(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
    else:
        acc_stack = acc.reshape(M, 2, 2, 2, BLOCK_K).trans(0, 4, 3, 2, 1)
        dot_asym_ptr2nd_3(acc_stack, a_full, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
        return acc

@triton.jit
def dot_asym_ptr2nd_block4(acc,
                           a_full,
                           bm_ptrs, bk_ptrs,
                           stride_bk,
                           M : tl.constexpr,
                           k_lo : tl.constexpr,
                           k_hi : tl.constexpr,
                           BLOCK_K : tl.constexpr,
                           STACKED_A : tl.constexpr,
                           ):
    if STACKED_A:
        a = a_full.reshape(M, 2, 2, 2, 2, BLOCK_K).trans(0, 5, 4, 3, 2, 1)
        return dot_asym_ptr2nd_4(acc, a, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
    else:
        acc_stack = acc.reshape(M, 2, 2, 2, 2, BLOCK_K).trans(0, 5, 4, 3, 2, 1)
        dot_asym_ptr2nd_4(acc_stack, a_full, bm_ptrs, bk_ptrs, stride_bk, k_lo, k_hi, BLOCK_K, STACKED_A)
        return acc

@triton.jit
def dot_asym_ptr2nd_auto(acc,
                         a,
                         bm_ptrs, bk_ptrs,
                         stride_bk,
                         M : tl.constexpr,
                         k_lo : tl.constexpr,
                         k_hi : tl.constexpr,
                         BLOCK_K : tl.constexpr,
                         STACKED_A : tl.constexpr,
                         ):
    N_BLOCKS : tl.constexpr = (k_hi - k_lo) // BLOCK_K
    if N_BLOCKS == 1:
        return dot_asym_ptr2nd_block0(acc, a, bm_ptrs, bk_ptrs, stride_bk, M, k_lo, k_hi, BLOCK_K, STACKED_A)
    elif N_BLOCKS == 2:
        return dot_asym_ptr2nd_block1(acc, a, bm_ptrs, bk_ptrs, stride_bk, M, k_lo, k_hi, BLOCK_K, STACKED_A)
    elif N_BLOCKS == 4:
        return dot_asym_ptr2nd_block2(acc, a, bm_ptrs, bk_ptrs, stride_bk, M, k_lo, k_hi, BLOCK_K, STACKED_A)
    elif N_BLOCKS == 8:
        return dot_asym_ptr2nd_block3(acc, a, bm_ptrs, bk_ptrs, stride_bk, M, k_lo, k_hi, BLOCK_K, STACKED_A)
    elif N_BLOCKS == 16:
        return dot_asym_ptr2nd_block4(acc, a, bm_ptrs, bk_ptrs, stride_bk, M, k_lo, k_hi, BLOCK_K, STACKED_A)
    else:
        tl.static_assert(False, f'Unsupport N_BLOCKS {N_BLOCKS} from ({hi=} - {lo=})/{BLOCK_K=}')

@triton.jit
def interleaved_dot_asym_ptr2nd(acc,
                                a,
                                bm_ptrs, bk_ptrs,
                                stride_bk,
                                M : tl.constexpr,
                                K : tl.constexpr,
                                BLOCK_K : tl.constexpr,
                                STACKED_A : tl.constexpr,
                                ):
    # tl.static_print(f'M = {M}')
    return dot_asym_ptr2nd_auto(acc, a, bm_ptrs, bk_ptrs, stride_bk, M, 0, K, BLOCK_K, STACKED_A)
