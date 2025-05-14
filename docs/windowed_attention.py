#!/usr/bin/env python

import numpy as np
import itertools
from dataclasses import dataclass

def cdiv(x: int, y: int):
    return (x + y - 1) // y

@dataclass
class FillerInfo:
    lb_lo             : int = 0
    lb_hi             : int = 0
    rb_lo             : int = 0
    rb_hi             : int = 0
    # leading_empty   : int = 0
    # leading_masked  : int = 0
    # solid           : int = 0
    # trailing_masked : int = 0
    # trailing_empty  : int = 0
    seqlen_q        : int = 0
    seqlen_k        : int = 0
    BLOCK_M         : int = 0
    BLOCK_N         : int = 0

def close_interval_isect(I_a, I_b):
    a_s, a_e = I_a
    b_s, b_e = I_b
    if b_s < a_e or a_s > b_e:
        return 0, -1
    o_s = max(a_s, b_s)
    o_e = max(a_e, b_e)
    return [o_s, o_e]

class Validator:
    def __init__(self, ref_mask, window_lef, window_right):
        self._ref_mask = ref_mask
        self._window_left = window_left
        self._window_right = window_right
        self._seqlen_q, self._seqlen_k = self._ref_mask.shape

    def validate(self, BLOCK_M, BLOCK_N):
        mask = np.full_like(self._ref_mask, 2)
        for start_m in range(0, self._seqlen_q, BLOCK_M):
            filler_info = self.create_filler(start_m, BLOCK_M, BLOCK_N)
            mask[start_m:start_m+BLOCK_M] = self.line_fill(start_m, filler_info)

    def create_filler(self, start_m, BLOCK_M, BLOCK_N):
        # Main diagnoal: M = N
        # Window Left line: M - window_left = N
        # Window Right line: M = N - window_right
        #
        # Intersection point (N axis) b/w window left line and M = start_m
        # start_m - window_left = N
        isec_lo = start_m - window_left
        # Intersection point (N axis) b/w window left line and M = start_m + BLOCK_M
        # M = + window_left + N = start_m + BLOCK_M
        isec_hi = start_m + BLOCK_M - window_left
        lsec_lob = cdiv(isec_lo, BLOCK_N)
        lsec_hib = cdiv(isec_hi, BLOCK_N)

        # Intersection point (N axis) b/w window left line and M = start_m
        # M = N - window_right = start_m
        isec_lo = start_m + window_right
        # Intersection point (N axis) b/w window left line and M = start_m + BLOCK_M
        # M = N - window_right = start_m + BLOCK_M
        isec_hi = start_m + BLOCK_M + window_right
        rsec_lob = cdiv(isec_lo, BLOCK_N)
        rsec_hib = cdiv(isec_hi, BLOCK_N)
        # Leading Empty blocks      : [-inf, lsec_lob)
        # Leading Masked Blocks     : [lsec_lob, lsec_hib]
        # Solid Blocks              : (lsec_hib, rsec_lob) = [lsec_hib + 1, rsec_lob - 1]
        # Trailing Masked Blocks    : [rsec_lob, rsec_hib]
        # Trailing Empty Blocks     : [rsec_hib, +inf)
        #
        # These blocks should be intersected with [0, cdiv(seqlen_k, BLOCK_N)]
        # https://scicomp.stackexchange.com/a/26260

        valid_bs = [0, cdiv(seqlen_k, BLOCK_N)]
        lb_lo, lb_hi = close_interval_isect([lsec_lob, lsec_hib], valid_bs)
        rb_lo, rb_hi = close_interval_isect([rsec_lob, rsec_hib], valid_bs)
        return FillerInfo(lb_lo=lb_lo,
                          lb_hi=lb_hi,
                          rb_lo=rb_lo,
                          rb_hi=rb_hi,
                          seqlen_q=self._seqlen_q,
                          seqlen_k=self._seqlen_k,
                          BLOCK_M=BLOCK_M,
                          BLOCK_N=BLOCK_N)
        # return FillerInfo(leading_empty=leading_empty,
        #                   leading_masked=leading_masked,
        #                   solid=solid,
        #                   trailing_masked=trailing_empty,
        #                   trailing_empty=trailing_masked,
        #                   seqlen_q=self._seqlen_q,
        #                   seqlen_k=self._seqlen_k,
        #                   BLOCK_M=BLOCK_M,
        #                   BLOCK_N=BLOCK_N)

    def line_fill(self, start_m, info):
        BLOCK_M = info.BLOCK_M
        BLOCK_N = info.BLOCK_N
        seqlen_k_round = cdiv(seqlen_k, BLOCK_N) * BLOCK_N
        mask = np.full_like((BLOCK_M, seqlen_k_round), 0, dtype=np.int8)
        start_n = 0
        # # Leading Empty
        # for n in range(start_n, start_n + info.leading_empty * BLOCK_N, BLOCK_N):
        #     mask[:, start_n:start_n+BLOCK_N] = 0
        start_n += info.leading_empty * BLOCK_N
        # Leading Masked
        for n in range(info.lb_lo, info.lb_hi+1):
            mask[:, n*BLOCK_N:n*BLOCK_N+BLOCK_N] = self.masked_block(info, start_m, start_n) # TODO: Partial Mask
        start_n += info.leading_masked * BLOCK_N
        # Solid Blocks
        for n in range(info.lb_hi+1, info.rb_lo):
            mask[:, n*BLOCK_N:n*BLOCK_N+BLOCK_N] = 1
        start_n += info.solid * BLOCK_N
        # Trailing Masked
        for n in range(info.rb_lo, info.rb_hi+1):
            mask[:, n*BLOCK_N:n*BLOCK_N+BLOCK_N] = self.masked_block(info, start_m, start_n) # TODO: Partial mask
        start_n += info.trailing_masked * BLOCK_N
        # # Trailing Empty
        # for n in range(start_n, start_n + info.trailing_empty * BLOCK_N, BLOCK_N):
        #     mask[:, start_n:start_n+BLOCK_N] = 0
        start_n += info.trailing_empty * BLOCK_N
        m = min(start_m + BLOCK_M, info.seqlen_q) - start_m
        n = info.seqlen_k
        return mask[:m, :n]

    # Note window left and window right lines may be in the same block
    def masked_block(self, info, start_m, start_n):
        BLOCK_M = info.BLOCK_M
        BLOCK_N = info.BLOCK_N
        # Window Left line: M - window_left = N
        # Window Right line: M = N - window_right
        # However, use things like
        #     (M[:, None] <= N[None, :]).astype(np.int8)
        # is much easier than calculating the line intersections
        MS = np.arange(start_m, start_m+BLOCK_M)
        NS = np.arange(start_n, start_n+BLOCK_N)
        def tril_mask():
            return MS[:, None] <= NS[None, :] + k
        def triu_mask(k):
            return MS + k <= NS

        mask = tril_mask(info.window_right) & triu_mask(-info.window_left)

        return None

def pt(t):
    print('Array:', np.array2string(t, prefix='Array: '))

print('Demo np.tril and triu')
ones = np.ones((8, 16), dtype=np.int8)
pt(ones)
window_left = 1
window_right = 2

win = np.tril(ones, window_right)
pt(win)
win = np.triu(win, -window_left)
pt(win)

def create_window_mask(M, N, window_left, window_right):
    ones = np.ones((M, N), dtype=np.int8)
    tmp = np.tril(ones, window_right)
    return np.triu(tmp, -window_left)

print('Simulate bottom right aligned causal with windowed attention')
pt(create_window_mask(8, 16, -8, 16))

X = np.arange(0, 8)
Y = np.arange(0, 16)
mask = (X[:, None] <= Y[None, :]).astype(np.int8)

pt(mask)

print('Simulate bottom right aligned causal with simulated windowed attention with X[:, None] <= Y[None, :]')

def sim_tril(m, k):
    X = np.arange(0, m.shape[0])
    Y = np.arange(0, m.shape[1]) + k
    return np.where(X[:, None] <= Y[None, :], m, 0)

def sim_triu(m, k):
    X = np.arange(0, m.shape[0]) + k
    Y = np.arange(0, m.shape[1])
    return np.where(X[:, None] <= Y[None, :], m, 0)

def create_window_mask_sim(M, N, window_left, window_right):
    ones = np.ones((M, N), dtype=np.int8)
    tmp = sim_tril(ones, window_right)
    return sim_triu(tmp, -window_left)
    # return sim_triu(ones, -window_left)

pt(create_window_mask_sim(8, 16, -8, 16))

exit()

print('Test calculation of leading_masked and trailing_masked')

REGULAR_SEQLEN_Q = [ 2 ** i for i in range(2, 15) ]
REGULAR_SEQLEN_K = [ 2 ** i for i in range(2, 15) ]

PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919, 10601]
PRIME_SEQLEN_K = [13, 31, 41, 71, 223, 337, 571, 1063, 2081, 5237, 11369]

Qs = sorted(REGULAR_SEQLEN_Q + PRIME_SEQLEN_Q)
Ks = sorted(REGULAR_SEQLEN_K + PRIME_SEQLEN_K)
BMS = [16,32,64,128]
BNS = [16,32,64,128]

for seqlen_q, seqlen_k in itertools.product(Qs, Ks):
    # print(f'{seqlen_q=} {seqlen_k=}')
    for window_left, window_right in itertools.product(range(seqlen_q), range(seqlen_k)):
        ref_mask = create_window_mask(seqlen_q, seqlen_k, window_left, window_right)
        # print(f'{window_left=} {window_right=}')
        # pt(mask)
        validator = Validator(ref_mask, window_left, window_right)
        for BLOCK_M, BLOCK_N in itertools.product(BMS, BNS):
            validator.validate(BLOCK_M, BLOCK_N)
