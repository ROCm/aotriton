#!/usr/bin/env python

import numpy as np
import itertools
from dataclasses import dataclass
import pytest

np.set_printoptions(linewidth=300)

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def fdiv(x: int, y: int):
    return x // y

@dataclass
class FillerInfo:
    lb_lo             : int = 0
    lb_hi             : int = 0
    fb_lo             : int = 0
    fb_hi             : int = 0
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
    if a_s > a_e or b_s > a_e or a_s > b_e:
        # print(f'Isect {I_a=} {I_b=} = Empty Set')
        return 0, -1
    o_s = max(a_s, b_s)
    o_e = min(a_e, b_e)
    # print(f'Isect {I_a=} {I_b=} = [{o_s}, {o_e}]')
    return [o_s, o_e]

class Validator:
    def __init__(self, ref_mask, window_left, window_right):
        self._ref_mask = ref_mask
        self._window_left = window_left
        self._window_right = window_right
        self._seqlen_q, self._seqlen_k = self._ref_mask.shape

    def validate(self, BLOCK_M, BLOCK_N, *, verbose=False):
        # print(f'window_left = {self._window_left} window_right = {self._window_right} {BLOCK_M=} {BLOCK_N=}')
        mask = np.full_like(self._ref_mask, 2)
        for start_m in range(0, self._seqlen_q, BLOCK_M):
            filler_info = self.create_filler(start_m, BLOCK_M, BLOCK_N, verbose=verbose)
            # print(f'{start_m=} {filler_info=}')
            mask[start_m:start_m+BLOCK_M] = self.line_fill(start_m, filler_info, verbose=verbose)
        # pt(mask)
        is_allclose = np.allclose(mask, self._ref_mask)
        if not is_allclose:
            for start_m in range(0, self._seqlen_q, BLOCK_M):
                filler_info = self.create_filler(start_m, BLOCK_M, BLOCK_N, verbose=True)
                print(f'{start_m=} {filler_info=}')
                mask[start_m:min(start_m+BLOCK_M, self._seqlen_q)] = self.line_fill(start_m, filler_info, verbose=True)
                pt(mask[start_m:min(start_m+BLOCK_M, self._seqlen_q)], pfx=f'Mask at {start_m:05}: ')
            assert False, f"seqlen_q={self._seqlen_q} seqlen_k={self._seqlen_k} window_left = {self._window_left} window_right = {self._window_right} {BLOCK_M=} {BLOCK_N=}\nRefMask: {np.array2string(self._ref_mask, prefix='RefMask: ')}\nOutMask: {np.array2string(mask, prefix='OutMask: ')}"

    def create_filler(self, start_m, BLOCK_M, BLOCK_N, *, verbose=False):
        seqlen_q = self._seqlen_q
        seqlen_k = self._seqlen_k
        window_left = self._window_left
        window_right = self._window_right
        # Main diagnoal: M = N
        # Window Left line: M - window_left = N
        # Window Right line: M = N - window_right
        #
        # Intersection point (N axis) b/w window left line and M = start_m
        # start_m - window_left = N
        isec_lo = start_m - window_left
        # Intersection point (N axis) b/w window left line and M = min(start_m + BLOCK_M, seqlen_q)
        # M = + window_left + N = min(start_m + BLOCK_M, seqlen_q)
        isec_hi = min(start_m + BLOCK_M, seqlen_q) - window_left
        lsec_lob = fdiv(isec_lo, BLOCK_N)
        lsec_hib = fdiv(isec_hi, BLOCK_N)
        if verbose:
            print(f'{window_left=} {isec_lo=} {isec_hi=} {lsec_lob=} {lsec_hib=}')

        # Intersection point (N axis) b/w window left line and M = start_m
        # M = N - window_right = start_m
        isec_lo = start_m + window_right
        # Intersection point (N axis) b/w window left line and M = min(start_m + BLOCK_M, seqlen_q)
        # M = N - window_right = min(start_m + BLOCK_M, seqlen_q)
        isec_hi = min(start_m + BLOCK_M, seqlen_q) + window_right
        rsec_lob = fdiv(isec_lo, BLOCK_N)
        rsec_hib = fdiv(isec_hi, BLOCK_N)
        if verbose:
            print(f'{window_right=} {isec_lo=} {isec_hi=} {rsec_lob=} {rsec_hib=}')
        # Leading Empty blocks      : [-inf, lsec_lob)
        # Leading Masked Blocks     : [lsec_lob, lsec_hib]
        # Solid Blocks              : (lsec_hib, rsec_lob) = [lsec_hib + 1, rsec_lob - 1]
        # Trailing Masked Blocks    : [rsec_lob, rsec_hib]
        # Trailing Empty Blocks     : [rsec_hib, +inf)
        #
        # These blocks should be intersected with [0, fdiv(seqlen_k, BLOCK_N)]
        # https://scicomp.stackexchange.com/a/26260

        valid_bs = [0, cdiv(self._seqlen_k, BLOCK_N) -1]  # Closing interval
        lb_lo, lb_hi = close_interval_isect([lsec_lob, lsec_hib], valid_bs)
        rb_lo, rb_hi = close_interval_isect([rsec_lob, rsec_hib], valid_bs)
        if verbose:
            print(f'{lsec_lob=} {lsec_hib=} isect {valid_bs=} -> {lb_lo=} {lb_hi=}')
            print(f'{rsec_lob=} {rsec_hib=} isect {valid_bs=} -> {rb_lo=} {rb_hi=}')
        # calc isec b/w lb and rb
        ub_lo, ub_hi = close_interval_isect([lb_lo, lb_hi], [rb_lo, rb_hi])
        if ub_lo > ub_hi:
            fb_lo, fb_hi = close_interval_isect([lsec_hib+1, rsec_lob-1], valid_bs)
            if verbose:
                print(f'{lsec_hib+1=} {rsec_lob-1=} isect {valid_bs=} -> {fb_lo=} {fb_hi=}')
            # lb not intersecting with rb
            return FillerInfo(lb_lo=lb_lo,
                              lb_hi=lb_hi,
                              fb_lo=fb_lo,
                              fb_hi=fb_hi,
                              rb_lo=rb_lo,
                              rb_hi=rb_hi,
                              seqlen_q=self._seqlen_q,
                              seqlen_k=self._seqlen_k,
                              BLOCK_M=BLOCK_M,
                              BLOCK_N=BLOCK_N)
        else:
            # lb intersecting with rb
            # Unify the processing
            return FillerInfo(lb_lo=lb_lo,
                              lb_hi=rb_hi,
                              fb_lo=-2,
                              fb_hi=-3,
                              rb_lo=-1,
                              rb_hi=-2,
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

    def line_fill(self, start_m, info, *, verbose=False):
        BLOCK_M = info.BLOCK_M
        BLOCK_N = info.BLOCK_N
        seqlen_k_round = cdiv(self._seqlen_k, BLOCK_N) * BLOCK_N
        mask = np.zeros((BLOCK_M, seqlen_k_round), dtype=np.int8)
        # Leading Masked
        for n in range(info.lb_lo, info.lb_hi+1):
            blk = self.masked_block(info, start_m, n*BLOCK_N)
            if verbose:
                pt(blk, f' Block {n*BLOCK_N:05}')
                pt(self._ref_mask[start_m:start_m+BLOCK_M, n*BLOCK_N:n*BLOCK_N+BLOCK_N], f'RefBlk {n*BLOCK_N:05}')
            mask[:, n*BLOCK_N:n*BLOCK_N+BLOCK_N] = blk
            if verbose:
                pt(mask[start_m:start_m+BLOCK_M, n*BLOCK_N:n*BLOCK_N+BLOCK_N], f'Updated Block {n*BLOCK_N:05}')
        # Solid Blocks
        for n in range(info.fb_lo, info.fb_hi+1):
            if verbose:
                print(f'Solid Block {n*BLOCK_N:05}')
            mask[:, n*BLOCK_N:n*BLOCK_N+BLOCK_N] = 1
        # Trailing Masked
        for n in range(info.rb_lo, info.rb_hi+1):
            blk = self.masked_block(info, start_m, n*BLOCK_N)
            if verbose:
                pt(blk, f' Block {n*BLOCK_N:05}')
                pt(self._ref_mask[start_m:start_m+BLOCK_M, n*BLOCK_N:n*BLOCK_N+BLOCK_N], f'RefBlk {n*BLOCK_N:05}')
            mask[:, n*BLOCK_N:n*BLOCK_N+BLOCK_N] = blk
        m = min(BLOCK_M, info.seqlen_q - start_m)
        n = info.seqlen_k
        if verbose:
            print(f'Return: {m=} {n=} {mask.shape=}')
            pt(mask, '  Full Return: ')
            pt(mask[:m, :n], 'Actual Return: ')
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
        def tril_mask(k):
            return MS[:, None] + k >= NS[None, :]
        def triu_mask(k):
            return MS[:, None] + k <= NS[None, :]

        mask = (tril_mask(self._window_right) & triu_mask(-self._window_left)).astype(np.int8)

        return mask

def pt(t, pfx='Array:'):
    print(pfx, np.array2string(t, prefix=pfx), sep='')

def create_window_mask(M, N, window_left, window_right):
    ones = np.ones((M, N), dtype=np.int8)
    tmp = np.tril(ones, window_right)
    return np.triu(tmp, -window_left)

def sim_tril(m, k):
    X = np.arange(0, m.shape[0]) + k
    Y = np.arange(0, m.shape[1])
    return np.where(X[:, None] >= Y[None, :], m, 0)

def sim_triu(m, k):
    X = np.arange(0, m.shape[0]) + k
    Y = np.arange(0, m.shape[1])
    return np.where(X[:, None] <= Y[None, :], m, 0)

def create_window_mask_sim(M, N, window_left, window_right):
    ones = np.ones((M, N), dtype=np.int8)
    tmp = sim_tril(ones, window_right)
    return sim_triu(tmp, -window_left)
    # return sim_triu(ones, -window_left)

def prelude():
    print('Demo np.tril and triu')
    ones = np.ones((8, 16), dtype=np.int8)
    pt(ones)
    window_left = 1
    window_right = 2

    win = np.tril(ones, window_right)
    pt(win)
    win = np.triu(win, -window_left)
    pt(win)

    print('Simulate bottom right aligned causal with windowed attention')
    pt(create_window_mask(8, 16, -8, 16))

    X = np.arange(0, 8)
    Y = np.arange(0, 16)
    mask = (X[:, None] <= Y[None, :]).astype(np.int8)

    pt(mask)

    print(f'Validating sim_tril')
    ones = np.ones((5, 7), dtype=np.int8)
    for i in range(-5, 7):
        s = sim_tril(ones, i)
        r = np.tril(ones, i)
        if not np.allclose(s, r):
            assert np.allclose(s, r)
            print(f'Validate sim_tril(k={i})')
            pt(s)
            pt(r)
    print(f'sim_tril Validated')

    print(f'Validating sim_triu')
    for i in range(-5, 7):
        s = sim_triu(ones, i)
        r = np.triu(ones, i)
        if not np.allclose(s, r):
            print(f'Validate sim_triu(k={i})')
            pt(s)
            pt(r)
            assert False
    print(f'sim_triu Validated')

    print('Simulate bottom right aligned causal with simulated windowed attention with X[:, None] <=/>= Y[None, :]')

    pt(create_window_mask_sim(8, 16, -8, 16))
    print('create_window_mask(5, 7, 1, 0)')
    pt(create_window_mask(5, 7, 1, 0))
    print('create_window_mask_sim(5, 7, 1, 0)')
    pt(create_window_mask_sim(5, 7, 1, 0))

REGULAR_SEQLEN_Q = [ 2 ** i for i in range(2, 15) ]
REGULAR_SEQLEN_K = [ 2 ** i for i in range(2, 15) ]

PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919, 10601]
PRIME_SEQLEN_K = [13, 31, 41, 71, 223, 337, 571, 1063, 2081, 5237, 11369]
Qs = sorted(REGULAR_SEQLEN_Q + PRIME_SEQLEN_Q)
Ks = sorted(REGULAR_SEQLEN_K + PRIME_SEQLEN_K)
BMS = [16,32,64,128]
BNS = [16,32,64,128]

# if False:
#     BMS = [16]
#     BNS = [16]
#     _validate_qkmask(5, 7, BMS, BNS)
#     exit()

def _validate_qkmask(seqlen_q, seqlen_k, BLOCK_M, BLOCK_N):
    for window_left, window_right in itertools.product(range(seqlen_q), range(seqlen_k)):
        ref_mask = create_window_mask(seqlen_q, seqlen_k, window_left, window_right)
        # print(f'Validating {window_left=} {window_right=}')
        # pt(ref_mask)
        validator = Validator(ref_mask, window_left, window_right)
        validator.validate(BLOCK_M, BLOCK_N)

@pytest.mark.parametrize("Q", Qs)
@pytest.mark.parametrize("K", Ks)
@pytest.mark.parametrize("BLOCK_M", BMS)
@pytest.mark.parametrize("BLOCK_N", BNS)
def test_qkmask(Q, K, BLOCK_M, BLOCK_N):
    _validate_qkmask(Q, K, BLOCK_M, BLOCK_N)

def _validate_negative_window(seqlen_q, seqlen_k, BLOCK_M, BLOCK_N):
    for window_left in range(seqlen_q):
        for window_right in range(0, window_left):
            ref_mask = create_window_mask(seqlen_q, seqlen_k, window_left, -window_right)
            validator = Validator(ref_mask, window_left, -window_right)
            validator.validate(BLOCK_M, BLOCK_N)
    for window_right in range(seqlen_k):
        for window_left in range(0, window_right):
            ref_mask = create_window_mask(seqlen_q, seqlen_k, -window_left, window_right)
            validator = Validator(ref_mask, -window_left, window_right)
            validator.validate(BLOCK_M, BLOCK_N)

@pytest.mark.parametrize("Q", Qs)
@pytest.mark.parametrize("K", Ks)
@pytest.mark.parametrize("BLOCK_M", BMS)
@pytest.mark.parametrize("BLOCK_N", BNS)
def test_negative_window(Q, K, BLOCK_M, BLOCK_N):
    _validate_negative_window(Q, K, BLOCK_M, BLOCK_N)

def main():
    Q = 143
    K = 4
    BLOCK_M = 128
    BLOCK_N = 64
    window_left = Q
    window_right = 0
    ref_mask = create_window_mask(Q, K, window_left, window_right)
    validator = Validator(ref_mask, window_left, window_right)
    validator.validate(BLOCK_M, BLOCK_N, verbose=True)

if __name__ == '__main__':
    prelude()
    main()
