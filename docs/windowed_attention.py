#!/usr/bin/env python

import numpy as np

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

print('Test calculation of leading_masked_blocks and trainling_masked_blocks')

REGULAR_SEQLEN_Q = [ 2 ** i for i in range(2, 15) ]
REGULAR_SEQLEN_K = [ 2 ** i for i in range(2, 15) ]

PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919, 10601]
PRIME_SEQLEN_K = [13, 31, 41, 71, 223, 337, 571, 1063, 2081, 5237, 11369]
