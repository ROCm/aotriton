#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""JIT-level gate for the fast_philox column layout.

The contract under test is the one stated in `dropout.py`: the four randoms
`tl.randint4x` draws from one Philox offset occupy that offset's four
consecutive key columns in generation order -- for offset g, column
4*g+i holds r<i>.

**The oracle is an independent host-side Philox, and it has to be.**  Every
dropout test in the tree takes its reference mask from the kernel under test
(`dropout_mask = encoded_softmax >= 0`), so all of them are blind to a
permutation of the four columns: the mask's distribution, its per-offset keep
count and every statistical property are identical under any permutation, and
only *which* column holds which random moves.  Nothing derived from the kernel
can see that.  A direct check against a separate derivation can.

`debug_fill_dropout_rng` is what makes the direct check possible: with an int32
destination it stores the raw PRNG words instead of a thresholded mask, so the
comparison is exact integer equality with no tolerance anywhere in it.

Independence, precisely.  `_philox_4x32_10` below is Philox-4x32-10 (Salmon et
al., 2011) written out from the algorithm; it is not a call into
`triton.language.random`.  What it does have to match is Triton's counter/key
wiring at the `tl.randint4x` boundary -- offset low word into c0, offset high
word into c1, c2 = c3 = 0, key = (seed_lo, seed_hi) -- because that wiring is
the interface `fast_philox` calls *through*, not the thing being checked.  The
subject of this gate is the join/reshape lane layout downstream of it.

The gate does not merely assert the expected layout.  It asserts that *no
other* permutation of the four host-computed randoms reproduces the kernel's
output.  That makes it self-validating in the direction that matters: if the
host Philox were wrong, all 24 candidate layouts would fail and the first
assertion would say the two derivations disagree, rather than the gate quietly
agreeing with a broken oracle.

Coverage is every offset group of the whole plane, not a sampled few.  One
group cannot tell a within-group permutation from a global one, and building
the entire expected plane from per-group quartets rejects both.

Needs a GPU, and nothing else: no built library, no reference framework.  It
imports the bare kernel from `flash` rather than the launcher in
`attn_torch_function`, which asserts `TRITON_F32_DEFAULT` and a live device at
import time and so cannot be skipped cleanly on a host without one.
"""

import itertools

import numpy as np
import pytest
import torch
import triton

from dropout import PHILOX_RN_PER_OFFSET
from flash import debug_fill_dropout_rng as bare_debug_fill_dropout_rng

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='JIT-level gate: launches a Triton kernel, needs a GPU')

# This gate is written for the 4-randoms-per-offset layout.  fast_philox's
# `if False:` branch lays 8 out per offset and has never been reachable; if it
# ever becomes reachable the gate must be rewritten, not skipped, so the
# mismatch is asserted rather than tolerated.
RN_PER_OFFSET = 4

_M32 = 0xFFFFFFFF
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57


def _philox_4x32_10(seed, offsets):
    """Philox-4x32-10 over `offsets`, wired the way tl.randint4x wires it.

    Every constant is an explicit np.uint64.  On numpy < 2 a uint64 array
    combined with a Python int promotes through int64 to float64 and silently
    loses the low bits -- the one trap in writing this by hand.
    """
    u = np.uint64
    off = np.asarray(offsets, dtype=np.uint64)
    c0 = off & u(_M32)
    c1 = (off >> u(32)) & u(_M32)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    k0 = u(seed & _M32)
    k1 = u((seed >> 32) & _M32)
    A = u(_PHILOX_ROUND_A)
    B = u(_PHILOX_ROUND_B)
    for _ in range(10):
        prev_c0, prev_c2 = c0, c2
        # umulhi(x, y) == (x * y) >> 32.  Both factors are below 2**32, so the
        # product is exact in uint64 and nothing wraps.
        c0 = ((B * prev_c2) >> u(32)) ^ c1 ^ k0
        c2 = ((A * prev_c0) >> u(32)) ^ c3 ^ k1
        c1 = (B * prev_c2) & u(_M32)
        c3 = (A * prev_c0) & u(_M32)
        k0 = (k0 + u(_PHILOX_KEY_A)) & u(_M32)
        k1 = (k1 + u(_PHILOX_KEY_B)) & u(_M32)
    # int32 is what the kernel stores: a bitcast, not a saturating convert.
    return [c.astype(np.uint32).view(np.int32) for c in (c0, c1, c2, c3)]


def _rng_offsets(batch, heads, seqlen_q, seqlen_k, philox_offset_base):
    """The Philox offset each (z, h, m, group) quadruple draws from.

    debug_fill_dropout_rng walks the plane in (BLOCK_M, BLOCK_N) tiles, but the
    tiling cancels.  A tile's base carries
    `start_m * BLOCK_M * stride + start_n // RN_PER_OFFSET`, and inside the
    tile fast_philox adds `local_row * stride + local_group`; the two halves of
    each coordinate recombine into the global one, leaving

        offset(z, h, m, g) = base + (z*H + h) * seqlen_q * stride
                                  + m * stride + g

    with stride = cdiv(seqlen_k, RN_PER_OFFSET).  That the block shape drops
    out is itself part of the contract -- forward and backward tile the same
    plane with different BLOCK_N and must still agree element for element -- so
    this function deliberately takes no BLOCK_M/BLOCK_N argument, and the test
    parametrises over block shapes to confirm the output does not depend on
    them.
    """
    stride = -(-seqlen_k // RN_PER_OFFSET)
    zh = np.arange(batch * heads, dtype=np.uint64)
    m = np.arange(seqlen_q, dtype=np.uint64)
    g = np.arange(stride, dtype=np.uint64)
    off = (np.uint64(philox_offset_base)
           + zh[:, None, None] * np.uint64(seqlen_q * stride)
           + m[None, :, None] * np.uint64(stride)
           + g[None, None, :])
    return off.reshape(batch, heads, seqlen_q, stride)


def _fill_rng(r, philox_seed, philox_offset, BLOCK_M, BLOCK_N):
    """Launch debug_fill_dropout_rng exactly as attn_torch_function does."""
    BATCH, N_HEADS, seqlen_q, seqlen_k = r.size()
    grid = (triton.cdiv(seqlen_q, BLOCK_M), N_HEADS, BATCH)
    bare_debug_fill_dropout_rng[grid](r,
                                      r.stride(0), r.stride(1), r.stride(2), r.stride(3),
                                      seqlen_q, seqlen_k,
                                      philox_seed,
                                      philox_offset,
                                      BLOCK_M, BLOCK_N,
                                      num_stages=1)


# seqlen_q = 72 leaves a partial trailing BLOCK_M tile for either block shape.
_SHAPES = [
    (2, 3, 72, 64),   # seqlen_k is a whole number of BLOCK_N tiles
    (1, 2, 72, 36),   # 9 groups: a partial trailing BLOCK_N tile
]

# BLOCK_N // RN_PER_OFFSET is what fast_philox passes to tl.arange, so it has to
# be a power of two; 32 and 64 give 8 and 16.
_BLOCKS = [(64, 32), (32, 64)]

_SEEDS = [
    # attn_torch_function's defaults (DEFAULT_PHILOX_SEED, OFFSET_1 + OFFSET_2),
    # duplicated rather than imported so this module stays importable, and
    # therefore skippable, without a GPU.
    (0x1BF52, 0x1D4B42),
    # Both high words non-zero: exercises seed_hi in the key and offset_hi in
    # c1, which the defaults above leave at zero.
    (0xDEADBEEFCAFE1234, 0x1_0000_0007),
]


@pytest.mark.parametrize('philox_seed,philox_offset', _SEEDS)
@pytest.mark.parametrize('BLOCK_M,BLOCK_N', _BLOCKS)
@pytest.mark.parametrize('BATCH,N_HEADS,seqlen_q,seqlen_k', _SHAPES)
def test_fast_philox_column_layout(BATCH, N_HEADS, seqlen_q, seqlen_k,
                                   BLOCK_M, BLOCK_N, philox_seed, philox_offset):
    assert int(PHILOX_RN_PER_OFFSET.value) == RN_PER_OFFSET, (
        f'fast_philox now emits {int(PHILOX_RN_PER_OFFSET.value)} randoms per offset; '
        'this gate only describes the 4-random layout and must be rewritten')

    # Prefilled with a sentinel so an unwritten cell shows up as a mismatch
    # rather than as whatever the allocator left behind.
    r = torch.full((BATCH, N_HEADS, seqlen_q, seqlen_k), -1,
                   device='cuda', dtype=torch.int32)
    _fill_rng(r, philox_seed, philox_offset, BLOCK_M, BLOCK_N)
    torch.cuda.synchronize()
    observed = r.cpu().numpy()

    randoms = _philox_4x32_10(philox_seed,
                              _rng_offsets(BATCH, N_HEADS, seqlen_q, seqlen_k, philox_offset))

    def lay_out(perm):
        """The whole plane, assuming column 4*g+i holds randoms[perm[i]]."""
        ngroups = randoms[0].shape[-1]
        plane = np.empty((BATCH, N_HEADS, seqlen_q, ngroups * RN_PER_OFFSET),
                         dtype=np.int32)
        for i, src in enumerate(perm):
            plane[..., i::RN_PER_OFFSET] = randoms[src]
        return plane[..., :seqlen_k]

    identity = tuple(range(RN_PER_OFFSET))
    fitting = [perm for perm in itertools.permutations(range(RN_PER_OFFSET))
               if np.array_equal(observed, lay_out(perm))]

    # Distinguish "the oracle is broken" from "the layout is wrong".  If the
    # host Philox disagreed with Triton's for any reason other than column
    # order -- wrong counter wiring, wrong round count, a numpy promotion --
    # nothing would fit and the gate would be making no claim at all.
    assert fitting, (
        'no permutation of the host-computed randoms reproduces the kernel output. '
        'The two Philox derivations disagree about the values themselves, so this '
        'gate cannot say anything about the column layout.')

    assert fitting == [identity], (
        f'fast_philox lays the four randoms of one offset out as {fitting[0]}, '
        f'expected {identity}. The contract is stated in dropout.py: column 4*g+i '
        f'holds r<i>. (0, 2, 1, 3) specifically is what nested tl.join produces '
        f'from the natural (r0,r1),(r2,r3) pairing -- see the comment there before '
        f'"fixing" the pairing back.')
