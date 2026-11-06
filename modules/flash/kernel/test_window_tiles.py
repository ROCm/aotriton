#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generalized sliding window attention (CAUSAL_TYPE = 3) at pinned tiles.

**The defect these gate.** A forward tile whose rows all see no key takes the
early exit in fwd_kernel.py and stores zeros. That store was bounded by
`o_cols=BLOCK_DMODEL` instead of `Hdim_vo`. BLOCK_DMODEL is the padded
register width, so every row wrote BLOCK_DMODEL - Hdim_vo zeros past its end;
Out's rows are not padded, so those zeros landed on the following rows. With
Hdim_vo = 8 and BLOCK_DMODEL = 128 (hdim (120, 8)) that is 15 rows of the same
head under a (B, H, S, D)-contiguous Out. When a dead tile sits right above a
live one -- a numeric window with a very negative R, or L larger than
seqlen_q, does exactly that -- the live rows were zeroed or not depending on
which workgroup stored last. Only head dims with BLOCK_DMODEL > Hdim_vo were
affected, which is why it looked like a tuned-tile (block interval) problem.

Three gates:

1. `test_interval_probe`: parse_window + calculate_intervals evaluated on the
   GPU and checked against the spec, block by block, for the row-major (fwd,
   dq) and transposed (dk/dv) uses, several tiles, p in {0, 1}, R very
   negative, L > seqlen_q, and bands that partially or fully miss
   [0, seqlen_k). For sentinel windows it also gates against visiting a block
   no row needs.
2. `test_out_row_padding`: Out allocated with NaN padding after every row. The
   forward must leave it untouched. Deterministic, unlike a value compare:
   nothing else writes the padding, so no race can hide a stray store.
3. `test_window_fwd_bwd` / `test_window_varlen`: values against the float32
   reference, forward and backward, split and fused.

Run with `TRITON_F32_DEFAULT=ieee`; `attn_torch_function` asserts it at import.
"""

import math

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from _common_test import SdpaContext, SdpaParams, VarlenSdpaContext
from attn_torch_function import attention, AttentionExtraArgs, PersistentType
from masked_load_store import parse_window, calculate_intervals
from varlen_attn_torch_function import varlen_attention

WINDOW_TOP_LEFT_ALIGNED = -2147483647
WINDOW_BOTTOM_RIGHT_ALIGNED = -2147483646
SENTINELS = (WINDOW_TOP_LEFT_ALIGNED, WINDOW_BOTTOM_RIGHT_ALIGNED)

# ---------------------------------------------------------------------------
# 1. Block intervals
# ---------------------------------------------------------------------------

@triton.jit
def _interval_probe(Params,
                    Out,
                    MAX_TILES: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    TRANSPOSED: tl.constexpr):
    """One program per (case, tile). Params rows: L, R, varlen_bits, sq, sk.

    TRANSPOSED reproduces how bwd_kernel_dk_dv / bwd_kernel_fuse call
    calculate_intervals: K-major, windows swapped, block sizes swapped.
    """
    tile = tl.program_id(0)
    case = tl.program_id(1)
    window_left_in = tl.load(Params + case * 5 + 0)
    window_right_in = tl.load(Params + case * 5 + 1)
    varlen_bits = tl.load(Params + case * 5 + 2)
    seqlen_q = tl.load(Params + case * 5 + 3)
    seqlen_k = tl.load(Params + case * 5 + 4)
    window_left, window_right = parse_window(True,
                                             3,
                                             window_left_in,
                                             window_right_in,
                                             seqlen_q,
                                             seqlen_k,
                                             varlen_bits)
    if TRANSPOSED:
        start = tile * BLOCK_N
        lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi = \
                calculate_intervals(True,
                                    3,
                                    window_right,
                                    window_left,
                                    start,
                                    seqlen_k,
                                    seqlen_q,
                                    start + BLOCK_N > seqlen_k,
                                    BLOCK_N,
                                    BLOCK_M)
    else:
        start = tile * BLOCK_M
        lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi = \
                calculate_intervals(True,
                                    3,
                                    window_left,
                                    window_right,
                                    start,
                                    seqlen_q,
                                    seqlen_k,
                                    start + BLOCK_M > seqlen_q,
                                    BLOCK_M,
                                    BLOCK_N)
    o = Out + (case * MAX_TILES + tile) * 8
    tl.store(o + 0, lb_lo)
    tl.store(o + 1, lb_hi)
    tl.store(o + 2, fb_lo)
    tl.store(o + 3, fb_hi)
    tl.store(o + 4, rb_lo)
    tl.store(o + 5, rb_hi)
    tl.store(o + 6, window_left)
    tl.store(o + 7, window_right)


def _resolve(window_left, window_right, p, seqlen_q, seqlen_k):
    """(L, R) per docs/Generalized Sliding Window Attention.md, A-2."""
    d = seqlen_k - seqlen_q
    L = seqlen_q if window_left in SENTINELS else window_left - p * d
    if window_right == WINDOW_TOP_LEFT_ALIGNED:
        R = 0
    elif window_right == WINDOW_BOTTOM_RIGHT_ALIGNED:
        R = d
    else:
        R = window_right + p * d
    return L, R


def _blocks(lo, hi):
    return set(range(lo, hi + 1)) if lo <= hi else set()


def _check_tile(L, R, seqlen_q, seqlen_k, start, tile_m, tile_n, intervals, exact):
    """Errors for one tile in row-major terms (rows: q, columns: k)."""
    lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi = intervals
    end = min(start + tile_m, seqlen_q)
    masked = _blocks(lb_lo, lb_hi) | _blocks(rb_lo, rb_hi)
    full = _blocks(fb_lo, fb_hi)
    # Rows are consecutive and every row's band has the same width L + R + 1,
    # so the union over the tile is one interval -- or nothing if L + R < 0.
    lo = max(0, start - L)
    hi = min(seqlen_k - 1, end - 1 + R)
    need = _blocks(lo // tile_n, hi // tile_n) if L + R >= 0 and lo <= hi else set()
    errors = []
    if need - masked - full:
        errors.append(f'misses blocks {sorted(need - masked - full)}')
    if masked & full:
        errors.append(f'block both masked and full {sorted(masked & full)}')
    n_blocks = triton.cdiv(seqlen_k, tile_n)
    if any(b < 0 or b >= n_blocks for b in masked | full):
        errors.append(f'block outside [0, {n_blocks})')
    for b in full:
        # The full-block path masks nothing: not seqlen_k, not either window
        # line, and (with mask_on_seq_q) not seqlen_q either.
        if (b + 1) * tile_n > seqlen_k:
            errors.append(f'full block {b} crosses seqlen_k')
        if b * tile_n < end - 1 - L or (b + 1) * tile_n - 1 > start + R:
            errors.append(f'full block {b} not inside the band')
        if start + tile_m > seqlen_q:
            errors.append(f'full block {b} in a tile that needs seq_q masking')
    if exact and (masked | full) - need:
        errors.append(f'visits unneeded blocks {sorted((masked | full) - need)}')
    return errors


LENGTHS = [1, 7, 16, 17, 63, 64, 65, 128, 143, 257]
NUMERIC_WINDOWS = [
    (16, 4),
    (0, 0),
    (4, -4),         # band of one column, right of nothing: L + R = 0
    (3, -5),         # L + R < 0: every row empty
    (300, -250),     # band entirely left of key 0 for the first rows
    (-100, 150),     # band entirely right of key 0
    (2**20, -2**20 + 5),
    (-2**20 + 5, 2**20),
    (0, 2**20),
    (2**20, 0),
]
SENTINEL_WINDOWS = [
    (WINDOW_TOP_LEFT_ALIGNED, WINDOW_TOP_LEFT_ALIGNED),
    (WINDOW_BOTTOM_RIGHT_ALIGNED, WINDOW_BOTTOM_RIGHT_ALIGNED),
    (WINDOW_BOTTOM_RIGHT_ALIGNED, 0),
    (WINDOW_BOTTOM_RIGHT_ALIGNED, 8),
]


def _probe_cases():
    cases = []
    for seqlen_q in LENGTHS:
        for seqlen_k in LENGTHS:
            for p in (0, 1):
                for wl, wr in NUMERIC_WINDOWS + SENTINEL_WINDOWS:
                    cases.append((wl, wr, p, seqlen_q, seqlen_k))
            # The case the library reported: sq=143, sk=63, (16, 4), p=1
            # resolves to L = 96, R = -76. Keep its neighbourhood dense.
            for wl in (16, 96, 143, 200):
                for wr in (-200, -76, -63, -1):
                    cases.append((wl, wr, 0, seqlen_q, seqlen_k))
    return cases

TILES = [(16, 16), (32, 16), (64, 16), (64, 32), (64, 64), (128, 64), (128, 128), (256, 64)]

@pytest.mark.parametrize('transposed', [False, True], ids=['QMajor', 'KMajor'])
@pytest.mark.parametrize('tile', TILES, ids=lambda t: f'M{t[0]}_N{t[1]}')
def test_interval_probe(tile, transposed):
    BLOCK_M, BLOCK_N = tile
    cases = _probe_cases()
    params = torch.tensor([(wl, wr, p << 18, sq, sk) for wl, wr, p, sq, sk in cases],
                          dtype=torch.int32,
                          device='cuda')
    major = BLOCK_N if transposed else BLOCK_M
    max_tiles = triton.cdiv(max(LENGTHS), major)
    out = torch.full((len(cases), max_tiles, 8), -999, dtype=torch.int32, device='cuda')
    _interval_probe[(max_tiles, len(cases))](params,
                                             out,
                                             MAX_TILES=max_tiles,
                                             BLOCK_M=BLOCK_M,
                                             BLOCK_N=BLOCK_N,
                                             TRANSPOSED=transposed)
    out = out.cpu().tolist()
    failures = []
    for ci, (wl, wr, p, sq, sk) in enumerate(cases):
        L, R = _resolve(wl, wr, p, sq, sk)
        # Only the Q-major walk of a sentinel window is tight today. The
        # K-major walk (dk/dv) and numeric windows may visit one block whose
        # columns lie past seqlen_k (or before 0) in element space; it is
        # fully masked, so it costs time, not correctness. Tighten this gate
        # if calculate_intervals ever clamps in element space.
        exact = not transposed and wl in SENTINELS and wr in SENTINELS + (0, 8)
        for t in range(triton.cdiv(sq if not transposed else sk, major)):
            row = out[ci][t]
            assert (row[6], row[7]) == (L, R), f'parse_window {(wl, wr, p, sq, sk)=} gave {row[6:]} != {(L, R)}'
            if transposed:
                # j in [i - L, i + R]  <=>  i in [j - R, j + L]: the K-major
                # view is the same check with (L, R) swapped.
                errors = _check_tile(R, L, sk, sq, t * BLOCK_N, BLOCK_N, BLOCK_M, row[:6], exact)
            else:
                errors = _check_tile(L, R, sq, sk, t * BLOCK_M, BLOCK_M, BLOCK_N, row[:6], exact)
            if errors:
                failures.append(f'{(wl, wr, p, sq, sk)=} (L, R)={(L, R)} tile {t}: {row[:6]} {errors}')
    assert not failures, f'{len(failures)} bad tiles, first: ' + '\n'.join(failures[:8])

# ---------------------------------------------------------------------------
# 2./3. End to end
# ---------------------------------------------------------------------------

# (fwd tile, bwd tile). The fwd tiles are the ones aot/attn_fwd.py compiles on
# wave64; the bwd tiles satisfy BLOCK_M >= BLOCK_N as aot/bwd_kernel_*.py do.
TILE_PAIRS = [
    ((32, 16), (16, 16)),
    ((64, 32), (32, 16)),
    ((64, 64), (32, 32)),
    ((128, 64), (64, 32)),
    ((128, 128), (64, 16)),
]

def _fmt_tile_pair(pair):
    (fm, fn), (bm, bn) = pair
    return f'F{fm}x{fn}_B{bm}x{bn}'

# (seqlen_q, seqlen_k, (window_left, window_right, p))
DENSE_CASES = [
    (143, 63, (16, 4, 1)),         # L = 96, R = -76: the reported library case
    (143, 63, (16, 4, 0)),
    (64, 8, (16, 4, 1)),           # L = 72 > seqlen_q, R = -52
    (128, 8, (16, 4, 1)),          # L = 136 > seqlen_q, R = -116
    (300, 64, (300, -250, 0)),     # first 250 rows miss [0, seqlen_k) on the left
    (64, 128, (-100, 150, 0)),     # rows >= 28 miss [0, seqlen_k) on the right
    (128, 64, (300, -250, 1)),     # every row misses
    (143, 63, (3, -5, 0)),         # L + R < 0: every row empty
    (143, 63, (WINDOW_BOTTOM_RIGHT_ALIGNED, WINDOW_BOTTOM_RIGHT_ALIGNED, 0)),
]

def _fmt_case(c):
    sq, sk, (wl, wr, p) = c
    def side(w):
        return {WINDOW_TOP_LEFT_ALIGNED: 'TL', WINDOW_BOTTOM_RIGHT_ALIGNED: 'BR'}.get(w, str(w))
    return f'q{sq}_k{sk}_w{side(wl)}_{side(wr)}_p{p}'

# (Hdim_qk, Hdim_vo). The first two have BLOCK_DMODEL > Hdim_vo, which is
# what the stray zeros needed; 64 is the control.
HDIMS = [(120, 8), (64, 16), 64]

def _fmt_hdim(h):
    return f'hdim{h}'


def _ext(fwd_block, bwd_block, bwd_fused, out_row_padding=0):
    return AttentionExtraArgs(return_encoded_softmax=False,
                              autotune=False,
                              return_autotune=False,
                              fillnan=True,
                              persistent_type=PersistentType.AUTOSELECT,
                              fwd_block=fwd_block,
                              bwd_block=bwd_block,
                              bwd_fused=bwd_fused,
                              out_row_padding=out_row_padding)


@pytest.mark.parametrize('persistent_type',
                         [PersistentType.NONE, PersistentType.DYNAMIC],
                         ids=['Plain', 'Persistent'])
@pytest.mark.parametrize('hdim', HDIMS, ids=_fmt_hdim)
@pytest.mark.parametrize('case', DENSE_CASES, ids=_fmt_case)
@pytest.mark.parametrize('tiles', TILE_PAIRS, ids=_fmt_tile_pair)
def test_out_row_padding(tiles, case, hdim, persistent_type):
    seqlen_q, seqlen_k, window = case
    dqk, dvo = hdim if isinstance(hdim, tuple) else (hdim, hdim)
    # >= BLOCK_DMODEL - Hdim_vo for every HDIMS entry, so a stray store
    # lands in the padding instead of past the allocation (a GPU fault).
    pad = 128
    torch.manual_seed(5)
    q = torch.rand((2, 3, seqlen_q, dqk), device='cuda', dtype=torch.float16)
    k = torch.rand((2, 3, seqlen_k, dqk), device='cuda', dtype=torch.float16)
    v = torch.rand((2, 3, seqlen_k, dvo), device='cuda', dtype=torch.float16)
    ext = _ext(tiles[0], tiles[1], False, out_row_padding=pad)
    ext.persistent_type = persistent_type
    out, _, _ = attention(q, k, v, None, window, 1.0 / dqk, 0.0, ext)
    padding = torch.as_strided(out,
                               size=(*out.shape[:3], pad),
                               stride=out.stride(),
                               storage_offset=out.storage_offset() + dvo)
    written = ~torch.isnan(padding)
    assert not written.any(), f'{int(written.sum())} padding elements written'
    assert not torch.isnan(out).any(), 'Out has rows the forward never stored'


def _run_dense(tiles, case, hdim, bwd_fused, dtype=torch.float16):
    seqlen_q, seqlen_k, window = case
    dqk = hdim[0] if isinstance(hdim, tuple) else hdim
    torch.manual_seed(20)
    ctx = SdpaContext(2, 3, hdim, seqlen_q, seqlen_k, dtype,
                      bias_type=None, storage_flip=None, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=False, skip_dk_dv=False, skip_db=True)
    q, k, v, b = ctx.dev_tensors
    sm_scale = 1.0 / dqk
    out, _, _ = attention(q, k, v, b, window, sm_scale, 0.0, _ext(tiles[0], tiles[1], bwd_fused))
    ctx.compute_ref_forward(SdpaParams(causal=window, sm_scale=sm_scale, dropout_p=0.0, dropout_mask=None))
    ctx.compute_backward(out, torch.rand_like(out))
    is_allclose, _, grads_allclose, _, tfts = ctx.validate_with_reference(out,
                                                                         ctx.dout_tensors,
                                                                         return_target_fudge_factors=True)
    assert is_allclose, f'Forward {tfts=}'
    dq_ok, dk_ok, dv_ok, _ = grads_allclose
    assert dq_ok and dk_ok and dv_ok, f'{dq_ok=} {dk_ok=} {dv_ok=} {tfts=}'


@pytest.mark.parametrize('bwd_fused', [False, True], ids=['Split', 'Fused'])
@pytest.mark.parametrize('hdim', HDIMS, ids=_fmt_hdim)
@pytest.mark.parametrize('case', DENSE_CASES, ids=_fmt_case)
@pytest.mark.parametrize('tiles', TILE_PAIRS, ids=_fmt_tile_pair)
def test_window_fwd_bwd(tiles, case, hdim, bwd_fused):
    _run_dense(tiles, case, hdim, bwd_fused)


# The batch modules/flash/tests/test_varlen.py drew for the reported failure:
# seq 5 and 6 are sq=143, sk=63, whose rows 76.. are the first live ones.
VARLEN_SEQLENS_Q = [4, 4, 4, 64, 128, 143, 143]
VARLEN_SEQLENS_K = [8, 64, 8, 8, 8, 63, 63]

@pytest.mark.parametrize('p', [0, 1], ids=['TL', 'BR'])
@pytest.mark.parametrize('hdim', HDIMS, ids=_fmt_hdim)
@pytest.mark.parametrize('tiles', TILE_PAIRS, ids=_fmt_tile_pair)
def test_window_varlen(tiles, hdim, p):
    dqk = hdim[0] if isinstance(hdim, tuple) else hdim
    torch.manual_seed(20)
    ctx = VarlenSdpaContext(3, hdim, VARLEN_SEQLENS_Q, VARLEN_SEQLENS_K, torch.float16, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=False, skip_dk_dv=False, skip_db=True)
    q, k, v, b = ctx.dev_tensors
    window = (16, 4, p)
    sm_scale = 1.0 / dqk
    ext = _ext(tiles[0], tiles[1], None)
    out, _, _ = varlen_attention(q,
                                 k,
                                 v,
                                 np.array(VARLEN_SEQLENS_Q),
                                 np.array(VARLEN_SEQLENS_K),
                                 window,
                                 sm_scale,
                                 0.0,
                                 ext)
    ctx.compute_ref_forward(SdpaParams(causal=window, sm_scale=sm_scale, dropout_p=0.0, dropout_mask=None))
    ctx.compute_backward(out, torch.rand_like(out))
    is_allclose, _, grads_allclose, _, tfts = ctx.validate_with_reference(out,
                                                                         ctx.dout_tensors,
                                                                         return_target_fudge_factors=True)
    assert is_allclose, f'Forward {tfts=}'
    dq_ok, dk_ok, dv_ok, _ = grads_allclose
    assert dq_ok and dk_ok and dv_ok, f'{dq_ok=} {dk_ok=} {dv_ok=} {tfts=}'
