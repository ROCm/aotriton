#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`principal_diagonal_choice` (VarlenBits bit 18) in the flyc dQ kernel, JIT.

With `d = seqlen_k - seqlen_q` per sequence and `p` the bit, a numeric window
bound becomes `L = window_left - p*d`, `R = window_right + p*d`; a sentinel
resolves as before and is not shifted. Row `i` keeps column `j` iff
`0 <= j < seqlen_k` and `i - L <= j <= i + R` (docs/Generalized Sliding Window
Attention.md, A-1/A-2).

The bit travels in the VarlenBits word and nowhere else: a dense call passes
`varlen=abi.varlen_dense(principal_diagonal=...)`, a varlen call sets it in
`varlen["bits"]`.

Every dQ-side consumer of the window -- the tile count, the left-edge tile
skip, the row masks -- reads the resolved pair. A shifted pair is where the
assumptions those consumers could have made stop holding: `L` negative, `R`
past `seqlen_k`, a band that does not touch the diagonal, a band that misses
every key of a row. So the oracle is a float32 autograd of the masked softmax,
over dense and varlen calls with `seqlen_q` <, == and > `seqlen_k` (mixed in
one varlen batch), decode and chunked prefill, both dtypes, and both values of
the bit.

LSE and Delta come from torch, not from the forward, so this tests dQ alone.
A row that sees no key must get dQ = 0 whatever its LSE says, so every call
runs twice: with the forward's `+inf` LSE for those rows, and with a finite
0.0, which leaves the mask as the only thing that can make the row's dQ zero.

Builds for the first GPU's arch through `tuning_dq`; skips on an arch it has
no row for, or without `AOTRITON_FLYDSL_KERNEL_ROOT`. gfx1201's launcher
plumbing (no GPU) is `gfx1201/test_bwd_dq_launcher.py`.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/tests/test_bwd_dq_principal_diagonal.py
"""

import functools
import math
import sys
from itertools import accumulate
from pathlib import Path

import pytest

import jit_arch
import tuning_dq
from fmha_meta import FmhaInputMetadata

torch = pytest.importorskip("torch")
ARCH = jit_arch.require(tuning_dq)

import fmha_utils as abi  # noqa: E402

NUM_HEADS = 2
HEAD_DIM = 64
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
# bf16: max error relative to the largest reference magnitude in the call.
# f16: elementwise, and dS is rounded to f16 for the last GEMM. Both against a
# float32 reference, accumulated over up to a few hundred keys.
BF16_REL_TOL = 2e-2
F16_ATOL = 2e-2
F16_RTOL = 2e-2

TL = abi.WINDOW_TOPLEFT
BR = abi.WINDOW_BOTRIGHT
_SENTINELS = (TL, BR)

# (window_left, window_right). The first is flash-attention's `(-1, 0)`, an
# unbounded left spelled with a sentinel; with p = 1 it is bottom-right causal.
# `(-1, 3)` is a band that never touches the diagonal. The last pair is all
# sentinels, which p must not move.
WINDOWS = {
    "sentinel_left_0": (BR, 0),
    "8_0": (8, 0),
    "0_8": (0, 8),
    "4_4": (4, 4),
    "neg1_3": (-1, 3),
    "sentinel_TL_BR": (TL, BR),
}

# (seqlen_q, seqlen_k): decode, chunked prefill, square, more queries than
# keys; several tiles each way, lengths off the tile multiples on purpose.
DENSE_SHAPES = [
    (1, 200),
    (48, 200),
    (64, 256),
    (96, 200),
    (150, 150),
    (160, 160),
    (200, 70),
    (200, 72),
]

# Varlen batches, each mixing every relation between the lengths: decode,
# chunked prefill, square, more queries than keys. `short` ends in a sequence
# whose shifted band misses its keys entirely for some windows.
VARLEN_BATCHES = {
    "short": ((1, 130), (37, 160), (96, 96), (150, 40), (20, 3)),
    "chunked": ((37, 100), (64, 64), (130, 50), (1, 77), (48, 240)),
}

# Knob variants: None is the policy's choice; gfx950 also builds its 16-row
# MFMA family, which head_dim 64 does not pick by itself.
VARIANTS = {
    "policy": None,
    "mfma_rows16": ("gfx950", {"mfma_rows": 16}),
}


def _shape_id(s):
    return f"q{s[0]}_k{s[1]}"


@functools.cache
def _build(dtype_str, variant):
    """A window dQ build at HEAD_DIM, compiled in-process."""
    meta = FmhaInputMetadata(num_heads=NUM_HEADS,
                             block_dmodel=None,
                             head_dim=HEAD_DIM,
                             dtype_str=dtype_str,
                             causal_type=3)
    return jit_arch.build(tuning_dq, ARCH, meta, VARIANTS[variant])[0]


def _resolve(window, seqlen_q, seqlen_k, p):
    """`(L, R)` per A-2, on the host. Mirrors `fmha.resolve_window`."""
    wl, wr = window
    d = seqlen_k - seqlen_q
    left = seqlen_q if wl in _SENTINELS else wl - p * d
    if wr in _SENTINELS:
        right = 0 if wr == TL else d
    else:
        right = wr + p * d
    return left, right


def _keep_mask(window, seqlen_q, seqlen_k, p, device):
    left, right = _resolve(window, seqlen_q, seqlen_k, p)
    i = torch.arange(seqlen_q, device=device).view(-1, 1)
    j = torch.arange(seqlen_k, device=device).view(1, -1)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, keep, sm_scale):
    """`(dq, lse, delta, empty)` for one sequence, `[H, S, D]` operands, float32.

    A row with no live key has LSE +inf (the forward's contract) and contributes
    nothing, so its dQ is exactly zero. Its scores are replaced by a finite
    constant before the logsumexp so that autograd never sees `-inf - -inf`.
    """
    qf = q.float().detach().requires_grad_(True)
    kf, vf, dof = k.float(), v.float(), do.float()
    s = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    empty = ~keep.any(dim=-1)
    s = s.masked_fill(~keep, float("-inf")).masked_fill(empty.view(1, -1, 1), 0.0)
    lse = torch.logsumexp(s, dim=-1, keepdim=True)
    prob = torch.exp(s - lse).masked_fill(~keep, 0.0)
    o = torch.einsum("hqk,hkd->hqd", prob, vf)
    (dq,) = torch.autograd.grad((o * dof).sum(), qf)
    lse = lse.squeeze(-1).detach().masked_fill(empty.view(1, -1), float("inf"))
    delta = (o.detach() * dof).sum(dim=-1)
    return dq, lse, delta, empty


def _dead_lse_variants(lse):
    """The LSE as the forward writes it (+inf for rows with no key), then finite."""
    return (("inf", lse), ("finite", torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)))


def _check(dq, ref, empty, dtype_str, what):
    """dQ matches on live rows, and is exactly zero on rows that see no key."""
    assert not bool(torch.isnan(dq).any()), f"{what}: NaN in dQ"
    assert bool(torch.isfinite(dq).all()), f"{what}: non-finite dQ"
    dead = dq[:, empty]
    assert int((dead != 0).sum()) == 0, f"{what}: {int((dead != 0).sum())} nonzero dQ elements in rows with no key"
    if dtype_str == "bf16":
        scale = max(1.0, ref.abs().max().item())
        worst = (dq.float() - ref).abs().max().item() / scale
        assert worst <= BF16_REL_TOL, f"{what}: dQ off by {worst:.3e} (relative to {scale:.3e})"
    else:
        torch.testing.assert_close(dq.float(), ref, atol=F16_ATOL, rtol=F16_RTOL, msg=lambda m: f"{what}: {m}")


def _inputs(shape_q, shape_k, dtype_str, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dtype = DTYPES[dtype_str]
    q = torch.randn(shape_q, device="cuda", dtype=dtype, generator=gen)
    k = torch.randn(shape_k, device="cuda", dtype=dtype, generator=gen)
    v = torch.randn(shape_k, device="cuda", dtype=dtype, generator=gen)
    do = torch.randn(shape_q, device="cuda", dtype=dtype, generator=gen)
    return q, k, v, do


def _bits(p):
    return abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype_str", list(DTYPES))
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape", DENSE_SHAPES, ids=_shape_id)
def test_dense(variant, dtype_str, p, window_name, shape):
    """Dense: the bit arrives in a dense word, `varlen_dense(principal_diagonal=...)`."""
    fn = _build(dtype_str, variant)
    window = WINDOWS[window_name]
    sq, sk = shape
    batch = 2
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v, do = _inputs((batch, NUM_HEADS, sq, HEAD_DIM), (batch, NUM_HEADS, sk, HEAD_DIM), dtype_str, sq * 1000 + sk)
    keep = _keep_mask(window, sq, sk, p, "cuda")
    refs = [_reference(q[b], k[b], v[b], do[b], keep, sm_scale) for b in range(batch)]
    lse = torch.stack([r[1] for r in refs]).reshape(batch * NUM_HEADS, sq).contiguous()
    delta = torch.stack([r[2] for r in refs]).reshape(batch * NUM_HEADS, sq).contiguous()

    for lse_name, lse_in in _dead_lse_variants(lse):
        dq = torch.full_like(q, float("nan"))
        fn(q,
           k,
           v,
           do,
           dq,
           lse_in,
           delta,
           batch,
           sq,
           seqlen_k=sk,
           scale=sm_scale,
           window=window,
           varlen=abi.varlen_dense(principal_diagonal=_bits(p)))
        torch.cuda.synchronize()
        for b, (ref, _, _, empty) in enumerate(refs):
            _check(dq[b], ref, empty, dtype_str, f"dense q{sq} k{sk} {window_name} p={p} {lse_name} LSE batch {b}")


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype_str", list(DTYPES))
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("batch_name", list(VARLEN_BATCHES))
def test_varlen(variant, dtype_str, p, window_name, batch_name):
    """Packed (1, H, T, D), so `d` differs per sequence: the shift has to be
    resolved inside the kernel, once per sequence. The bit arrives in
    `varlen['bits']`."""
    fn = _build(dtype_str, variant)
    window = WINDOWS[window_name]
    seqs = VARLEN_BATCHES[batch_name]
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    lens_q = [s[0] for s in seqs]
    lens_k = [s[1] for s in seqs]
    tq, tk = sum(lens_q), sum(lens_k)
    q, k, v, do = _inputs((1, NUM_HEADS, tq, HEAD_DIM), (1, NUM_HEADS, tk, HEAD_DIM), dtype_str, 17)
    cu_q = torch.tensor(list(accumulate(lens_q, initial=0)), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor(list(accumulate(lens_k, initial=0)), dtype=torch.int32, device="cuda")

    lse = torch.empty(NUM_HEADS, tq, device="cuda", dtype=torch.float32)
    delta = torch.empty_like(lse)
    refs = []
    for z, (sq, sk) in enumerate(seqs):
        rq = slice(int(cu_q[z]), int(cu_q[z + 1]))
        rk = slice(int(cu_k[z]), int(cu_k[z + 1]))
        keep = _keep_mask(window, sq, sk, p, "cuda")
        ref, r_lse, r_delta, empty = _reference(q[0, :, rq], k[0, :, rk], v[0, :, rk], do[0, :, rq], keep, sm_scale)
        lse[:, rq] = r_lse
        delta[:, rq] = r_delta
        refs.append((rq, ref, empty, sq, sk))

    varlen = abi.varlen_compact(cu_q, cu_k, max(lens_q), max(lens_k), lse_tokens=tq)
    varlen["bits"] = abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                     abi.VARLEN_COMPACT_SIDE,
                                     principal_diagonal=_bits(p))
    for lse_name, lse_in in _dead_lse_variants(lse):
        dq = torch.full_like(q, float("nan"))
        fn(q,
           k,
           v,
           do,
           dq,
           lse_in,
           delta,
           1,
           max(lens_q),
           seqlen_k=max(lens_k),
           num_seqlens=len(seqs),
           scale=sm_scale,
           window=window,
           varlen=varlen)
        torch.cuda.synchronize()
        for rq, ref, empty, sq, sk in refs:
            _check(dq[0, :, rq], ref, empty, dtype_str, f"varlen q{sq} k{sk} {window_name} p={p} {lse_name} LSE")


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype_str", list(DTYPES))
@pytest.mark.parametrize("window", ((BR, BR), (TL, TL)), ids=("BR_BR", "TL_TL"))
def test_sentinels_ignore_the_bit(variant, dtype_str, window):
    """A fully-sentinel window is bit-identical under p = 0 and p = 1."""
    fn = _build(dtype_str, variant)
    sq, sk = 150, 70
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v, do = _inputs((1, NUM_HEADS, sq, HEAD_DIM), (1, NUM_HEADS, sk, HEAD_DIM), dtype_str, 99)
    keep = _keep_mask(window, sq, sk, 0, "cuda")
    _, lse, delta, _ = _reference(q[0], k[0], v[0], do[0], keep, sm_scale)
    outs = []
    for p in (0, 1):
        dq = torch.full_like(q, float("nan"))
        fn(q,
           k,
           v,
           do,
           dq,
           lse.contiguous(),
           delta.contiguous(),
           1,
           sq,
           seqlen_k=sk,
           scale=sm_scale,
           window=window,
           varlen=abi.varlen_dense(principal_diagonal=_bits(p)))
        torch.cuda.synchronize()
        outs.append(dq)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dtype_str", list(DTYPES))
def test_dense_word_th_layout(variant, dtype_str):
    """A dense word may select the TH LSE layout: (B*S, H) rows, same dQ as HT.

    `seqlen_q != NUM_HEADS`, so a row check that assumed HT would reject it.
    """
    fn = _build(dtype_str, variant)
    window = WINDOWS["4_4"]
    sq, sk = 48, 200
    batch = 2
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v, do = _inputs((batch, NUM_HEADS, sq, HEAD_DIM), (batch, NUM_HEADS, sk, HEAD_DIM), dtype_str, 1234)
    keep = _keep_mask(window, sq, sk, 1, "cuda")
    refs = [_reference(q[b], k[b], v[b], do[b], keep, sm_scale) for b in range(batch)]
    lse_bhs = torch.stack([r[1] for r in refs])
    delta_bhs = torch.stack([r[2] for r in refs])
    outs = []
    for layout in (abi.VARLEN_LSE_LAYOUT_HT, abi.VARLEN_LSE_LAYOUT_TH):
        if layout == abi.VARLEN_LSE_LAYOUT_TH:
            lse = lse_bhs.transpose(1, 2).reshape(batch * sq, NUM_HEADS).contiguous()
            delta = delta_bhs.transpose(1, 2).reshape(batch * sq, NUM_HEADS).contiguous()
        else:
            lse = lse_bhs.reshape(batch * NUM_HEADS, sq).contiguous()
            delta = delta_bhs.reshape(batch * NUM_HEADS, sq).contiguous()
        dq = torch.full_like(q, float("nan"))
        fn(q,
           k,
           v,
           do,
           dq,
           lse,
           delta,
           batch,
           sq,
           seqlen_k=sk,
           scale=sm_scale,
           window=window,
           varlen=abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT,
                                   lse_layout=layout))
        torch.cuda.synchronize()
        outs.append(dq)
    assert torch.equal(outs[0], outs[1])
    for b, (ref, _, _, empty) in enumerate(refs):
        _check(outs[1][b], ref, empty, dtype_str, f"dense TH batch {b}")


@pytest.mark.parametrize("key", ("seqinfo_q0", "seqinfo_k1"))
def test_dense_word_rejects_seqinfo(key):
    """A dense word reads no sequence info; handing it some is a caller error."""
    fn = _build("bf16", "policy")
    q = torch.zeros((1, NUM_HEADS, 8, HEAD_DIM), dtype=DTYPES["bf16"], device="cuda")
    rows = torch.zeros((NUM_HEADS, 8), dtype=torch.float32, device="cuda")
    varlen = abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)
    varlen[key] = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="dense VarlenBits word"):
        fn(q,
           q,
           q,
           q,
           q.clone(),
           rows,
           rows.clone(),
           1,
           8,
           window=(4, 4),
           varlen=varlen)


def test_varlen_dense_rejects_bad_choice():
    """Only the two `VARLEN_PRINCIPAL_DIAGONAL_*` words; a bare 1 is not bit 18."""
    with pytest.raises(ValueError, match="principal_diagonal"):
        abi.varlen_dense(principal_diagonal=1)


def test_reference_shift_is_not_vacuous():
    """Every non-sentinel window over a `d != 0` shape must move with p, and
    the all-sentinel one must not -- otherwise the GPU sweep proves nothing."""
    for sq, sk in DENSE_SHAPES:
        for w in WINDOWS.values():
            same = torch.equal(_keep_mask(w, sq, sk, 0, "cpu"), _keep_mask(w, sq, sk, 1, "cpu"))
            moves = sk != sq and not all(x in _SENTINELS for x in w)
            assert same != moves, (sq, sk, w)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
