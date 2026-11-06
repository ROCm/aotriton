#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`principal_diagonal_choice` (VarlenBits bit 18) in the gfx950 dQ kernel, JIT.

With `d = seqlen_k - seqlen_q` per sequence and `p` the bit, a numeric window
bound becomes `L = window_left - p*d`, `R = window_right + p*d`; a sentinel
resolves as before and is not shifted. The predicate `i - L <= j <= i + R` is
unchanged (docs/Generalized Sliding Window Attention.md, A-1/A-2).

The kernel resolves the window in `ParityKernelContext.init_tile_bounds`, and
every dQ-side consumer -- the tile count, the left-edge tile skip, the 32-row
and 16-row window masks -- reads the resolved pair. A shifted pair is where
the assumptions those consumers could have made stop holding: `L` negative, `R`
past `seqlen_k`, a band that does not touch the diagonal at all, a band that
misses every key of a row. So the oracle is a float32 autograd of the masked
softmax, over dense and varlen calls with `seqlen_q` <, == and > `seqlen_k`
(mixed in one varlen batch), both MFMA families, and both values of the bit.

LSE and Delta come from torch, not from the forward, so this tests dQ alone.
A row that sees no key gets an LSE of **0.0** rather than the forward's `+inf`
contract: `+inf` would zero `P` by itself and let a missing mask pass, while a
finite LSE leaves the mask as the only thing that can make the row's dQ zero.

Requires a gfx950 device, `AOTRITON_FLYDSL_KERNEL_ROOT`, and a ROCm the
FlyDSL compiler can link against; skips cleanly without any of them. Set
`FLYDSL_EXTRA_SOURCE_DIRS` to this directory when editing the kernel: the
FlyDSL disk cache does not hash these sources.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
    FLYDSL_EXTRA_SOURCE_DIRS=$PWD/modules/flash/flyc \\
        python -m pytest modules/flash/flyc/test_bwd_dq_principal_diagonal_gfx950.py
"""

import functools
import math
import os
import sys
from pathlib import Path

import pytest

_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not _KERNEL_ROOT
    or not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="needs a gfx950 device and AOTRITON_FLYDSL_KERNEL_ROOT",
)

NUM_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16
# Relative to the largest reference magnitude in the call: bf16 operands
# against an f32 reference, accumulated over up to a few hundred keys.
REL_TOL = 2e-2

# Imported lazily by the fixtures below, so a host without the FlyDSL checkout
# still collects (and skips) this file.
WINDOW_SENTINEL = -2147483646  # fmha.WINDOW_BOTRIGHT; either sentinel is unbounded on the left
WINDOW_TOPLEFT = -2147483647  # fmha.WINDOW_TOPLEFT

WINDOWS = {
    "sentinel_left_0": (WINDOW_SENTINEL, 0),
    "8_0": (8, 0),
    "0_8": (0, 8),
    "4_4": (4, 4),
    "neg1_3": (-1, 3),
}

# Dense `(seqlen_q, seqlen_k)`: decode, chunked prefill, square, and more
# queries than keys. Lengths are not tile multiples on purpose.
DENSE_SHAPES = {
    "decode": (1, 200),
    "chunked_prefill": (48, 200),
    "square": (150, 150),
    "q_longer": (200, 70),
}

# One varlen batch, every relation between the lengths in it: decode, chunked
# prefill, square, more queries than keys, and a short sequence whose shifted
# band misses its keys entirely for some windows.
VARLEN_SEQS = ((1, 130), (37, 160), (96, 96), (150, 40), (20, 3))

# 32 is the default family at head_dim 64; 16 is selected by the knob and is
# the default from head_dim 256 up.
MFMA_ROWS = (32, 16)


@functools.cache
def _build(mfma_rows):
    """A causal, window-capable dQ build at HEAD_DIM, compiled in-process."""
    from fmha_bwd_dq_gfx950 import build_fmha_bwd_dq_gfx950_module_primary
    from fmha_tuning_bwd_dq_gfx950 import bwd_dq_knobs
    from fmha_tuning_gfx950 import FmhaInputMetadata

    meta = FmhaInputMetadata(num_heads=NUM_HEADS,
                             head_dim=HEAD_DIM,
                             causal=True,
                             window=True,
                             dtype_str="bf16")
    knobs = bwd_dq_knobs("gfx950", mfma_rows=mfma_rows).resolve(meta)
    fn = build_fmha_bwd_dq_gfx950_module_primary(meta, knobs)
    assert fn.traits.MFMA_N == mfma_rows
    return fn


def _abi():
    import fmha_abi_gfx1201 as abi

    return abi


def _resolved_bounds(window, seqlen_q, seqlen_k, p):
    """`(L, R)` per the spec, on the host. Mirrors `fmha.resolve_window`."""
    wl, wr = window
    d = seqlen_k - seqlen_q
    sentinels = (WINDOW_SENTINEL, WINDOW_TOPLEFT)
    left = seqlen_q if wl in sentinels else wl - p * d
    if wr == WINDOW_TOPLEFT:
        right = 0
    elif wr == WINDOW_SENTINEL:
        right = d
    else:
        right = wr + p * d
    return left, right


def _reference(q, k, v, do, window, p, sm_scale):
    """`(dq, lse, delta, live_rows)` for one sequence, `[H, S, D]` operands.

    float32 autograd of `softmax(mask(scale * q k^T)) v`. A row with no live
    key has `P = 0`, so its O and dQ are zero; it is built that way (a large
    negative fill, then the row zeroed) rather than through an all-`-inf`
    softmax, whose gradient is NaN.
    """
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    left, right = _resolved_bounds(window, seqlen_q, seqlen_k, p)
    i = torch.arange(seqlen_q, device=q.device).view(-1, 1)
    j = torch.arange(seqlen_k, device=q.device).view(1, -1)
    keep = (j >= i - left) & (j <= i + right)
    live = keep.any(dim=-1)

    qf = q.float().requires_grad_(True)
    s = torch.einsum("hqd,hkd->hqk", qf, k.float()) * sm_scale
    s = s.masked_fill(~keep, -1e30)
    prob = torch.softmax(s, dim=-1) * live.view(1, -1, 1)
    o = torch.einsum("hqk,hkd->hqd", prob, v.float())
    o.backward(do.float())

    with torch.no_grad():
        lse = torch.logsumexp(s, dim=-1)
        lse = torch.where(live.view(1, -1), lse, torch.zeros_like(lse))
        delta = (o.detach() * do.float()).sum(dim=-1)
    return qf.grad, lse, delta, live


def _check(got, ref, live, label):
    """dQ matches on live rows, and is exactly zero on rows that see no key."""
    scale = max(1.0, ref.abs().max().item())
    err = (got.float() - ref).abs()
    assert bool(torch.isfinite(got).all()), f"{label}: non-finite dQ"
    worst = err.max().item() / scale
    assert worst <= REL_TOL, f"{label}: dQ off by {worst:.3e} (relative to {scale:.3e})"
    dead = got[:, ~live]
    assert int((dead != 0).sum()) == 0, (
        f"{label}: {int((dead != 0).sum())} nonzero dQ elements in {int((~live).sum())} rows that see no key"
    )


def _inputs(seqlen_q, seqlen_k, batch, generator):
    shape_q = (batch, NUM_HEADS, seqlen_q, HEAD_DIM)
    shape_k = (batch, NUM_HEADS, seqlen_k, HEAD_DIM)
    q = torch.randn(shape_q, generator=generator, device="cuda").to(DTYPE)
    k = torch.randn(shape_k, generator=generator, device="cuda").to(DTYPE)
    v = torch.randn(shape_k, generator=generator, device="cuda").to(DTYPE)
    do = torch.randn(shape_q, generator=generator, device="cuda").to(DTYPE)
    return q, k, v, do


@pytest.mark.parametrize("mfma_rows", MFMA_ROWS)
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window_name", tuple(WINDOWS))
@pytest.mark.parametrize("shape_name", tuple(DENSE_SHAPES))
def test_dense(mfma_rows, p, window_name, shape_name):
    """Dense: the bit arrives in a dense word, `varlen_dense(principal_diagonal=...)`."""
    abi = _abi()
    fn = _build(mfma_rows)
    window = WINDOWS[window_name]
    seqlen_q, seqlen_k = DENSE_SHAPES[shape_name]
    batch = 2
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    q, k, v, do = _inputs(seqlen_q, seqlen_k, batch, gen)

    refs = [_reference(q[b], k[b], v[b], do[b], window, p, sm_scale) for b in range(batch)]
    lse = torch.stack([r[1] for r in refs]).reshape(batch * NUM_HEADS, seqlen_q).contiguous()
    delta = torch.stack([r[2] for r in refs]).reshape(batch * NUM_HEADS, seqlen_q).contiguous()
    dq = torch.full_like(q, float("nan"))
    principal_diagonal = (abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT
                          if p
                          else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT)
    fn(q,
       k,
       v,
       do,
       dq,
       lse,
       delta,
       batch,
       seqlen_q,
       seqlen_k=seqlen_k,
       scale=sm_scale,
       window=window,
       varlen=abi.varlen_dense(principal_diagonal=principal_diagonal))
    torch.cuda.synchronize()
    for b in range(batch):
        _check(dq[b], refs[b][0], refs[b][3], f"dense {shape_name} {window_name} p={p} m{mfma_rows} batch {b}")


@pytest.mark.parametrize("mfma_rows", MFMA_ROWS)
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window_name", tuple(WINDOWS))
def test_varlen(mfma_rows, p, window_name):
    """Packed varlen, lengths mixed in one batch: the bit arrives in `varlen['bits']`.

    Each sequence's shift is its own `seqlen_k - seqlen_q`, which is the case
    a host-side conversion cannot express and the reason the bit exists.
    """
    abi = _abi()
    fn = _build(mfma_rows)
    window = WINDOWS[window_name]
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    gen = torch.Generator(device="cuda").manual_seed(4321)

    seqs = [_inputs(sq, sk, 1, gen) for sq, sk in VARLEN_SEQS]
    refs = [_reference(q[0], k[0], v[0], do[0], window, p, sm_scale) for q, k, v, do in seqs]
    q = torch.cat([s[0] for s in seqs], dim=2).contiguous()
    k = torch.cat([s[1] for s in seqs], dim=2).contiguous()
    v = torch.cat([s[2] for s in seqs], dim=2).contiguous()
    do = torch.cat([s[3] for s in seqs], dim=2).contiguous()
    lse = torch.cat([r[1] for r in refs], dim=1).contiguous()
    delta = torch.cat([r[2] for r in refs], dim=1).contiguous()
    dq = torch.full_like(q, float("nan"))

    lens_q = [sq for sq, _ in VARLEN_SEQS]
    lens_k = [sk for _, sk in VARLEN_SEQS]
    cu_q = torch.tensor([0] + list(_cumsum(lens_q)), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0] + list(_cumsum(lens_k)), dtype=torch.int32, device="cuda")
    varlen = abi.varlen_compact(cu_q,
                                cu_k,
                                max(lens_q),
                                max(lens_k),
                                lse_tokens=sum(lens_q))
    principal_diagonal = (abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT
                          if p
                          else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT)
    varlen["bits"] = abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                     abi.VARLEN_COMPACT_SIDE,
                                     principal_diagonal=principal_diagonal)
    fn(q,
       k,
       v,
       do,
       dq,
       lse,
       delta,
       1,
       max(lens_q),
       seqlen_k=max(lens_k),
       scale=sm_scale,
       window=window,
       varlen=varlen,
       num_seqlens=len(VARLEN_SEQS))
    torch.cuda.synchronize()

    start = 0
    for n, ((sq, sk), ref) in enumerate(zip(VARLEN_SEQS, refs)):
        got = dq[0, :, start:start + sq]
        _check(got, ref[0], ref[3], f"varlen seq {n} ({sq}, {sk}) {window_name} p={p} m{mfma_rows}")
        start += sq


@pytest.mark.parametrize("mfma_rows", MFMA_ROWS)
@pytest.mark.parametrize("window", ((WINDOW_SENTINEL, WINDOW_SENTINEL), (WINDOW_TOPLEFT, WINDOW_TOPLEFT)))
def test_sentinels_ignore_the_bit(mfma_rows, window):
    """A fully-sentinel window is bit-identical under p = 0 and p = 1."""
    abi = _abi()
    fn = _build(mfma_rows)
    seqlen_q, seqlen_k = 150, 70
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    gen = torch.Generator(device="cuda").manual_seed(99)
    q, k, v, do = _inputs(seqlen_q, seqlen_k, 1, gen)
    _, lse, delta, _ = _reference(q[0], k[0], v[0], do[0], window, 0, sm_scale)
    outs = []
    for pd in (abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT, abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT):
        dq = torch.full_like(q, float("nan"))
        fn(q,
           k,
           v,
           do,
           dq,
           lse.contiguous(),
           delta.contiguous(),
           1,
           seqlen_q,
           seqlen_k=seqlen_k,
           scale=sm_scale,
           window=window,
           varlen=abi.varlen_dense(principal_diagonal=pd))
        torch.cuda.synchronize()
        outs.append(dq)
    assert torch.equal(outs[0], outs[1])


def test_dense_word_rejects_seqinfo():
    """A dense word reads no sequence info; handing it some is a caller error."""
    abi = _abi()
    fn = _build(32)
    q = torch.zeros((1, NUM_HEADS, 8, HEAD_DIM), dtype=DTYPE, device="cuda")
    rows = torch.zeros((NUM_HEADS, 8), dtype=torch.float32, device="cuda")
    varlen = abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)
    varlen["seqinfo_q0"] = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="dense VarlenBits word"):
        fn(q,
           q,
           q,
           q,
           q.clone(),
           rows,
           rows,
           1,
           8,
           window=(4, 4),
           varlen=varlen)


def _cumsum(values):
    total = 0
    for x in values:
        total += x
        yield total


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
