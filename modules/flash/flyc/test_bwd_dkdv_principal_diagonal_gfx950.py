#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""gfx950 dK/dV under `principal_diagonal_choice` (VarlenBits bit 18), JIT.

With `d = seqlen_k - seqlen_q` for each sequence and `p` the bit, a numeric
window bound becomes `L = window_left - p*d`, `R = window_right + p*d`;
sentinels resolve as they always have and are never shifted. The predicate
`i - L <= j <= i + R` is unchanged. See `causal-principle-axis-plan0.md`
section 3.2 and `docs/Generalized Sliding Window Attention.md` A-2.

The bit travels in the VarlenBits word and nowhere else: a dense call passes
`varlen=abi.varlen_dense(principal_diagonal=...)`, a varlen call sets it in
`varlen["bits"]`.

dK/dV walk column-major -- each KV block cuts its own Q range out of the
window -- so a shifted band is exactly what stresses it: `L` negative, `R`
past `seqlen_k`, a band clear of the diagonal, and KV columns no row sees
(their dK and dV must come back zero, not unwritten).

The reference is a float32 autograd backward of a masked softmax built from
that predicate. LSE and Delta are computed here with torch, not by the
forward kernel: this file tests the backward alone. A row that attends
nothing gets LSE `+inf`, the forward's contract (see
`test_fwd_masked_lse_gfx950.py`).

Requires a gfx950 device, `AOTRITON_FLYDSL_KERNEL_ROOT`, and a `ROCM_PATH`
the FlyDSL compiler can link against; skips cleanly without the first two.

    FLYDSL_EXTRA_SOURCE_DIRS=$PWD/modules/flash/flyc \\
    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_bwd_dkdv_principal_diagonal_gfx950.py
"""

import math
import os
import sys
import zlib
from functools import cache
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
DTYPE = torch.bfloat16
# Max abs error against the fp32 reference, relative to the reference's own
# max magnitude (floored at 1). bf16 inputs and a bf16-packed P/dS feed the
# kernel's GEMMs, so a few 1e-3 of the scale is the expected noise.
REL_TOL = 2e-2

# `None` stands for a sentinel left edge (unbounded); either sentinel resolves
# the left side the same way, so one is enough.
WINDOWS = (
    (None, 0),
    (8, 0),
    (0, 8),
    (4, 4),
    (-1, 3),
)

# Dense shapes, `(seqlen_q, seqlen_k)`: q shorter, equal, longer, decode, and
# a chunked prefill whose chunk does not fill a Q tile.
DENSE_SHAPES = (
    (48, 160),
    (128, 128),
    (160, 48),
    (1, 200),
    (32, 160),
)

# One varlen batch mixing all of the above, packed (`varlen_compact`).
VARLEN_SHAPES = (
    (40, 100),
    (64, 64),
    (100, 40),
    (1, 77),
    (16, 130),
)

# `(head_dim, mfma_rows)`: the 32-row and the 16-row families. head_dim 64's
# table picks 32 rows and 128's picks 16; 64 is also forced to 16 so that
# both families are covered at one width.
BUILDS = (
    (64, None),
    (64, 16),
    (128, None),
)


@cache
def _build(head_dim, mfma_rows):
    from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build

    knobs = {} if mfma_rows is None else {"mfma_rows": mfma_rows}
    fn = build(num_heads=NUM_HEADS,
               num_kv_heads=NUM_HEADS,
               head_dim=head_dim,
               causal=True,
               window=True,
               **knobs)
    if mfma_rows is not None:
        assert fn.knobs.mfma_rows == mfma_rows, f"mfma_rows override ignored: {fn.knobs.mfma_rows}"
    return fn


def _wire_window(window):
    import fmha_common_gfx1201 as fmha

    wl, wr = window
    return (fmha.WINDOW_TOPLEFT if wl is None else wl), wr


def _band(window, p, seqlen_q, seqlen_k, device):
    """`[seqlen_q, seqlen_k]` bool: the live elements, by the spec's predicate."""
    wl, wr = window
    d = seqlen_k - seqlen_q
    left = seqlen_q if wl is None else wl - p * d
    right = wr + p * d
    i = torch.arange(seqlen_q, device=device).view(-1, 1)
    j = torch.arange(seqlen_k, device=device).view(1, -1)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, live, sm_scale):
    """`(dK, dV, LSE, Delta)` in fp32 for one sequence, `[H, S, D]` tensors.

    Rows with no live key take LSE `+inf` and contribute nothing; they are
    kept out of the softmax entirely so no NaN reaches autograd.
    """
    qf, kf, vf, dof = (t.float() for t in (q, k, v, do))
    kf.requires_grad_(True)
    vf.requires_grad_(True)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    empty = ~live.any(dim=-1)  # [Sq]
    masked = scores.masked_fill(~live, float("-inf"))
    masked = torch.where(empty.view(1, -1, 1), torch.zeros_like(masked), masked)
    probs = torch.softmax(masked, dim=-1) * (~empty).view(1, -1, 1).float()
    out = torch.einsum("hqk,hkd->hqd", probs, vf)
    (out * dof).sum().backward()
    lse = torch.logsumexp(masked, dim=-1).detach()
    lse = torch.where(empty.view(1, -1), torch.full_like(lse, float("inf")), lse)
    delta = (out.detach() * dof).sum(dim=-1)
    return kf.grad, vf.grad, lse, delta


def _check(name, got, ref, live):
    """`got` / `ref` are `[..., seqlen_k, D]`; `live` is the `[seqlen_q, seqlen_k]` band."""
    assert not torch.isnan(got).any(), f"{name}: {int(torch.isnan(got).sum())} NaN (unwritten or poisoned)"
    scale = max(1.0, ref.abs().max().item())
    err = (got.float() - ref).abs().max().item()
    assert err <= REL_TOL * scale, f"{name}: max abs error {err:.3e} against ref scale {scale:.3e}"
    # A KV column no q row sees must be exactly zero: it is the one place an
    # unmasked or unwalked tile shows up as a wrong value rather than noise.
    # Taken from the band, not from `ref == 0`: a live column can have an
    # analytically zero dK (a lone live key has `P = 1`, so `dS = 0`), which
    # the kernel reproduces only to rounding.
    dead = ~live.any(dim=0)
    if dead.any():
        stray = got.float()[..., dead, :].abs().max().item()
        assert stray == 0.0, f"{name}: KV rows seen by no q row carry {stray:.3e}, expected 0"


def _rand(*shape):
    return torch.randn(shape, device="cuda", dtype=DTYPE)


@pytest.mark.parametrize("head_dim, mfma_rows", BUILDS, ids=lambda b: f"{b}")
@pytest.mark.parametrize("p", (0, 1), ids=lambda p: f"p{p}")
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w[0]}_{w[1]}")
@pytest.mark.parametrize("shape", DENSE_SHAPES, ids=lambda s: f"sq{s[0]}_sk{s[1]}")
def test_dense(head_dim, mfma_rows, p, window, shape):
    import fmha_abi_gfx1201 as abi

    fn = _build(head_dim, mfma_rows)
    seqlen_q, seqlen_k = shape
    batch = 2
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(zlib.crc32(repr((head_dim, p, window, shape)).encode()))
    q = _rand(batch, NUM_HEADS, seqlen_q, head_dim)
    k = _rand(batch, NUM_HEADS, seqlen_k, head_dim)
    v = _rand(batch, NUM_HEADS, seqlen_k, head_dim)
    do = _rand(batch, NUM_HEADS, seqlen_q, head_dim)
    live = _band(window, p, seqlen_q, seqlen_k, q.device)

    ref_dk, ref_dv, lse, delta = [], [], [], []
    for b in range(batch):
        r = _reference(q[b], k[b], v[b], do[b], live, sm_scale)
        ref_dk.append(r[0])
        ref_dv.append(r[1])
        lse.append(r[2])
        delta.append(r[3])
    ref_dk, ref_dv = torch.stack(ref_dk), torch.stack(ref_dv)
    lse = torch.cat(lse).contiguous()
    delta = torch.cat(delta).contiguous()

    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    bit = abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       lse,
       delta,
       batch,
       seqlen_q,
       seqlen_k=seqlen_k,
       scale=sm_scale,
       window=_wire_window(window),
       varlen=abi.varlen_dense(principal_diagonal=bit))
    torch.cuda.synchronize()
    _check("dV", dv, ref_dv, live)
    _check("dK", dk, ref_dk, live)


@pytest.mark.parametrize("head_dim, mfma_rows", BUILDS, ids=lambda b: f"{b}")
def test_dense_word_th_layout(head_dim, mfma_rows):
    """A dense word may select the TH LSE layout: (B*S, H) rows, same dK/dV as HT.

    `seqlen_q != NUM_HEADS`, so a row check that assumed HT would reject it.
    """
    import fmha_abi_gfx1201 as abi

    fn = _build(head_dim, mfma_rows)
    seqlen_q, seqlen_k = 48, 160
    window, p = (4, 4), 1
    batch = 2
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(zlib.crc32(repr((head_dim, "th")).encode()))
    q = _rand(batch, NUM_HEADS, seqlen_q, head_dim)
    k = _rand(batch, NUM_HEADS, seqlen_k, head_dim)
    v = _rand(batch, NUM_HEADS, seqlen_k, head_dim)
    do = _rand(batch, NUM_HEADS, seqlen_q, head_dim)
    live = _band(window, p, seqlen_q, seqlen_k, q.device)
    refs = [_reference(q[b], k[b], v[b], do[b], live, sm_scale) for b in range(batch)]
    ref_dk = torch.stack([r[0] for r in refs])
    ref_dv = torch.stack([r[1] for r in refs])
    lse_bhs = torch.cat([r[2] for r in refs]).reshape(batch, NUM_HEADS, seqlen_q)
    delta_bhs = torch.cat([r[3] for r in refs]).reshape(batch, NUM_HEADS, seqlen_q)
    outs = []
    for layout in (abi.VARLEN_LSE_LAYOUT_HT, abi.VARLEN_LSE_LAYOUT_TH):
        if layout == abi.VARLEN_LSE_LAYOUT_TH:
            lse = lse_bhs.transpose(1, 2).reshape(batch * seqlen_q, NUM_HEADS).contiguous()
            delta = delta_bhs.transpose(1, 2).reshape(batch * seqlen_q, NUM_HEADS).contiguous()
        else:
            lse = lse_bhs.reshape(batch * NUM_HEADS, seqlen_q).contiguous()
            delta = delta_bhs.reshape(batch * NUM_HEADS, seqlen_q).contiguous()
        dk = torch.full_like(k, float("nan"))
        dv = torch.full_like(v, float("nan"))
        fn(q,
           k,
           v,
           do,
           dk,
           dv,
           lse,
           delta,
           batch,
           seqlen_q,
           seqlen_k=seqlen_k,
           scale=sm_scale,
           window=_wire_window(window),
           varlen=abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT,
                                   lse_layout=layout))
        torch.cuda.synchronize()
        outs.append((dk, dv))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    _check("dV", outs[1][1], ref_dv, live)
    _check("dK", outs[1][0], ref_dk, live)


@pytest.mark.parametrize("head_dim, mfma_rows", BUILDS, ids=lambda b: f"{b}")
@pytest.mark.parametrize("p", (0, 1), ids=lambda p: f"p{p}")
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w[0]}_{w[1]}")
def test_varlen(head_dim, mfma_rows, p, window):
    """Packed batch, one `(seqlen_q, seqlen_k)` pair per sequence, bit in `varlen["bits"]`."""
    import fmha_abi_gfx1201 as abi

    fn = _build(head_dim, mfma_rows)
    sm_scale = 1.0 / math.sqrt(head_dim)
    lens_q = [s for s, _ in VARLEN_SHAPES]
    lens_k = [s for _, s in VARLEN_SHAPES]
    total_q, total_k = sum(lens_q), sum(lens_k)
    cu_q = torch.tensor([0] + list(_cumsum(lens_q)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(_cumsum(lens_k)), device="cuda", dtype=torch.int32)
    torch.manual_seed(zlib.crc32(repr((head_dim, p, window, "varlen")).encode()))
    q = _rand(1, NUM_HEADS, total_q, head_dim)
    k = _rand(1, NUM_HEADS, total_k, head_dim)
    v = _rand(1, NUM_HEADS, total_k, head_dim)
    do = _rand(1, NUM_HEADS, total_q, head_dim)

    ref_dk = torch.zeros(NUM_HEADS, total_k, head_dim, device="cuda")
    ref_dv = torch.zeros_like(ref_dk)
    lse = torch.empty(NUM_HEADS, total_q, device="cuda", dtype=torch.float32)
    delta = torch.empty_like(lse)
    for z, (sq, sk) in enumerate(VARLEN_SHAPES):
        qs = slice(int(cu_q[z]), int(cu_q[z + 1]))
        ks = slice(int(cu_k[z]), int(cu_k[z + 1]))
        live = _band(window, p, sq, sk, q.device)
        r = _reference(q[0, :, qs], k[0, :, ks], v[0, :, ks], do[0, :, qs], live, sm_scale)
        ref_dk[:, ks] = r[0]
        ref_dv[:, ks] = r[1]
        lse[:, qs] = r[2]
        delta[:, qs] = r[3]

    varlen = abi.varlen_compact(cu_q,
                                cu_k,
                                max(lens_q),
                                max(lens_k),
                                lse_tokens=total_q)
    if p:
        varlen["bits"] |= abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       lse,
       delta,
       1,
       max(lens_q),
       seqlen_k=max(lens_k),
       scale=sm_scale,
       window=_wire_window(window),
       varlen=varlen,
       num_seqlens=len(VARLEN_SHAPES))
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(VARLEN_SHAPES):
        ks = slice(int(cu_k[z]), int(cu_k[z + 1]))
        tag = f"seq {z} (sq={sq}, sk={sk})"
        live = _band(window, p, sq, sk, q.device)
        _check(f"dV {tag}", dv[0, :, ks], ref_dv[:, ks], live)
        _check(f"dK {tag}", dk[0, :, ks], ref_dk[:, ks], live)


def test_dense_word_rejects_seqinfo():
    """A dense word reads no sequence info; handing it some is a caller error."""
    import fmha_abi_gfx1201 as abi

    fn = _build(*BUILDS[0])
    t = _rand(1, NUM_HEADS, 8, BUILDS[0][0])
    lse = torch.zeros(NUM_HEADS, 8, device="cuda", dtype=torch.float32)
    varlen = abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)
    varlen["seqinfo_k0"] = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="dense VarlenBits word"):
        fn(t,
           t,
           t,
           t,
           t.clone(),
           t.clone(),
           lse,
           lse.clone(),
           1,
           8,
           window=(0, 0),
           varlen=varlen)


def _cumsum(xs):
    acc = 0
    for x in xs:
        acc += x
        yield acc


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
