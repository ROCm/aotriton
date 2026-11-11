#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fmha_bwd_dkdv_gfx942_wide (head dims 160..512), JIT, against fp64.

gSWA windows with both principal diagonals (L negative, R past seqlen_k, key
columns no row sees), causal and GQA, varlen and LSE layouts, bias, dropout
against the host Philox, padded heads with NaN slack (hdim_qk != hdim_vo too),
rows that see no key (LSE = +inf), and an unbiased P/dS rounding. LSE and Delta
come from the reference, so this tests the backward alone.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_bwd_dkdv_wide_gfx942.py
"""

import functools
import os
import sys
from pathlib import Path

import pytest

_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(
    not _KERNEL_ROOT
    or not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"),
    reason="needs a gfx942 device and AOTRITON_FLYDSL_KERNEL_ROOT",
)

import fmha_abi_gfx1201 as abi  # noqa: E402
import philox  # noqa: E402
from test_fwd_features_gfx942 import _philox_4x32_10, _with_nan_slack  # noqa: E402
from test_fwd_principal_diagonal_gfx942 import (  # noqa: E402
    DENSE_SHAPES,
    PRINCIPAL_DIAGONAL,
    VARLEN_SEQS,
    WINDOWS,
    _keep_mask,
)

HEAD_DIMS = (160, 192, 224, 256)
# 384 and 512 run fewer cases: same body, single RM buffer, two passes at 512.
XWIDE = (384, 512)
DTYPE = torch.bfloat16
# Max abs error against fp64, relative to the reference's max magnitude (floored at 1):
# bf16 inputs and bf16 P/dS operands leave a few 1e-3.
REL_TOL = 1e-2


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False, kv_subs=None):
    from fmha_bwd_dkdv_gfx942_wide import build_fmha_bwd_dkdv_gfx942_wide

    return build_fmha_bwd_dkdv_gfx942_wide(head_dim,
                                           causal_type=causal_type,
                                           bias=bias,
                                           dropout=dropout,
                                           padded_head=padded_head,
                                           kv_subs=kv_subs)


def _reference(q, k, v, do, scale, keep_mask=None, bias=None, keep=None, p=0.0):
    """fp64 `(dK, dV, LSE, Delta)` over `[..., s, D]`; K/V already expanded to Q's heads.

    A row that sees no key has LSE +inf and contributes nothing; Delta is
    rowsum(dO * O) with O rounded to the input dtype, as the forward stores it.
    """
    kd = k.double().requires_grad_(True)
    vd = v.double().requires_grad_(True)
    s = (q.double() @ kd.transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    if keep_mask is not None:
        s = s.masked_fill(~keep_mask, float("-inf"))
    w = torch.softmax(s, dim=-1).nan_to_num()
    if keep is not None:
        w = w * keep / (1 - p)
    o = w @ vd
    (o * do.double()).sum().backward()
    lse = torch.logsumexp(s.detach(), dim=-1)
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float("inf")))
    delta = (o.detach().to(q.dtype).double() * do.double()).sum(-1)
    return kd.grad, vd.grad, lse.float(), delta.float()


def _fold(g, hk):
    """Sum a `[B, H, s, D]` gradient over the query heads of each key head."""
    b, h = g.shape[:2]
    return g.view(b, hk, h // hk, *g.shape[2:]).sum(2)


def _check(tag, got, ref, dead_cols=None):
    """`got` / `ref` are `[..., sk, D]`; keys no row sees must be exactly 0."""
    assert not bool(torch.isnan(got).any()), f"{tag}: {int(torch.isnan(got).sum())} NaN"
    scale = max(1.0, ref.abs().max().item())
    err = (got.double() - ref).abs().max().item()
    assert err <= REL_TOL * scale, f"{tag}: max abs error {err:.3e} against ref scale {scale:.3e}"
    if dead_cols is not None and bool(dead_cols.any()):
        stray = got.double()[..., dead_cols, :].abs().max().item()
        assert stray == 0.0, f"{tag}: keys seen by no row carry {stray:.3e}, expected 0"


def _rand(*shape):
    return torch.randn(shape, device="cuda", dtype=DTYPE)


def _run(fn, q, k, v, do, lse, delta, b, sq, sk, **kw):
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    fn(q, k, v, do, dk, dv, lse, delta, b, sq, seqlen_k=sk, **kw)
    torch.cuda.synchronize()
    return dk, dv


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape_name", list(DENSE_SHAPES))
@pytest.mark.parametrize("head_dim", HEAD_DIMS + XWIDE)
def test_window(head_dim, shape_name, window_name, p):
    """gSWA (`causal_type=3`) windows measured from either principal diagonal."""
    if head_dim in XWIDE and (p or window_name not in ("4_4", "neg1_3", "sentinel_br")):
        pytest.skip("a window subset at the extra-wide rungs")
    sq, sk = DENSE_SHAPES[shape_name]
    window = WINDOWS[window_name]
    b, h = 2, 2
    torch.manual_seed(sq * 1000 + sk)
    q, do = _rand(b, h, sq, head_dim), _rand(b, h, sq, head_dim)
    k, v = _rand(b, h, sk, head_dim), _rand(b, h, sk, head_dim)
    keep = _keep_mask(window, p, sq, sk, q.device)
    rk, rv, lse, delta = _reference(q, k, v, do, head_dim ** -0.5, keep_mask=keep)
    dk, dv = _run(_build(head_dim, causal_type=3),
                  q,
                  k,
                  v,
                  do,
                  lse.view(b * h, sq),
                  delta.view(b * h, sq),
                  b,
                  sq,
                  sk,
                  window=window,
                  principal_diagonal=PRINCIPAL_DIAGONAL[p])
    dead = ~keep.any(dim=0)
    tag = f"hd={head_dim} {shape_name} {window_name} p={p}"
    _check(f"dV {tag}", dv, rv, dead)
    _check(f"dK {tag}", dk, rk, dead)


def _bottom_right(sq, sk, device):
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return j <= i + (sk - sq)


@pytest.mark.parametrize("kv_subs", [None, 1])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (777, 300), (1, 1001), (1000, 1000)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS + XWIDE)
def test_dense(head_dim, shape, causal, kv_subs):
    """Full attention and bottom-right causal (`causal_type=2`), GQA 2:1 summed per key head."""
    if kv_subs is not None and head_dim != 192:
        pytest.skip("the other kv_subs runs at one rung")
    sq, sk = shape
    b, h, hk = 2, 4, 2
    torch.manual_seed(sq + sk + head_dim)
    q, do = _rand(b, h, sq, head_dim), _rand(b, h, sq, head_dim)
    k, v = _rand(b, hk, sk, head_dim), _rand(b, hk, sk, head_dim)
    keep = _bottom_right(sq, sk, q.device) if causal else None
    rk, rv, lse, delta = _reference(q,
                                    k.repeat_interleave(h // hk, 1),
                                    v.repeat_interleave(h // hk, 1),
                                    do,
                                    head_dim ** -0.5,
                                    keep_mask=keep)
    dk, dv = _run(_build(head_dim, causal_type=2 if causal else 0, kv_subs=kv_subs),
                  q,
                  k,
                  v,
                  do,
                  lse.view(b * h, sq),
                  delta.view(b * h, sq),
                  b,
                  sq,
                  sk)
    dead = ~keep.any(dim=0) if causal else None
    tag = f"hd={head_dim} {shape} causal={causal} kv_subs={kv_subs}"
    _check(f"dV {tag}", dv, _fold(rv, hk), dead)
    _check(f"dK {tag}", dk, _fold(rk, hk), dead)


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_window(head_dim, p):
    """Packed sequences with one `seqlen_k - seqlen_q` each; `p` rides in varlen bits."""
    window = WINDOWS["4_4"]
    h = 2
    cu_q, cu_k = [0], [0]
    for a, c in VARLEN_SEQS:
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + c)
    torch.manual_seed(head_dim + p)
    q, do = _rand(1, h, cu_q[-1], head_dim), _rand(1, h, cu_q[-1], head_dim)
    k, v = _rand(1, h, cu_k[-1], head_dim), _rand(1, h, cu_k[-1], head_dim)
    lse = torch.empty(h, cu_q[-1], device="cuda")
    delta = torch.empty_like(lse)
    refs = []
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        keep = _keep_mask(window, p, sq, sk, q.device)
        rk, rv, lse[:, qs], delta[:, qs] = _reference(q[0, :, qs],
                                                      k[0, :, ks],
                                                      v[0, :, ks],
                                                      do[0, :, qs],
                                                      head_dim ** -0.5,
                                                      keep_mask=keep)
        refs.append((rk, rv, ~keep.any(dim=0)))
    lq = [a for a, _ in VARLEN_SEQS]
    lk = [c for _, c in VARLEN_SEQS]
    varlen = {
        "bits": abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                abi.VARLEN_COMPACT_SIDE,
                                principal_diagonal=PRINCIPAL_DIAGONAL[p]),
        "seqinfo_q0": torch.tensor(cu_q, device="cuda", dtype=torch.int32),
        "seqinfo_k0": torch.tensor(cu_k, device="cuda", dtype=torch.int32),
        "max_seqlen_q": max(lq),
        "max_seqlen_k": max(lk),
        "lse_tokens": cu_q[-1],
    }
    dk, dv = _run(_build(head_dim, causal_type=3),
                  q,
                  k,
                  v,
                  do,
                  lse,
                  delta,
                  1,
                  max(lq),
                  max(lk),
                  window=window,
                  varlen=varlen,
                  num_seqlens=len(VARLEN_SEQS))
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        ks = slice(cu_k[z], cu_k[z + 1])
        rk, rv, dead = refs[z]
        tag = f"hd={head_dim} varlen seq{z}=({sq},{sk}) p={p}"
        _check(f"dV {tag}", dv[0, :, ks], rv, dead)
        _check(f"dK {tag}", dk[0, :, ks], rk, dead)


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("lse_layout", ["ht", "th"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS + XWIDE)
def test_varlen(head_dim, layout, lse_layout):
    """Compact packs sequences into (1, H, T, D); padded keeps (B, H, max, D). GQA 2:1."""
    h, hk = 4, 2
    lq = [a for a, _ in SEQS]
    lk = [c for _, c in SEQS]
    n = len(SEQS)
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device="cuda", dtype=torch.int32)
    lse_bits = abi.VARLEN_LSE_LAYOUT_HT if lse_layout == "ht" else abi.VARLEN_LSE_LAYOUT_TH
    torch.manual_seed(head_dim)
    if layout == "compact":
        tq, tk = sum(lq), sum(lk)
        q, do = _rand(1, h, tq, head_dim), _rand(1, h, tq, head_dim)
        k, v = _rand(1, hk, tk, head_dim), _rand(1, hk, tk, head_dim)
        varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = 1, n
        shape = (h, tq) if lse_layout == "ht" else (tq, h)
    else:
        tq = max(lq)
        q, do = _rand(n, h, tq, head_dim), _rand(n, h, tq, head_dim)
        k, v = _rand(n, hk, max(lk), head_dim), _rand(n, hk, max(lk), head_dim)
        varlen = abi.varlen_padded(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = n, 0
        shape = (n * h, tq) if lse_layout == "ht" else (n * tq, h)
    lse = torch.zeros(shape, device="cuda")
    delta = torch.zeros(shape, device="cuda")
    scale = head_dim ** -0.5
    refs = []
    for z, (sq, sk) in enumerate(SEQS):
        if layout == "compact":
            qz, doz = (t[0, :, cu_q[z]:cu_q[z + 1]] for t in (q, do))
            kz, vz = (t[0, :, cu_k[z]:cu_k[z + 1]] for t in (k, v))
        else:
            qz, doz, kz, vz = q[z, :, :sq], do[z, :, :sq], k[z, :, :sk], v[z, :, :sk]
        rk, rv, rl, rd = _reference(qz, kz.repeat_interleave(h // hk, 0), vz.repeat_interleave(h // hk, 0), doz, scale)
        refs.append((_fold(rk[None], hk)[0], _fold(rv[None], hk)[0]))
        if layout == "compact":
            rows = slice(int(cu_q[z]), int(cu_q[z + 1]))
            if lse_layout == "ht":
                lse[:, rows], delta[:, rows] = rl, rd
            else:
                lse[rows], delta[rows] = rl.T, rd.T
        elif lse_layout == "ht":
            lse[z * h:(z + 1) * h, :sq], delta[z * h:(z + 1) * h, :sq] = rl, rd
        else:
            lse[z * tq:z * tq + sq], delta[z * tq:z * tq + sq] = rl.T, rd.T
    dk, dv = _run(_build(head_dim), q, k, v, do, lse, delta, batch, max(lq), max(lk), num_seqlens=nseq, varlen=varlen)
    for z, (sq, sk) in enumerate(SEQS):
        if layout == "compact":
            gk, gv = (t[0, :, cu_k[z]:cu_k[z + 1]] for t in (dk, dv))
        else:
            gk, gv = dk[z, :, :sk], dv[z, :, :sk]
        tag = f"hd={head_dim} {layout} {lse_layout} seq{z}=({sq},{sk})"
        _check(f"dV {tag}", gv, refs[z][1])
        _check(f"dK {tag}", gk, refs[z][0])


@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001)])
@pytest.mark.parametrize("head_dim", (160, 256, 384))
def test_bias(head_dim, shape):
    """A (B, H, Sq, Sk) bias, odd pitches included; bias magnitudes up to ~16."""
    sq, sk = shape
    b, h = 2, 3
    torch.manual_seed(sq + sk)
    q, do = _rand(b, h, sq, head_dim), _rand(b, h, sq, head_dim)
    k, v = _rand(b, h, sk, head_dim), _rand(b, h, sk, head_dim)
    bias = (torch.randn(b, h, sq, sk, device="cuda") * 4).to(DTYPE)
    rk, rv, lse, delta = _reference(q, k, v, do, head_dim ** -0.5, bias=bias)
    dk, dv = _run(_build(head_dim, bias=True),
                  q,
                  k,
                  v,
                  do,
                  lse.view(b * h, sq),
                  delta.view(b * h, sq),
                  b,
                  sq,
                  sk,
                  bias=bias)
    _check(f"dV hd={head_dim} bias {shape}", dv, rv)
    _check(f"dK hd={head_dim} bias {shape}", dk, rk)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (777, 300)])
@pytest.mark.parametrize("head_dim", (192, 256, 512))
def test_dropout(head_dim, shape, causal):
    """The forward's mask: offset = base + plane*sq*stride + row*stride + col//4."""
    sq, sk = shape
    b, h, p = 2, 3, 0.3
    seed, off1, off2 = 0x123456789ABCDEF0, 1000, 77
    torch.manual_seed(sq * sk)
    q, do = _rand(b, h, sq, head_dim), _rand(b, h, sq, head_dim)
    k, v = _rand(b, h, sk, head_dim), _rand(b, h, sk, head_dim)
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off1 + off2) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = _philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    keep = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk)
    mask = _bottom_right(sq, sk, q.device) if causal else None
    rk, rv, lse, delta = _reference(q, k, v, do, head_dim ** -0.5, keep_mask=mask, keep=keep, p=p)
    dk, dv = _run(_build(head_dim, causal_type=2 if causal else 0, dropout=True),
                  q,
                  k,
                  v,
                  do,
                  lse.view(b * h, sq),
                  delta.view(b * h, sq),
                  b,
                  sq,
                  sk,
                  dropout_p=p,
                  philox_seed=seed,
                  philox_offset1=torch.tensor([off1], dtype=torch.int64, device="cuda"),
                  philox_offset2=off2)
    tag = f"hd={head_dim} dropout {shape} causal={causal}"
    _check(f"dV {tag}", dv, rv)
    _check(f"dK {tag}", dk, rk)


# (rung, hdim_qk, hdim_vo): primes and odd sizes above the rung below, and QK != VO.
PADDED = [(160, 149, 149), (192, 179, 179), (224, 211, 211), (256, 241, 241), (192, 192, 128), (160, 24, 152),
          (256, 256, 200), (384, 301, 301), (512, 449, 449), (512, 512, 392)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rung,hqk,hvo", PADDED)
def test_padded_head(rung, hqk, hvo, causal):
    """Columns past hdim are NaN in Q, K, V and dO; none may reach a stored dK/dV column."""
    b, h, sq, sk = 2, 3, 257, 519
    if causal:
        sq, sk = sk, sq                 # bottom-right with dead rows
    torch.manual_seed(hqk * 1000 + hvo)
    q = _with_nan_slack(b, h, sq, hqk)
    do = _with_nan_slack(b, h, sq, hvo)
    k = _with_nan_slack(b, h, sk, hqk)
    v = _with_nan_slack(b, h, sk, hvo)
    mask = _bottom_right(sq, sk, q.device) if causal else None
    rk, rv, lse, delta = _reference(q, k, v, do, hqk ** -0.5, keep_mask=mask)
    dk = _with_nan_slack(b, h, sk, hqk)
    dv = _with_nan_slack(b, h, sk, hvo)
    dk.fill_(float("nan"))
    dv.fill_(float("nan"))
    _build(rung, causal_type=2 if causal else 0, padded_head=True)(q,
                                                                    k,
                                                                    v,
                                                                    do,
                                                                    dk,
                                                                    dv,
                                                                    lse.view(b * h, sq),
                                                                    delta.view(b * h, sq),
                                                                    b,
                                                                    sq,
                                                                    seqlen_k=sk)
    torch.cuda.synchronize()
    tag = f"rung={rung} hqk={hqk} hvo={hvo} causal={causal}"
    _check(f"dV {tag}", dv, rv)
    _check(f"dK {tag}", dk, rk)


@pytest.mark.parametrize("head_dim", HEAD_DIMS + XWIDE)
def test_one_signed_unbiased(head_dim):
    """U[0, 1) inputs: a truncated P biases dV (numerical lessons, #16); bound eps/8.

    dK's signed error is dS's bf16 rounding, which dK's cancellation magnifies
    (~1e-3 of |dK| here): it is held to an fp64 model that rounds only dS."""
    sq, sk = 143, 579
    scale = head_dim ** -0.5
    torch.manual_seed(head_dim)
    q, k, v, do = (torch.rand(2, 2, s, head_dim, device="cuda", dtype=DTYPE) for s in (sq, sk, sk, sq))
    rk, rv, lse, delta = _reference(q, k, v, do, scale)
    dk, dv = _run(_build(head_dim), q, k, v, do, lse.view(4, sq), delta.view(4, sq), 2, sq, sk)
    signed = ((dv.double() - rv).sum() / rv.abs().sum()).item()
    assert abs(signed) <= 2 ** -10, f"hd={head_dim} dV: biased by {signed:+.2e} of its magnitude"
    p = torch.exp((q.double() @ k.double().transpose(-1, -2)) * scale - lse.double()[..., None])
    ds = p * (do.double() @ v.double().transpose(-1, -2) - delta.double()[..., None])
    model = (ds.to(DTYPE).double().transpose(-1, -2) @ q.double()) * scale
    signed = ((dk.double() - model).sum() / rk.abs().sum()).item()
    assert abs(signed) <= 2 ** -10, f"hd={head_dim} dK: biased by {signed:+.2e} beyond bf16 dS rounding"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
