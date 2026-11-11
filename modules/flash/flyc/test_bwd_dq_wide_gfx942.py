#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fmha_bwd_dq_gfx942_wide (head dims 160..512), JIT, against fp64.

dQ (and dB) from torch's LSE and Delta, so this tests the dQ kernel alone:
gSWA windows with both principal diagonals, GQA, varlen and LSE layouts, bias
with and without dB, dropout against the host Philox, padded heads with NaN
slack (hdim_qk != hdim_vo too), and rows that see no key.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_bwd_dq_wide_gfx942.py
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

HEAD_DIMS = (160, 192, 224, 256, 384, 512)
DTYPE = torch.bfloat16
# Relative to the largest reference magnitude of the call.
REL_TOL = 1e-2


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False, dtype_str="bf16"):
    from fmha_bwd_dq_gfx942_wide import build_fmha_bwd_dq_gfx942_wide

    return build_fmha_bwd_dq_gfx942_wide(head_dim,
                                         causal_type=causal_type,
                                         dtype_str=dtype_str,
                                         bias=bias,
                                         dropout=dropout,
                                         padded_head=padded_head)


def _reference(q, k, v, do, scale, keep_mask=None, bias=None, keep=None, p=0.0):
    """fp64 `(dQ, dS, LSE, Delta, live)` over `[..., sq, D]`; LSE is +inf on dead rows."""
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    if keep_mask is not None:
        s = s.masked_fill(~keep_mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    live = torch.isfinite(lse)
    prob = torch.exp(s - lse[..., None]).nan_to_num(0.0)
    dp = do.double() @ v.double().transpose(-1, -2)
    pd = prob
    if keep is not None:
        pd = prob * keep / (1 - p)
        dp = dp * keep / (1 - p)
    o = pd @ v.double()
    delta = (do.double() * o).sum(-1)
    ds = prob * (dp - delta[..., None])
    dq = (ds @ k.double()) * scale
    lse = torch.where(live, lse, torch.full_like(lse, float("inf")))
    return dq, ds, lse.float(), delta.float(), live


def _check(tag, got, ref, live=None):
    """Within REL_TOL of the largest reference value; rows that see no key exactly 0."""
    assert not bool(torch.isnan(got).any()), f"{tag}: NaN"
    # Floored: a lone visible key makes dQ exactly 0, and dP - Delta then cancels in fp32.
    scale = max(1e-3, ref.abs().max().item())
    err = (got.double() - ref).abs().max().item() / scale
    assert err <= REL_TOL, f"{tag}: off by {err:.3e} of {scale:.3e}"
    if live is not None and bool((~live).any()):
        dead = got[..., ~live, :]
        assert int((dead != 0).sum()) == 0, f"{tag}: nonzero dQ in rows that see no key"


def _bottom_right(sq, sk, device):
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return j <= i + (sk - sq)


def _inputs(shape_q, shape_k, hvo=None, seed=0):
    torch.manual_seed(seed)
    *lead_k, sk, hqk = shape_k
    hvo = hqk if hvo is None else hvo
    q = torch.randn(shape_q, device="cuda", dtype=DTYPE)
    k = torch.randn(shape_k, device="cuda", dtype=DTYPE)
    v = torch.randn(*lead_k, sk, hvo, device="cuda", dtype=DTYPE)
    do = torch.randn(*shape_q[:-1], hvo, device="cuda", dtype=DTYPE)
    return q, k, v, do


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (777, 300), (1, 1001), (1000, 1000)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_dense(head_dim, shape, causal):
    """Full attention and bottom-right causal (`causal_type=2`), GQA 2:1."""
    sq, sk = shape
    b, h, hk = 2, 4, 2
    q, k, v, do = _inputs((b, h, sq, head_dim), (b, hk, sk, head_dim), seed=sq + sk + head_dim)
    kr, vr = k.repeat_interleave(h // hk, 1), v.repeat_interleave(h // hk, 1)
    mask = _bottom_right(sq, sk, q.device) if causal else None
    ref, _, lse, delta, live = _reference(q, kr, vr, do, head_dim ** -0.5, keep_mask=mask)
    dq = torch.full_like(q, float("nan"))
    _build(head_dim, causal_type=2 if causal else 0)(q,
                                                     k,
                                                     v,
                                                     do,
                                                     dq,
                                                     lse.view(b * h, sq),
                                                     delta.view(b * h, sq),
                                                     b,
                                                     sq,
                                                     seqlen_k=sk)
    torch.cuda.synchronize()
    _check(f"hd={head_dim} {shape} causal={causal}", dq, ref, live[0, 0])


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape_name", list(DENSE_SHAPES))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_window(head_dim, shape_name, window_name, p):
    """gSWA windows from either principal diagonal. Dead rows get a finite LSE, so
    only the mask can zero their dQ."""
    sq, sk = DENSE_SHAPES[shape_name]
    window = WINDOWS[window_name]
    b, h = 2, 2
    q, k, v, do = _inputs((b, h, sq, head_dim), (b, h, sk, head_dim), seed=sq * 1000 + sk)
    keep = _keep_mask(window, p, sq, sk, q.device)
    ref, _, lse, delta, live = _reference(q, k, v, do, head_dim ** -0.5, keep_mask=keep)
    lse = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    dq = torch.full_like(q, float("nan"))
    _build(head_dim, causal_type=3)(q,
                                    k,
                                    v,
                                    do,
                                    dq,
                                    lse.view(b * h, sq),
                                    delta.view(b * h, sq),
                                    b,
                                    sq,
                                    seqlen_k=sk,
                                    window=window,
                                    principal_diagonal=PRINCIPAL_DIAGONAL[p])
    torch.cuda.synchronize()
    _check(f"hd={head_dim} {shape_name} {window_name} p={p}", dq, ref, live[0, 0])


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
    q, k, v, do = _inputs((1, h, cu_q[-1], head_dim), (1, h, cu_k[-1], head_dim), seed=head_dim + p)
    lse = torch.empty(h, cu_q[-1], device="cuda")
    delta = torch.empty_like(lse)
    refs = []
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        keep = _keep_mask(window, p, sq, sk, q.device)
        ref, _, lz, dz, live = _reference(q[0, :, qs], k[0, :, ks], v[0, :, ks], do[0, :, qs], head_dim ** -0.5,
                                          keep_mask=keep)
        lse[:, qs], delta[:, qs] = lz, dz
        refs.append((qs, ref, live))
    varlen = {
        "bits": abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                abi.VARLEN_COMPACT_SIDE,
                                principal_diagonal=PRINCIPAL_DIAGONAL[p]),
        "seqinfo_q0": torch.tensor(cu_q, device="cuda", dtype=torch.int32),
        "seqinfo_k0": torch.tensor(cu_k, device="cuda", dtype=torch.int32),
        "max_seqlen_q": max(a for a, _ in VARLEN_SEQS),
        "max_seqlen_k": max(c for _, c in VARLEN_SEQS),
        "lse_tokens": cu_q[-1],
    }
    dq = torch.full_like(q, float("nan"))
    _build(head_dim, causal_type=3)(q,
                                    k,
                                    v,
                                    do,
                                    dq,
                                    lse,
                                    delta,
                                    1,
                                    varlen["max_seqlen_q"],
                                    seqlen_k=varlen["max_seqlen_k"],
                                    window=window,
                                    varlen=varlen,
                                    num_seqlens=len(VARLEN_SEQS))
    torch.cuda.synchronize()
    for z, (qs, ref, live) in enumerate(refs):
        _check(f"hd={head_dim} varlen seq{z}={VARLEN_SEQS[z]} p={p}", dq[0, :, qs], ref, live[0])


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("lse_layout", ["ht", "th"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen(head_dim, layout, lse_layout):
    """Compact packs sequences into (1, H, T, D); padded keeps (B, H, max, D). GQA 2:1."""
    h, hk = 4, 2
    lq = [a for a, _ in SEQS]
    lk = [c for _, c in SEQS]
    n = len(SEQS)
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device="cuda", dtype=torch.int32)
    lse_bits = abi.VARLEN_LSE_LAYOUT_HT if lse_layout == "ht" else abi.VARLEN_LSE_LAYOUT_TH
    if layout == "compact":
        tq, tk = sum(lq), sum(lk)
        q, k, v, do = _inputs((1, h, tq, head_dim), (1, hk, tk, head_dim), seed=head_dim)
        varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = 1, n
        lse = torch.zeros((h, tq) if lse_layout == "ht" else (tq, h), device="cuda")
    else:
        tq = max(lq)
        q, k, v, do = _inputs((n, h, tq, head_dim), (n, hk, max(lk), head_dim), seed=head_dim)
        varlen = abi.varlen_padded(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = n, 0
        lse = torch.zeros((n * h, tq) if lse_layout == "ht" else (n * tq, h), device="cuda")
    delta = torch.zeros_like(lse)
    scale = head_dim ** -0.5
    refs = []
    for z, (sq, sk) in enumerate(SEQS):
        if layout == "compact":
            qs = (0, slice(None), slice(int(cu_q[z]), int(cu_q[z + 1])))
            ks = (0, slice(None), slice(int(cu_k[z]), int(cu_k[z + 1])))
        else:
            qs = (z, slice(None), slice(0, sq))
            ks = (z, slice(None), slice(0, sk))
        kz = k[ks].repeat_interleave(h // hk, 0)
        vz = v[ks].repeat_interleave(h // hk, 0)
        ref, _, lz, dz, live = _reference(q[qs], kz, vz, do[qs], scale)
        if layout == "compact":
            rows = slice(int(cu_q[z]), int(cu_q[z + 1]))
            if lse_layout == "ht":
                lse[:, rows], delta[:, rows] = lz, dz
            else:
                lse[rows], delta[rows] = lz.T, dz.T
        elif lse_layout == "ht":
            lse[z * h:(z + 1) * h, :sq], delta[z * h:(z + 1) * h, :sq] = lz, dz
        else:
            lse[z * tq:z * tq + sq], delta[z * tq:z * tq + sq] = lz.T, dz.T
        refs.append((qs, ref))
    dq = torch.full_like(q, float("nan"))
    _build(head_dim)(q, k, v, do, dq, lse, delta, batch, max(lq), seqlen_k=max(lk), num_seqlens=nseq, varlen=varlen)
    torch.cuda.synchronize()
    for z, (qs, ref) in enumerate(refs):
        _check(f"hd={head_dim} {layout} {lse_layout} seq{z}={SEQS[z]}", dq[qs], ref)


@pytest.mark.parametrize("with_db", [True, False])
@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001), (128, 64)])
@pytest.mark.parametrize("head_dim", (160, 256, 512))
def test_bias(head_dim, shape, with_db):
    """dB = dS. Odd pitches start bias rows mid-dword; an odd slab ends mid-dword.
    A bias that needs no gradient passes a null dB with zero strides."""
    sq, sk = shape
    b, h = 2, 3
    q, k, v, do = _inputs((b, h, sq, head_dim), (b, h, sk, head_dim), seed=sq + sk)
    bias = torch.randn(b, h, sq, sk, device="cuda", dtype=DTYPE)
    ref, ref_ds, lse, delta, _ = _reference(q, k, v, do, head_dim ** -0.5, bias=bias)
    dq = torch.full_like(q, float("nan"))
    if with_db:
        # Its own layout, padded rows, to catch borrowed strides and row overruns.
        dbias = torch.full((b, h, sq, sk + 5), float("nan"), device="cuda", dtype=DTYPE)[..., :sk]
    else:
        # All-zero (batch, head, row) strides: the kernel must not store.
        dbias = torch.full((sk,), float("nan"), device="cuda", dtype=DTYPE).as_strided((b, h, sq, sk), (0, 0, 0, 1))
    _build(head_dim, bias=True)(q,
                                k,
                                v,
                                do,
                                dq,
                                lse.view(b * h, sq),
                                delta.view(b * h, sq),
                                b,
                                sq,
                                seqlen_k=sk,
                                bias=bias,
                                dbias=dbias)
    torch.cuda.synchronize()
    _check(f"hd={head_dim} bias {shape} dQ", dq, ref)
    if with_db:
        _check(f"hd={head_dim} bias {shape} dB", dbias, ref_ds)
        base = dbias.as_strided((b, h, sq, sk + 5), dbias.stride())
        assert bool(torch.isnan(base[..., sk:]).all()), "dB written past seqlen_k"
    else:
        assert bool(torch.isnan(dbias).all()), "dB stored with all-zero dB strides"


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (777, 300)])
@pytest.mark.parametrize("head_dim", (192, 256, 384))
def test_dropout(head_dim, shape, causal):
    """The mask must be the Triton kernels': offset = base + plane*sq*stride + row*stride + col//4."""
    sq, sk = shape
    b, h, p = 2, 3, 0.3
    seed, off1, off2 = 0x123456789ABCDEF0, 1000, 77
    q, k, v, do = _inputs((b, h, sq, head_dim), (b, h, sk, head_dim), seed=sq * sk)
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off1 + off2) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = _philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    keep = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk)
    mask = _bottom_right(sq, sk, q.device) if causal else None
    ref, _, lse, delta, live = _reference(q, k, v, do, head_dim ** -0.5, keep_mask=mask, keep=keep, p=p)
    dq = torch.full_like(q, float("nan"))
    _build(head_dim, causal_type=2 if causal else 0, dropout=True)(
        q,
        k,
        v,
        do,
        dq,
        lse.view(b * h, sq),
        delta.view(b * h, sq),
        b,
        sq,
        seqlen_k=sk,
        dropout_p=p,
        philox_seed=seed,
        philox_offset1=torch.tensor([off1], dtype=torch.int64, device="cuda"),
        philox_offset2=off2)
    torch.cuda.synchronize()
    _check(f"hd={head_dim} dropout {shape} causal={causal}", dq, ref, live[0, 0])


# (rung, hdim_qk, hdim_vo): primes and odd sizes above the rung below, and QK != VO.
PADDED = [(160, 149, 149), (192, 179, 179), (224, 211, 211), (256, 241, 241), (192, 192, 128), (160, 24, 152),
          (256, 256, 200), (256, 136, 256), (384, 263, 263), (384, 384, 300), (512, 449, 449), (512, 392, 512)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rung,hqk,hvo", PADDED)
def test_padded_head(rung, hqk, hvo, causal):
    """Columns past hdim are NaN in Q, K, V and dO; none may reach a stored dQ column."""
    b, h, sq, sk = 2, 3, 257, 519
    if causal:
        sq, sk = sk, sq                 # bottom-right with dead rows
    torch.manual_seed(hqk * 1000 + hvo)
    q = _with_nan_slack(b, h, sq, hqk)
    k = _with_nan_slack(b, h, sk, hqk)
    v = _with_nan_slack(b, h, sk, hvo)
    do = _with_nan_slack(b, h, sq, hvo)
    mask = _bottom_right(sq, sk, q.device) if causal else None
    ref, _, lse, delta, live = _reference(q, k, v, do, hqk ** -0.5, keep_mask=mask)
    dq = _with_nan_slack(b, h, sq, hqk)
    dq.fill_(float("nan"))
    _build(rung, causal_type=2 if causal else 0, padded_head=True)(q,
                                                                   k,
                                                                   v,
                                                                   do,
                                                                   dq,
                                                                   lse.view(b * h, sq),
                                                                   delta.view(b * h, sq),
                                                                   b,
                                                                   sq,
                                                                   seqlen_k=sk)
    torch.cuda.synchronize()
    _check(f"rung={rung} hqk={hqk} hvo={hvo} causal={causal}", dq, ref, live[0, 0])


@pytest.mark.parametrize("head_dim", (160, 256, 512))
def test_layouts(head_dim):
    """Every tensor its own layout: BSHD Q and dQ, BHSD K, transposed-batch V and dO."""
    b, h, sq, sk = 2, 3, 200, 333
    torch.manual_seed(head_dim)
    q = torch.randn(b, sq, h, head_dim, device="cuda", dtype=DTYPE).transpose(1, 2)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn(h, b, sk, head_dim, device="cuda", dtype=DTYPE).transpose(0, 1)
    do = torch.randn(sq, b, h, head_dim, device="cuda", dtype=DTYPE).permute(1, 2, 0, 3)
    dq = torch.full((h, sq, b, head_dim), float("nan"), device="cuda", dtype=DTYPE).permute(2, 0, 1, 3)
    ref, _, lse, delta, _ = _reference(q, k, v, do, head_dim ** -0.5)
    _build(head_dim)(q, k, v, do, dq, lse.view(b * h, sq), delta.view(b * h, sq), b, sq, seqlen_k=sk)
    torch.cuda.synchronize()
    _check(f"hd={head_dim} layouts", dq, ref)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", (192, 512))
def test_fp16(head_dim, causal):
    """f16 inputs and outputs; dS rounds to nearest even through v_cvt_f16_f32."""
    sq, sk = 333, 555
    b, h = 2, 2
    torch.manual_seed(head_dim)
    q, k, v, do = (torch.randn(b, h, n, head_dim, device="cuda", dtype=torch.float16) for n in (sq, sk, sk, sq))
    mask = _bottom_right(sq, sk, q.device) if causal else None
    ref, _, lse, delta, live = _reference(q, k, v, do, head_dim ** -0.5, keep_mask=mask)
    dq = torch.full_like(q, float("nan"))
    _build(head_dim, causal_type=2 if causal else 0, dtype_str="f16")(q,
                                                                     k,
                                                                     v,
                                                                     do,
                                                                     dq,
                                                                     lse.view(b * h, sq),
                                                                     delta.view(b * h, sq),
                                                                     b,
                                                                     sq,
                                                                     seqlen_k=sk)
    torch.cuda.synchronize()
    _check(f"hd={head_dim} f16 causal={causal}", dq, ref, live[0, 0])


@pytest.mark.parametrize("head_dim", (192, 384))
def test_large_logits(head_dim):
    """sm_scale 1.2 on N(0, 1): a sharp softmax, where an exponent error shows."""
    b, h, sq, sk = 1, 2, 256, 512
    q, k, v, do = _inputs((b, h, sq, head_dim), (b, h, sk, head_dim), seed=7)
    q, k = q * 0.25, k * 0.25
    ref, _, lse, delta, _ = _reference(q, k, v, do, 1.2)
    dq = torch.full_like(q, float("nan"))
    _build(head_dim)(q, k, v, do, dq, lse.view(b * h, sq), delta.view(b * h, sq), b, sq, seqlen_k=sk, scale=1.2)
    torch.cuda.synchronize()
    _check(f"hd={head_dim} large logits", dq, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
