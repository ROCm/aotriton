#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""fmha_fwd_gfx942_wide (head dims 160..256), JIT, against fp64.

The narrow kernel's two test files at the wide rungs: gSWA windows with both
principal diagonals, varlen layouts and LSE layouts, bias, dropout against the
host Philox, padded heads with NaN slack (hdim_qk != hdim_vo too), rows that
see no key, a null LSE, and an unbiased P rounding.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_fwd_wide_gfx942.py
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
DTYPE = torch.bfloat16
ATOL = 1.5e-2
LSE_ATOL = 1e-4


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False):
    from fmha_fwd_gfx942_wide import build_fmha_fwd_gfx942_wide

    return build_fmha_fwd_gfx942_wide(head_dim,
                                      causal_type=causal_type,
                                      bias=bias,
                                      dropout=dropout,
                                      padded_head=padded_head)


def _reference(q, k, v, scale, bias=None, keep=None, p=0.0, keep_mask=None):
    """fp64 `(O, LSE)` over `[..., sq, D]`; a row that sees no key has O = 0, LSE = -inf."""
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    if keep_mask is not None:
        s = s.masked_fill(~keep_mask, float("-inf"))
    w = torch.softmax(s, dim=-1).nan_to_num()
    if keep is not None:
        w = w * keep / (1 - p)
    return w @ v.double(), torch.logsumexp(s, dim=-1)


def _bottom_right(sq, sk, device):
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return j <= i + (sk - sq)


def _check(tag, out, lse, ref, ref_lse):
    """Live rows against fp64; rows that see no key: O exactly 0, LSE +inf."""
    assert not bool(torch.isnan(out).any()), f"{tag}: NaN in O"
    err = (out.double() - ref).abs().max().item()
    assert err <= ATOL, f"{tag}: O off by {err:.3e}"
    if lse is None:
        return
    live = torch.isfinite(ref_lse)
    if bool(live.any()):
        lse_err = (lse.double() - ref_lse)[live].abs().max().item()
        assert lse_err <= LSE_ATOL, f"{tag}: LSE off by {lse_err:.3e}"
    dead = ~live
    assert bool(torch.isinf(lse[dead]).all()) and bool((lse[dead] > 0).all()), f"{tag}: dead-row LSE is not +inf"
    assert int((out[dead] != 0).sum()) == 0, f"{tag}: nonzero O in rows that see no key"


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape_name", list(DENSE_SHAPES))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_window(head_dim, shape_name, window_name, p):
    """gSWA (`causal_type=3`) windows measured from either principal diagonal."""
    sq, sk = DENSE_SHAPES[shape_name]
    window = WINDOWS[window_name]
    b, h = 2, 2
    torch.manual_seed(sq * 1000 + sk)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    _build(head_dim, causal_type=3)(q,
                                    k,
                                    v,
                                    out,
                                    b,
                                    sq,
                                    seqlen_k=sk,
                                    lse=lse,
                                    window=window,
                                    principal_diagonal=PRINCIPAL_DIAGONAL[p])
    torch.cuda.synchronize()
    keep = _keep_mask(window, p, sq, sk, q.device)
    ref, ref_lse = _reference(q, k, v, head_dim ** -0.5, keep_mask=keep)
    _check(f"hd={head_dim} {shape_name} {window_name} p={p}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (777, 300), (1, 1001), (1000, 1000)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_dense(head_dim, shape, causal):
    """Full attention and bottom-right causal (`causal_type=2`), GQA 2:1."""
    sq, sk = shape
    b, h, hk = 2, 4, 2
    torch.manual_seed(sq + sk + head_dim)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, hk, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    _build(head_dim, causal_type=2 if causal else 0)(q, k, v, out, b, sq, seqlen_k=sk, lse=lse)
    torch.cuda.synchronize()
    ref, ref_lse = _reference(q,
                              k.repeat_interleave(h // hk, 1),
                              v.repeat_interleave(h // hk, 1),
                              head_dim ** -0.5,
                              keep_mask=_bottom_right(sq, sk, q.device) if causal else None)
    _check(f"hd={head_dim} {shape} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_window(head_dim, p):
    """Packed sequences with one `seqlen_k - seqlen_q` each; `p` rides in varlen bits."""
    window = WINDOWS["4_4"]
    h = 2
    cu_q, cu_k = [0], [0]
    for a, b in VARLEN_SEQS:
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + b)
    torch.manual_seed(head_dim + p)
    q = torch.randn(1, h, cu_q[-1], head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(1, h, cu_k[-1], head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((h, cu_q[-1]), float("nan"), device="cuda")
    lq = [a for a, _ in VARLEN_SEQS]
    lk = [b for _, b in VARLEN_SEQS]
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
    _build(head_dim, causal_type=3)(q,
                                    k,
                                    v,
                                    out,
                                    1,
                                    max(lq),
                                    seqlen_k=max(lk),
                                    lse=lse,
                                    window=window,
                                    varlen=varlen,
                                    num_seqlens=len(VARLEN_SEQS))
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        keep = _keep_mask(window, p, sq, sk, q.device)
        ref, ref_lse = _reference(q[0, :, qs], k[0, :, ks], v[0, :, ks], head_dim ** -0.5, keep_mask=keep)
        _check(f"hd={head_dim} varlen seq{z}=({sq},{sk}) p={p}", out[0, :, qs], lse[:, qs], ref, ref_lse)


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("lse_layout", ["ht", "th"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen(head_dim, layout, lse_layout):
    """Compact packs sequences into (1, H, T, D); padded keeps (B, H, max, D)."""
    h, hk = 4, 2
    lq = [a for a, _ in SEQS]
    lk = [b for _, b in SEQS]
    n = len(SEQS)
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device="cuda", dtype=torch.int32)
    lse_bits = abi.VARLEN_LSE_LAYOUT_HT if lse_layout == "ht" else abi.VARLEN_LSE_LAYOUT_TH
    torch.manual_seed(head_dim)
    if layout == "compact":
        tq, tk = sum(lq), sum(lk)
        q = torch.randn(1, h, tq, head_dim, device="cuda", dtype=DTYPE)
        k = torch.randn(1, hk, tk, head_dim, device="cuda", dtype=DTYPE)
        varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = 1, n
        lse = torch.full((h, tq) if lse_layout == "ht" else (tq, h), float("nan"), device="cuda")
    else:
        tq = max(lq)
        q = torch.randn(n, h, tq, head_dim, device="cuda", dtype=DTYPE)
        k = torch.randn(n, hk, max(lk), head_dim, device="cuda", dtype=DTYPE)
        varlen = abi.varlen_padded(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = n, 0
        lse = torch.full((n * h, tq) if lse_layout == "ht" else (n * tq, h), float("nan"), device="cuda")
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    _build(head_dim)(q, k, v, out, batch, max(lq), seqlen_k=max(lk), num_seqlens=nseq, lse=lse, varlen=varlen)
    torch.cuda.synchronize()
    scale = head_dim ** -0.5
    for z, (sq, sk) in enumerate(SEQS):
        if layout == "compact":
            qz = q[0, :, cu_q[z]:cu_q[z + 1]]
            kz, vz = (t[0, :, cu_k[z]:cu_k[z + 1]] for t in (k, v))
            oz = out[0, :, cu_q[z]:cu_q[z + 1]]
            lz = lse[:, cu_q[z]:cu_q[z + 1]] if lse_layout == "ht" else lse[cu_q[z]:cu_q[z + 1]].T
        else:
            qz, kz, vz = q[z, :, :sq], k[z, :, :sk], v[z, :, :sk]
            oz = out[z, :, :sq]
            lz = lse[z * h:(z + 1) * h, :sq] if lse_layout == "ht" else lse[z * tq:z * tq + sq].T
        ref, ref_lse = _reference(qz, kz.repeat_interleave(h // hk, 0), vz.repeat_interleave(h // hk, 0), scale)
        _check(f"hd={head_dim} {layout} {lse_layout} seq{z}=({sq},{sk})", oz, lz, ref, ref_lse)


@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001)])
@pytest.mark.parametrize("head_dim", (160, 256))
def test_bias(head_dim, shape):
    """Odd pitches start bias rows mid-dword; an odd slab ends mid-dword."""
    sq, sk = shape
    b, h = 2, 3
    torch.manual_seed(sq + sk)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    bias = torch.randn(b, h, sq, sk, device="cuda", dtype=DTYPE)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    _build(head_dim, bias=True)(q, k, v, out, b, sq, seqlen_k=sk, lse=lse, bias=bias)
    torch.cuda.synchronize()
    ref, ref_lse = _reference(q, k, v, head_dim ** -0.5, bias=bias)
    _check(f"hd={head_dim} bias {shape}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (777, 300)])
@pytest.mark.parametrize("head_dim", (192, 256))
# 2**32 - 100003: the counter's low word wraps inside the planes.
@pytest.mark.parametrize("off2", [77, 2**32 - 100003])
def test_dropout(head_dim, shape, causal, off2):
    """The mask must be the Triton kernels': offset = base + plane*sq*stride + row*stride + col//4."""
    sq, sk = shape
    b, h, p = 2, 3, 0.3
    seed, off1 = 0x123456789ABCDEF0, 1000
    torch.manual_seed(sq * sk)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    seed_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    off_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    _build(head_dim, causal_type=2 if causal else 0, dropout=True)(
        q,
        k,
        v,
        out,
        b,
        sq,
        seqlen_k=sk,
        lse=lse,
        dropout_p=p,
        philox_seed=seed,
        philox_offset1=torch.tensor([off1], dtype=torch.int64, device="cuda"),
        philox_offset2=off2,
        philox_seed_output=seed_out,
        philox_offset_output=off_out)
    torch.cuda.synchronize()
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off1 + off2) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = _philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    keep = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk)
    ref, ref_lse = _reference(q,
                              k,
                              v,
                              head_dim ** -0.5,
                              keep=keep,
                              p=p,
                              keep_mask=_bottom_right(sq, sk, q.device) if causal else None)
    _check(f"hd={head_dim} dropout {shape} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)
    assert seed_out.item() == np.int64(np.uint64(seed)).item()
    assert off_out.item() == off1 + off2


# (rung, hdim_qk, hdim_vo): primes and odd sizes above the rung below, and QK != VO.
PADDED = [(160, 149, 149), (192, 179, 179), (224, 211, 211), (256, 241, 241), (192, 192, 128), (160, 24, 152),
          (256, 256, 200)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rung,hqk,hvo", PADDED)
def test_padded_head(rung, hqk, hvo, causal):
    """Columns past hdim are NaN in Q, K and V; none may reach a stored O column."""
    b, h, sq, sk = 2, 3, 257, 519
    if causal:
        sq, sk = sk, sq                 # bottom-right with dead rows
    torch.manual_seed(hqk * 1000 + hvo)
    q = _with_nan_slack(b, h, sq, hqk)
    k = _with_nan_slack(b, h, sk, hqk)
    v = _with_nan_slack(b, h, sk, hvo)
    out = _with_nan_slack(b, h, sq, hvo)
    out.fill_(float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    _build(rung, causal_type=2 if causal else 0, padded_head=True)(q, k, v, out, b, sq, seqlen_k=sk, lse=lse)
    torch.cuda.synchronize()
    ref, ref_lse = _reference(q, k, v, hqk ** -0.5, keep_mask=_bottom_right(sq, sk, q.device) if causal else None)
    _check(f"rung={rung} hqk={hqk} hvo={hvo} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("head_dim", (160, 256))
def test_null_lse(head_dim):
    """Inference passes no L; the kernel must not store to it."""
    q = torch.randn(2, 3, 300, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(2, 3, 500, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    _build(head_dim)(q, k, v, out, 2, 300, seqlen_k=500)
    torch.cuda.synchronize()
    ref, _ = _reference(q, k, v, head_dim ** -0.5)
    _check(f"hd={head_dim} null lse", out, None, ref, None)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_one_signed_unbiased(head_dim):
    """U[0, 1) inputs: a truncated P biases O low (numerical lessons, #16); bound eps/8."""
    sq, sk = 143, 579
    torch.manual_seed(head_dim)
    q = torch.rand(2, 2, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.rand(2, 2, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.rand_like(k)
    out = torch.empty_like(q)
    _build(head_dim)(q, k, v, out, 2, sq, seqlen_k=sk)
    torch.cuda.synchronize()
    ref, _ = _reference(q, k, v, head_dim ** -0.5)
    signed = ((out.double() - ref).sum() / ref.abs().sum()).item()
    assert abs(signed) <= 2 ** -10, f"hd={head_dim}: O biased by {signed:+.2e} of its magnitude"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
