#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gfx942 forward's features other than windows, JIT, against fp64.

Windows and principal_diagonal_choice are test_fwd_principal_diagonal_gfx942.py.
This covers varlen layouts and both LSE layouts, bias, dropout (against an
independent host Philox), padded head dims with NaN slack, rows that see no
key, and a null LSE.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_fwd_features_gfx942.py
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

HEAD_DIMS = (64, 128)
DTYPE = torch.bfloat16
ATOL = 1.5e-2
LSE_ATOL = 1e-4


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False):
    from fmha_fwd_gfx942 import build_fmha_fwd_gfx942

    return build_fmha_fwd_gfx942(head_dim,
                                 causal_type=causal_type,
                                 bias=bias,
                                 dropout=dropout,
                                 padded_head=padded_head)


def _reference(q, k, v, scale, bias=None, keep=None, p=0.0, bottom_right=False):
    """fp64 `(O, LSE)` over `[..., sq, D]`; a row that sees no key has O = 0."""
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    if bottom_right:
        sq, sk = s.shape[-2:]
        i = torch.arange(sq, device=s.device)[:, None]
        j = torch.arange(sk, device=s.device)[None, :]
        s = s.masked_fill(j > i + (sk - sq), float("-inf"))
    w = torch.softmax(s, dim=-1).nan_to_num()
    if keep is not None:
        w = w * keep / (1 - p)
    return w @ v.double(), torch.logsumexp(s, dim=-1)


def _check(tag, out, lse, ref, ref_lse):
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
            qz, kz, vz = (t[0, :, s] for t, s in ((q, slice(cu_q[z], cu_q[z + 1])),
                                                  (k, slice(cu_k[z], cu_k[z + 1])),
                                                  (v, slice(cu_k[z], cu_k[z + 1]))))
            oz = out[0, :, cu_q[z]:cu_q[z + 1]]
            lz = lse[:, cu_q[z]:cu_q[z + 1]] if lse_layout == "ht" else lse[cu_q[z]:cu_q[z + 1]].T
        else:
            qz, kz, vz = q[z, :, :sq], k[z, :, :sk], v[z, :, :sk]
            oz = out[z, :, :sq]
            lz = lse[z * h:(z + 1) * h, :sq] if lse_layout == "ht" else lse[z * tq:z * tq + sq].T
        ref, ref_lse = _reference(qz, kz.repeat_interleave(h // hk, 0), vz.repeat_interleave(h // hk, 0), scale)
        _check(f"hd={head_dim} {layout} {lse_layout} seq{z}=({sq},{sk})", oz, lz, ref, ref_lse)


@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001), (500, 500)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
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


_M32 = 0xFFFFFFFF


def _philox_4x32_10(seed, offsets):
    """Philox-4x32-10 with tl.randint4x's wiring, as in kernel/test_dropout_layout.py."""
    u = np.uint64
    off = np.asarray(offsets, dtype=np.uint64)
    c0, c1 = off & u(_M32), (off >> u(32)) & u(_M32)
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    k0, k1 = u(seed & _M32), u((seed >> 32) & _M32)
    mul_a, mul_b = u(0xD2511F53), u(0xCD9E8D57)
    for _ in range(10):
        p0, p2 = c0, c2
        c0 = ((mul_b * p2) >> u(32)) ^ c1 ^ k0
        c2 = ((mul_a * p0) >> u(32)) ^ c3 ^ k1
        c1 = (mul_b * p2) & u(_M32)
        c3 = (mul_a * p0) & u(_M32)
        k0 = (k0 + u(0x9E3779B9)) & u(_M32)
        k1 = (k1 + u(0xBB67AE85)) & u(_M32)
    return np.stack([c.astype(np.uint32).view(np.int32) for c in (c0, c1, c2, c3)], -1)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (257, 519), (777, 300)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_dropout(head_dim, shape, causal):
    """The mask must be the Triton kernels': offset = base + plane*sq*stride + row*stride + col//4."""
    sq, sk = shape
    b, h, p = 2, 3, 0.3
    seed, off1, off2 = 0x123456789ABCDEF0, 1000, 77
    torch.manual_seed(sq * sk)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    seed_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    off_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    _build(head_dim, causal_type=2 if causal else 0, dropout=True)(
        q, k, v, out, b, sq,
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
    ref, ref_lse = _reference(q, k, v, head_dim ** -0.5, keep=keep, p=p, bottom_right=causal)
    _check(f"hd={head_dim} dropout {shape} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)
    assert seed_out.item() == np.int64(np.uint64(seed)).item()
    assert off_out.item() == off1 + off2


def _with_nan_slack(*shape):
    """A (..., hdim) view of a (..., ceil8(hdim)) tensor whose slack is NaN: the 8xD contract."""
    *lead, d = shape
    full = torch.full((*lead, -(-d // 8) * 8), float("nan"), device="cuda", dtype=DTYPE)
    view = full[..., :d]
    view.copy_(torch.randn(*shape, device="cuda", dtype=DTYPE))
    return view


# (rung, hdim_qk, hdim_vo): primes, odd sizes, and QK != VO, each above the rung below.
PADDED = [(32, 7, 7), (32, 23, 23), (64, 37, 37), (64, 53, 61), (96, 67, 67), (96, 83, 83),
          (128, 113, 113), (128, 120, 72), (128, 97, 128)]


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
    ref, ref_lse = _reference(q, k, v, hqk ** -0.5, bottom_right=causal)
    _check(f"rung={rung} hqk={hqk} hvo={hvo} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
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


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
