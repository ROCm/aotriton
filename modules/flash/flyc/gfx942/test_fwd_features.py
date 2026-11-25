#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gfx942 forward's features other than windows, JIT, against fp64.

Windows and principal_diagonal_choice are tests/test_fwd_principal_diagonal.py.
This covers varlen layouts and both LSE layouts, bias, dropout (against an
independent host Philox), padded head dims with NaN slack, rows that see no
key, a null LSE, and the launcher's `principal_diagonal=` checks.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_fwd_features.py
"""

import functools
import sys
from pathlib import Path

import pytest

from .jit_common import (  # noqa: F401
    bottom_right,
    check_fwd,
    fwd_reference,
    np,
    philox_4x32_10,
    pytestmark,
    torch,
    with_nan_slack,
)

import fmha_utils as abi  # noqa: E402
import philox  # noqa: E402

HEAD_DIMS = (64, 128)
DTYPE = torch.bfloat16


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False):
    from .fwd import build_fmha_fwd_gfx942

    return build_fmha_fwd_gfx942(head_dim,
                                 causal_type=causal_type,
                                 bias=bias,
                                 dropout=dropout,
                                 padded_head=padded_head)


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
        ref, ref_lse = fwd_reference(qz, kz.repeat_interleave(h // hk, 0), vz.repeat_interleave(h // hk, 0), scale)
        check_fwd(f"hd={head_dim} {layout} {lse_layout} seq{z}=({sq},{sk})", oz, lz, ref, ref_lse)


@pytest.mark.parametrize("head_dim", (64, 128, 192, 512))
def test_varlen_bias(head_dim):
    """Packed bias (1, H, sum(Sq), sum(Sk)): sequence z reads its own [q_off:, k_off:] block."""
    from .fwd import build_fmha_fwd_gfx942_by_head_dim

    h = 2
    lq = [a for a, _ in SEQS]
    lk = [b for _, b in SEQS]
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device="cuda", dtype=torch.int32)
    tq, tk = sum(lq), sum(lk)
    torch.manual_seed(head_dim + 1)
    q = torch.randn(1, h, tq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(1, h, tk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    bias = torch.randn(1, h, tq, tk, device="cuda", dtype=DTYPE)
    varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=abi.VARLEN_LSE_LAYOUT_HT)
    lse = torch.full((h, tq), float("nan"), device="cuda")
    out = torch.full_like(q, float("nan"))
    launch = build_fmha_fwd_gfx942_by_head_dim(head_dim, bias=True)
    launch(q, k, v, out, 1, max(lq), seqlen_k=max(lk), num_seqlens=len(SEQS), lse=lse, varlen=varlen, bias=bias)
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(SEQS):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        ref, ref_lse = fwd_reference(q[0, :, qs], k[0, :, ks], v[0, :, ks], head_dim ** -0.5, bias=bias[0, :, qs, ks])
        check_fwd(f"hd={head_dim} seq{z}=({sq},{sk})", out[0, :, qs], lse[:, qs], ref, ref_lse)


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
    ref, ref_lse = fwd_reference(q, k, v, head_dim ** -0.5, bias=bias)
    check_fwd(f"hd={head_dim} bias {shape}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(300, 777), (257, 519), (777, 300)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
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
    rand = philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    keep = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk)
    ref, ref_lse = fwd_reference(q,
                                 k,
                                 v,
                                 head_dim ** -0.5,
                                 keep=keep,
                                 p=p,
                                 mask=bottom_right(sq, sk) if causal else None)
    check_fwd(f"hd={head_dim} dropout {shape} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)
    assert seed_out.item() == np.int64(np.uint64(seed)).item()
    assert off_out.item() == off1 + off2


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
    q = with_nan_slack(b, h, sq, hqk)
    k = with_nan_slack(b, h, sk, hqk)
    v = with_nan_slack(b, h, sk, hvo)
    out = with_nan_slack(b, h, sq, hvo)
    out.fill_(float("nan"))
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    _build(rung, causal_type=2 if causal else 0, padded_head=True)(q, k, v, out, b, sq, seqlen_k=sk, lse=lse)
    torch.cuda.synchronize()
    ref, ref_lse = fwd_reference(q, k, v, hqk ** -0.5, mask=bottom_right(sq, sk) if causal else None)
    check_fwd(f"rung={rung} hqk={hqk} hvo={hvo} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_null_lse(head_dim):
    """Inference passes no L; the kernel must not store to it."""
    q = torch.randn(2, 3, 300, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(2, 3, 500, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    _build(head_dim)(q, k, v, out, 2, 300, seqlen_k=500)
    torch.cuda.synchronize()
    ref, _ = fwd_reference(q, k, v, head_dim ** -0.5)
    check_fwd(f"hd={head_dim} null lse", out, None, ref, None)


def _tiny_call(fn, **kwargs):
    q = torch.zeros((1, 2, 8, HEAD_DIMS[0]), device="cuda", dtype=DTYPE)
    fn(q,
       q,
       q,
       torch.empty_like(q),
       1,
       8,
       lse=torch.empty((2, 8), device="cuda", dtype=torch.float32),
       window=(0, 0),
       **kwargs)


def test_principal_diagonal_rejects_other_values():
    """Only the two `VARLEN_PRINCIPAL_DIAGONAL_*` words; a bare 1 is not bit 18."""
    with pytest.raises(ValueError, match="principal_diagonal"):
        _tiny_call(_build(HEAD_DIMS[0], causal_type=3), principal_diagonal=1)


def test_principal_diagonal_conflicting_with_varlen_bits():
    """`varlen["bits"]` says BOTTOM_RIGHT, `principal_diagonal=` says TOP_LEFT."""
    cu = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    varlen = {
        "bits": abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                abi.VARLEN_COMPACT_SIDE,
                                principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT),
        "seqinfo_q0": cu,
        "seqinfo_k0": cu,
        "max_seqlen_q": 8,
        "max_seqlen_k": 8,
    }
    with pytest.raises(ValueError, match="BOTTOM_RIGHT"):
        _tiny_call(_build(HEAD_DIMS[0], causal_type=3),
                   varlen=varlen,
                   num_seqlens=1,
                   principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
