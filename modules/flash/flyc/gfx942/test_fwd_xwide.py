#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gfx942 forward's xwide body (head dims 384 and 512), JIT, against fp64.

Covers what tests/test_fwd_principal_diagonal.py and
test_fwd_features.py cover for the narrow body: windows with both
principal diagonals (dense and varlen), varlen layouts and both LSE layouts,
bias, dropout against an independent host Philox, padded head dims with NaN
slack (odd ones, and hdim_qk != hdim_vo), rows that see no key, and a null LSE.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_fwd_xwide.py
"""

import functools
import sys
from pathlib import Path

import pytest

from .jit_common import (  # noqa: F401
    TL,
    BR,
    bottom_right,
    check_fwd,
    fwd_reference,
    keep_mask,
    np,
    philox_4x32_10,
    pytestmark,
    torch,
    with_nan_slack,
)

import fmha_utils as abi  # noqa: E402
import philox  # noqa: E402

HEAD_DIMS = (384, 512)
DTYPE = torch.bfloat16


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False, grid_axis_order=None):
    from .fwd import build_fmha_fwd_gfx942_by_head_dim

    return build_fmha_fwd_gfx942_by_head_dim(head_dim,
                                             body='xwide',
                                             causal_type=causal_type,
                                             bias=bias,
                                             dropout=dropout,
                                             padded_head=padded_head,
                                             grid_axis_order=grid_axis_order)


@pytest.mark.parametrize("shape", [(1, 300), (300, 37), (129, 130), (1000, 777)])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_dense(head_dim, shape):
    """Ragged q and kv tiles; GQA; BSHD strides."""
    sq, sk = shape
    b, h, hk = 2, 4, 2
    torch.manual_seed(sq + sk + head_dim)
    q = torch.randn(b, sq, h, head_dim, device="cuda", dtype=DTYPE).transpose(1, 2)
    k = torch.randn(b, sk, hk, head_dim, device="cuda", dtype=DTYPE).transpose(1, 2)
    v = torch.randn(b, sk, hk, head_dim, device="cuda", dtype=DTYPE).transpose(1, 2)
    out = torch.full((b, sq, h, head_dim), float("nan"), device="cuda", dtype=DTYPE).transpose(1, 2)
    lse = torch.full((b * h, sq), float("nan"), device="cuda")
    _build(head_dim)(q, k, v, out, b, sq, seqlen_k=sk, lse=lse)
    torch.cuda.synchronize()
    ref, ref_lse = fwd_reference(q,
                                 k.repeat_interleave(h // hk, 1),
                                 v.repeat_interleave(h // hk, 1),
                                 head_dim ** -0.5)
    check_fwd(f"hd={head_dim} {shape}", out, lse.view(b, h, sq), ref, ref_lse)


WINDOWS = {
    "sentinel_left_0": (TL, 0),
    "8_0": (8, 0),
    "0_8": (0, 8),
    "4_4": (4, 4),
    "neg1_3": (-1, 3),
    "100_20": (100, 20),
    "sentinel_br": (BR, BR),
}
PRINCIPAL_DIAGONAL = {
    0: abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT,
    1: abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT,
}
DENSE_SHAPES = {"q_lt_k": (37, 300), "q_eq_k": (130, 130), "q_gt_k": (300, 37), "decode": (1, 300)}
VARLEN_SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape_name", list(DENSE_SHAPES))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_window(head_dim, shape_name, window_name, p):
    """gSWA windows, `p` through `principal_diagonal=`."""
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
    ref, ref_lse = fwd_reference(q, k, v, head_dim ** -0.5, mask=keep_mask(window, p, sq, sk, q.device))
    check_fwd(f"hd={head_dim} {shape_name} {window_name} p={p}", out, lse.view(b, h, sq), ref, ref_lse)


def _varlen_tensors(head_dim, layout, lse_layout, h, hk, seed):
    lq = [a for a, _ in VARLEN_SEQS]
    lk = [c for _, c in VARLEN_SEQS]
    n = len(VARLEN_SEQS)
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device="cuda", dtype=torch.int32)
    lse_bits = abi.VARLEN_LSE_LAYOUT_HT if lse_layout == "ht" else abi.VARLEN_LSE_LAYOUT_TH
    torch.manual_seed(seed)
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
    return q, k, v, out, lse, varlen, batch, nseq, cu_q, cu_k


def _varlen_slices(z, layout, lse_layout, tensors, h):
    q, k, v, out, lse, _, _, _, cu_q, cu_k = tensors
    sq, sk = VARLEN_SEQS[z]
    if layout == "compact":
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        lz = lse[:, qs] if lse_layout == "ht" else lse[qs].T
        return q[0, :, qs], k[0, :, ks], v[0, :, ks], out[0, :, qs], lz
    tq = q.shape[2]
    lz = lse[z * h:(z + 1) * h, :sq] if lse_layout == "ht" else lse[z * tq:z * tq + sq].T
    return q[z, :, :sq], k[z, :, :sk], v[z, :, :sk], out[z, :, :sq], lz


@pytest.mark.parametrize("lse_layout", ["ht", "th"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen(head_dim, layout, lse_layout):
    """Compact packs sequences into (1, H, T, D); padded keeps (B, H, max, D)."""
    h, hk = 4, 2
    tensors = _varlen_tensors(head_dim, layout, lse_layout, h, hk, head_dim)
    q, k, v, out, lse, varlen, batch, nseq, _, _ = tensors
    lq = [a for a, _ in VARLEN_SEQS]
    lk = [c for _, c in VARLEN_SEQS]
    _build(head_dim)(q, k, v, out, batch, max(lq), seqlen_k=max(lk), num_seqlens=nseq, lse=lse, varlen=varlen)
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        qz, kz, vz, oz, lz = _varlen_slices(z, layout, lse_layout, tensors, h)
        ref, ref_lse = fwd_reference(qz,
                                     kz.repeat_interleave(h // hk, 0),
                                     vz.repeat_interleave(h // hk, 0),
                                     head_dim ** -0.5)
        check_fwd(f"hd={head_dim} {layout} {lse_layout} seq{z}=({sq},{sk})", oz, lz, ref, ref_lse)


@pytest.mark.parametrize("grid", [0, 1])
@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", ["8_0", "neg1_3", "100_20", "sentinel_br"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_window(head_dim, window_name, p, grid):
    """One `d = sk - sq` per sequence; `p` travels in `varlen["bits"]`. Both grid
    orders, since a causal build runs q tiles in reverse."""
    h = hk = 2
    tensors = _varlen_tensors(head_dim, "compact", "ht", h, hk, head_dim + p)
    q, k, v, out, lse, varlen, batch, nseq, _, _ = tensors
    varlen = dict(varlen, bits=varlen["bits"] | PRINCIPAL_DIAGONAL[p])
    window = WINDOWS[window_name]
    lq = [a for a, _ in VARLEN_SEQS]
    lk = [c for _, c in VARLEN_SEQS]
    _build(head_dim, causal_type=3, grid_axis_order=grid)(q,
                                                          k,
                                                          v,
                                                          out,
                                                          batch,
                                                          max(lq),
                                                          seqlen_k=max(lk),
                                                          num_seqlens=nseq,
                                                          lse=lse,
                                                          window=window,
                                                          varlen=varlen)
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        qz, kz, vz, oz, lz = _varlen_slices(z, "compact", "ht", tensors, h)
        ref, ref_lse = fwd_reference(qz, kz, vz, head_dim ** -0.5, mask=keep_mask(window, p, sq, sk, q.device))
        check_fwd(f"hd={head_dim} varlen seq{z}=({sq},{sk}) {window_name} p={p} grid={grid}", oz, lz, ref, ref_lse)


@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001), (129, 33)])
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
    allow = bottom_right(sq, sk, q.device) if causal else None
    ref, ref_lse = fwd_reference(q, k, v, head_dim ** -0.5, keep=keep, p=p, mask=allow)
    check_fwd(f"hd={head_dim} dropout {shape} causal={causal}", out, lse.view(b, h, sq), ref, ref_lse)
    assert seed_out.item() == np.int64(np.uint64(seed)).item()
    assert off_out.item() == off1 + off2


# (rung, hdim_qk, hdim_vo): odd and even, QK != VO, and one side far below the rung.
PADDED = [(384, 300, 300), (384, 257, 257), (384, 264, 384), (384, 37, 333),
          (512, 401, 401), (512, 408, 408), (512, 512, 128), (512, 385, 449), (512, 129, 511)]


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
    allow = bottom_right(sq, sk, q.device) if causal else None
    ref, ref_lse = fwd_reference(q, k, v, hqk ** -0.5, mask=allow)
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


def test_rejects_wrong_head_dim():
    q = torch.zeros(1, 1, 8, 400, device="cuda", dtype=DTYPE)
    with pytest.raises(ValueError, match="padded_head"):
        _build(512)(q, q, q, torch.empty_like(q), 1, 8)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
