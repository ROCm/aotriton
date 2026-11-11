#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gfx942 backward dQ/dB kernel, JIT, against fp64.

LSE and Delta come from the fp64 reference in the layout the forward and
Triton's bwd_preprocess write them: LSE natural base and +inf on rows that see
no key, Delta = rowsum(dO * O) with O rounded to the input dtype. So this tests
dQ alone. A row that sees no key must get dQ exactly 0.

Covers causal top-left and bottom-right, gSWA windows with both
principal-diagonal choices, GQA, varlen with both LSE layouts, per-tensor
strides, bias with dB (and a caller that wants no dB), dropout against an
independent host Philox, padded head dims with NaN slack, f16, and zero and
negative sm_scale.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_bwd_dq_gfx942.py
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
import fmha_common_gfx1201 as fmha  # noqa: E402
import philox  # noqa: E402

HEAD_DIMS = (32, 64, 96, 128)
DTYPE = torch.bfloat16
# Max error relative to max(1, max |dQ|); a correct kernel sits near 3e-3.
REL_TOL = 1e-2

TL = fmha.WINDOW_TOPLEFT
BR = fmha.WINDOW_BOTRIGHT


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False, dtype_str="bf16"):
    from fmha_bwd_dq_gfx942 import build_fmha_bwd_dq_gfx942

    return build_fmha_bwd_dq_gfx942(head_dim,
                                    causal_type=causal_type,
                                    bias=bias,
                                    dropout=dropout,
                                    padded_head=padded_head,
                                    dtype_str=dtype_str)


def _keep_mask(causal_type, window, p, sq, sk, device="cuda"):
    """`[sq, sk]` keep mask: none, top-left, bottom-right, or the gSWA A-2 predicate."""
    i = torch.arange(sq, device=device).view(sq, 1)
    j = torch.arange(sk, device=device).view(1, sk)
    if causal_type == 0:
        return torch.ones(sq, sk, dtype=torch.bool, device=device)
    if causal_type == 1:
        return j <= i
    if causal_type == 2:
        return j <= i + (sk - sq)
    wl, wr = window
    d = sk - sq
    left = sq if wl in (TL, BR) else wl - p * d
    right = (d if wr == BR else 0) if wr in (TL, BR) else wr + p * d
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, scale, keep, bias=None, drop=None, p=0.0):
    """fp64 `(dQ, dB, LSE, Delta, live)` over `[H, s, D]` operands; K and V already
    repeated to the query heads. `drop` is the keep mask of dropout."""
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    s = s.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    live = torch.isfinite(lse)
    pr = torch.exp(s - lse.unsqueeze(-1)).nan_to_num()
    pd = pr if drop is None else pr * drop / (1 - p)
    o = (pd @ v.double()).to(q.dtype).double()
    delta = (do.double() * o).sum(-1)
    dp = do.double() @ v.double().transpose(-1, -2)
    if drop is not None:
        dp = dp * drop / (1 - p)
    ds = pr * (dp - delta.unsqueeze(-1))
    dq = ds @ k.double() * scale
    lse = lse.masked_fill(~live, float("inf"))
    return dq, ds, lse.float(), delta.float(), live


def _check(tag, got, ref, live=None):
    assert bool(torch.isfinite(got).all()), f"{tag}: non-finite dQ"
    scale = max(1.0, ref.abs().max().item())
    err = (got.double() - ref).abs().max().item() / scale
    assert err <= REL_TOL, f"{tag}: dQ off by {err:.3e} (relative to {scale:.3e})"
    if live is not None and not bool(live.all()):
        dead = got[..., ~live, :]
        assert int((dead != 0).sum()) == 0, f"{tag}: nonzero dQ on {int((~live).sum())} rows that see no key"


def _run_dense(head_dim, causal_type, sq, sk, h=4, hk=2, b=2, window=None, p=None, scale=None, seed=0,
               layouts=False):
    torch.manual_seed(seed)
    if layouts:
        # Every tensor its own outer-axis order (numerical lessons, #10).
        def mk(s, hh, perm):
            """A (b, hh, s, D) view of storage whose outer axes are in `perm` order."""
            shape = [b, hh, s]
            base = torch.randn(*[shape[i] for i in perm], head_dim, device="cuda", dtype=DTYPE)
            return base.permute(*[perm.index(i) for i in range(3)], 3)
        q = mk(sq, h, [0, 2, 1])
        k = mk(sk, hk, [2, 0, 1])
        v = mk(sk, hk, [1, 2, 0])
        do = mk(sq, h, [2, 1, 0])
        dq = torch.full((sq, b, h, head_dim), float("nan"), device="cuda", dtype=DTYPE).permute(1, 2, 0, 3)
    else:
        q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
        k = torch.randn(b, hk, sk, head_dim, device="cuda", dtype=DTYPE)
        v = torch.randn_like(k)
        do = torch.randn_like(q)
        dq = torch.full_like(q, float("nan"))
    sm = head_dim ** -0.5 if scale is None else scale
    keep = _keep_mask(causal_type, window, p or 0, sq, sk)
    refs = [_reference(q[z], k[z].repeat_interleave(h // hk, 0), v[z].repeat_interleave(h // hk, 0), do[z], sm, keep)
            for z in range(b)]
    lse = torch.stack([r[2] for r in refs]).reshape(b * h, sq).contiguous()
    delta = torch.stack([r[3] for r in refs]).reshape(b * h, sq).contiguous()
    pd = None if p is None else (abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT)
    _build(head_dim, causal_type)(q,
                                  k,
                                  v,
                                  do,
                                  dq,
                                  lse,
                                  delta,
                                  b,
                                  sq,
                                  seqlen_k=sk,
                                  scale=scale,
                                  window=window,
                                  principal_diagonal=pd)
    torch.cuda.synchronize()
    for z in range(b):
        _check(f"hd={head_dim} ct={causal_type} {sq}x{sk} w={window} p={p} batch {z}", dq[z], refs[z][0], refs[z][4])


DENSE_SHAPES = [(256, 256), (300, 519), (519, 300), (1, 1001), (37, 64), (130, 130), (1000, 777)]


@pytest.mark.parametrize("shape", DENSE_SHAPES)
@pytest.mark.parametrize("causal_type", [0, 1, 2])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_dense(head_dim, causal_type, shape):
    """GQA 4/2; bottom-right with sq > sk leaves whole Q blocks without keys."""
    _run_dense(head_dim, causal_type, *shape)


@pytest.mark.parametrize("causal_type", [0, 2])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_layouts(head_dim, causal_type):
    _run_dense(head_dim, causal_type, 300, 519, layouts=True)


WINDOWS = {
    "sentinel_left_0": (TL, 0),
    "8_0": (8, 0),
    "0_8": (0, 8),
    "4_4": (4, 4),
    "neg1_3": (-1, 3),
    "100_37": (100, 37),
    "sentinel_br": (BR, BR),
}

WINDOW_SHAPES = {
    "q_lt_k": (37, 300),
    "q_eq_k": (130, 130),
    "q_gt_k": (300, 37),
    "decode": (1, 300),
    "chunked_prefill": (64, 320),
    "long": (600, 700),
}


@pytest.mark.parametrize("shape", tuple(WINDOW_SHAPES))
@pytest.mark.parametrize("window", tuple(WINDOWS))
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("head_dim", (64, 128))
def test_window(head_dim, p, window, shape):
    _run_dense(head_dim, 3, *WINDOW_SHAPES[shape], window=WINDOWS[window], p=p)


@pytest.mark.parametrize("scale", [0.0, -0.3, 1.2])
@pytest.mark.parametrize("causal_type", [0, 2])
def test_sm_scale(causal_type, scale):
    """Masking after the scale keeps a zero or negative sm_scale finite (lessons, #7)."""
    _run_dense(64, causal_type, 300, 519, scale=scale)


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
VARLEN_SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200), (513, 600))


@pytest.mark.parametrize("causal_type", [0, 2, 3])
@pytest.mark.parametrize("lse_layout", ["ht", "th"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_varlen(head_dim, layout, lse_layout, causal_type):
    """Compact packs sequences into (1, H, T, D); padded keeps (B, H, max, D).
    Windows resolve per sequence: (8, 0) from the bottom-right diagonal."""
    h, hk = 4, 2
    lq = [a for a, _ in VARLEN_SEQS]
    lk = [b for _, b in VARLEN_SEQS]
    n = len(VARLEN_SEQS)
    window = (8, 0) if causal_type == 3 else None
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
        rows = (h, tq) if lse_layout == "ht" else (tq, h)
    else:
        tq = max(lq)
        q = torch.randn(n, h, tq, head_dim, device="cuda", dtype=DTYPE)
        k = torch.randn(n, hk, max(lk), head_dim, device="cuda", dtype=DTYPE)
        varlen = abi.varlen_padded(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = n, 0
        rows = (n * h, tq) if lse_layout == "ht" else (n * tq, h)
    if causal_type == 3:
        varlen["bits"] |= abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dq = torch.full_like(q, float("nan"))
    lse = torch.full(rows, float("nan"), device="cuda")
    delta = torch.full(rows, float("nan"), device="cuda")
    scale = head_dim ** -0.5

    def views(z, sq, sk):
        if layout == "compact":
            qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
            out = (q[0, :, qs], k[0, :, ks], v[0, :, ks], do[0, :, qs], dq[0, :, qs])
            rw = (lse[:, qs], delta[:, qs]) if lse_layout == "ht" else (lse[qs].T, delta[qs].T)
        else:
            out = (q[z, :, :sq], k[z, :, :sk], v[z, :, :sk], do[z, :, :sq], dq[z, :, :sq])
            if lse_layout == "ht":
                rw = (lse[z * h:(z + 1) * h, :sq], delta[z * h:(z + 1) * h, :sq])
            else:
                rw = (lse[z * tq:z * tq + sq].T, delta[z * tq:z * tq + sq].T)
        return out, rw

    refs = []
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        (qz, kz, vz, doz, _), (lz, dz) = views(z, sq, sk)
        keep = _keep_mask(causal_type, (8, 0), 1, sq, sk)
        r = _reference(qz, kz.repeat_interleave(h // hk, 0), vz.repeat_interleave(h // hk, 0), doz, scale, keep)
        lz.copy_(r[2])
        dz.copy_(r[3])
        refs.append(r)
    _build(head_dim, causal_type)(q,
                                  k,
                                  v,
                                  do,
                                  dq,
                                  lse,
                                  delta,
                                  batch,
                                  max(lq),
                                  seqlen_k=max(lk),
                                  num_seqlens=nseq,
                                  window=window,
                                  varlen=varlen)
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        (_, _, _, _, dqz), _ = views(z, sq, sk)
        _check(f"hd={head_dim} {layout} {lse_layout} ct={causal_type} seq{z}=({sq},{sk})", dqz, refs[z][0], refs[z][4])


@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001), (500, 500), (257, 64)])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_bias(head_dim, shape):
    """dB = dS. Odd pitches start bias and dB rows mid-dword; an odd slab ends mid-dword."""
    sq, sk = shape
    b, h = 2, 3
    torch.manual_seed(sq + sk)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    bias = torch.randn(b, h, sq, sk, device="cuda", dtype=DTYPE) * 2
    # dB in its own layout: (Sq, B, H, Sk) storage.
    dbias = torch.full((sq, b, h, sk), float("nan"), device="cuda", dtype=DTYPE).permute(1, 2, 0, 3)
    dq = torch.full_like(q, float("nan"))
    scale = head_dim ** -0.5
    keep = _keep_mask(0, None, 0, sq, sk)
    refs = [_reference(q[z], k[z], v[z], do[z], scale, keep, bias=bias[z]) for z in range(b)]
    lse = torch.stack([r[2] for r in refs]).reshape(b * h, sq).contiguous()
    delta = torch.stack([r[3] for r in refs]).reshape(b * h, sq).contiguous()
    fn = _build(head_dim, bias=True)
    fn(q, k, v, do, dq, lse, delta, b, sq, seqlen_k=sk, bias=bias, dbias=dbias)
    torch.cuda.synchronize()
    for z in range(b):
        _check(f"hd={head_dim} bias {shape} batch {z}", dq[z], refs[z][0])
        assert bool(torch.isfinite(dbias[z]).all()), f"bias {shape}: dB not fully written"
        err = (dbias[z].double() - refs[z][1]).abs().max().item()
        tol = 2 ** -8 * max(1.0, refs[z][1].abs().max().item())
        assert err <= tol, f"hd={head_dim} bias {shape} batch {z}: dB off by {err:.3e}"
    # A bias that wants no gradient: all-zero dB strides, nothing may be stored.
    sentinel = torch.full((sk,), float("nan"), device="cuda", dtype=DTYPE)
    nograd = sentinel.as_strided(bias.shape, (0, 0, 0, 1))
    dq2 = torch.full_like(q, float("nan"))
    fn(q, k, v, do, dq2, lse, delta, b, sq, seqlen_k=sk, bias=bias, dbias=nograd)
    torch.cuda.synchronize()
    assert bool(torch.isnan(sentinel).all()), "dB was stored with all-zero dB strides"
    assert torch.equal(dq2, dq), "dQ differs when dB is not wanted"


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
@pytest.mark.parametrize("head_dim", (64, 128))
def test_dropout(head_dim, shape, causal):
    """The mask is the Triton kernels': offset = base + plane*sq*stride + row*stride + col//4.
    dP is masked and scaled by 1/(1-p); P is not (numerical lessons, #9)."""
    sq, sk = shape
    b, h, p = 2, 3, 0.3
    seed, off1, off2 = 0x123456789ABCDEF0, 1000, 77
    torch.manual_seed(sq * sk)
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dq = torch.full_like(q, float("nan"))
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off1 + off2) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = _philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    drop = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk)
    keep = _keep_mask(2 if causal else 0, None, 0, sq, sk)
    refs = [_reference(q[z], k[z], v[z], do[z], head_dim ** -0.5, keep, drop=drop[z], p=p) for z in range(b)]
    lse = torch.stack([r[2] for r in refs]).reshape(b * h, sq).contiguous()
    delta = torch.stack([r[3] for r in refs]).reshape(b * h, sq).contiguous()
    _build(head_dim, causal_type=2 if causal else 0, dropout=True)(
        q, k, v, do, dq, lse, delta, b, sq,
        seqlen_k=sk,
        dropout_p=p,
        philox_seed=seed,
        philox_offset1=torch.tensor([off1], dtype=torch.int64, device="cuda"),
        philox_offset2=off2)
    torch.cuda.synchronize()
    for z in range(b):
        _check(f"hd={head_dim} dropout {shape} causal={causal} batch {z}", dq[z], refs[z][0], refs[z][4])


def _with_nan_slack(*shape):
    """A (..., hdim) view of a (..., ceil8(hdim)) tensor whose slack is NaN: the 8xD contract."""
    *lead, d = shape
    full = torch.full((*lead, -(-d // 8) * 8), float("nan"), device="cuda", dtype=DTYPE)
    view = full[..., :d]
    view.copy_(torch.randn(*shape, device="cuda", dtype=DTYPE))
    return view


# (rung, hdim_qk, hdim_vo): primes, odd sizes, and QK != VO, each above the rung below.
PADDED = [(32, 7, 7), (32, 23, 23), (64, 37, 37), (64, 53, 61), (96, 67, 67), (96, 83, 90),
          (128, 113, 113), (128, 120, 72), (128, 97, 128)]


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
    dq = _with_nan_slack(b, h, sq, hqk)
    dq.fill_(float("nan"))
    scale = hqk ** -0.5
    keep = _keep_mask(2 if causal else 0, None, 0, sq, sk)
    refs = [_reference(q[z], k[z], v[z], do[z], scale, keep) for z in range(b)]
    lse = torch.stack([r[2] for r in refs]).reshape(b * h, sq).contiguous()
    delta = torch.stack([r[3] for r in refs]).reshape(b * h, sq).contiguous()
    _build(rung, causal_type=2 if causal else 0, padded_head=True)(q, k, v, do, dq, lse, delta, b, sq, seqlen_k=sk)
    torch.cuda.synchronize()
    for z in range(b):
        _check(f"rung={rung} hqk={hqk} hvo={hvo} causal={causal} batch {z}", dq[z], refs[z][0], refs[z][4])


@pytest.mark.parametrize("causal_type", [0, 2])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_f16(head_dim, causal_type):
    sq, sk, b, h = 300, 519, 2, 3
    torch.manual_seed(7)
    dt = torch.float16
    q = torch.randn(b, h, sq, head_dim, device="cuda", dtype=dt)
    k = torch.randn(b, h, sk, head_dim, device="cuda", dtype=dt)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dq = torch.full_like(q, float("nan"))
    keep = _keep_mask(causal_type, None, 0, sq, sk)
    refs = [_reference(q[z], k[z], v[z], do[z], head_dim ** -0.5, keep) for z in range(b)]
    lse = torch.stack([r[2] for r in refs]).reshape(b * h, sq).contiguous()
    delta = torch.stack([r[3] for r in refs]).reshape(b * h, sq).contiguous()
    _build(head_dim, causal_type, dtype_str="f16")(q, k, v, do, dq, lse, delta, b, sq, seqlen_k=sk)
    torch.cuda.synchronize()
    for z in range(b):
        _check(f"hd={head_dim} f16 ct={causal_type} batch {z}", dq[z], refs[z][0], refs[z][4])


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
