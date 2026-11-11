#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The gfx942 dK/dV backward (fmha_bwd_dkdv_gfx942.py), JIT, against fp64.

LSE and Delta come from the fp64 forward here (Delta from the O a forward would
store), so this tests the backward alone. Each case is held to the floor of
docs/attention-kernel-numerical-error-lessons: relative RMS error at most twice
that of the fp64 answer rounded where a correct kernel must round (P before
P^T.dO, dS before dS^T.Q, the stored outputs). KV rows no query sees must come
back exactly zero, and nothing may be NaN.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_bwd_dkdv_gfx942.py
"""

import functools
import os
import sys
import zlib
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

TL = fmha.WINDOW_TOPLEFT
BR = fmha.WINDOW_BOTRIGHT
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
FLOOR_FACTOR = 2.0


@functools.cache
def _build(head_dim, causal_type=0, bias=False, dropout=False, padded_head=False, dtype="bf16"):
    from fmha_bwd_dkdv_gfx942 import build_fmha_bwd_dkdv_gfx942

    return build_fmha_bwd_dkdv_gfx942(head_dim,
                                      causal_type=causal_type,
                                      bias=bias,
                                      dropout=dropout,
                                      padded_head=padded_head,
                                      dtype_str=dtype)


def _seed(*key):
    torch.manual_seed(zlib.crc32(repr(key).encode()))


def _band(window, p, sq, sk, device="cuda"):
    """[sq, sk] live mask: i - left <= j <= i + right, sentinels resolved as the kernel does."""
    if window is None:
        return None
    wl, wr = window
    d = sk - sq
    left = sq if wl in (TL, BR) else wl - p * d
    right = {TL: 0, BR: d}.get(wr, wr + p * d if wr not in (TL, BR) else 0)
    i = torch.arange(sq, device=device).view(-1, 1)
    j = torch.arange(sk, device=device).view(1, -1)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, scale, dt, live=None, bias=None, keep=None, p=0.0):
    """fp64 over [..., H, S, D]: (dK, dV, dK floor, dV floor, LSE, Delta).

    A row with no live key has LSE +inf and contributes nothing, the forward's contract."""
    g = q.shape[-3] // k.shape[-3]
    qd, dod = q.double(), do.double()
    kd = k.double().repeat_interleave(g, -3)
    vd = v.double().repeat_interleave(g, -3)
    s = qd @ kd.transpose(-1, -2) * scale
    if bias is not None:
        s = s + bias.double()
    if live is not None:
        s = s.masked_fill(~live, float("-inf"))
    lse = torch.logsumexp(s, -1)
    lse = torch.where(torch.isneginf(lse), torch.full_like(lse, float("inf")), lse)
    pm = torch.exp(s - lse[..., None]).nan_to_num()
    pd = pm if keep is None else pm * keep / (1 - p)
    o = pd @ vd
    delta = (o.to(dt).double() * dod).sum(-1).float().double()
    dp = dod @ vd.transpose(-1, -2)
    if keep is not None:
        dp = dp * keep / (1 - p)
    ds = pm * (dp - delta[..., None])

    def fold(x):
        return x.unflatten(-3, (k.shape[-3], g)).sum(-3)

    def rnd(x):
        return x.to(dt).double()

    dv = fold(pd.transpose(-1, -2) @ dod)
    dk = fold(scale * ds.transpose(-1, -2) @ qd)
    dv_f = rnd(fold(rnd(pd).transpose(-1, -2) @ dod))
    dk_f = rnd(fold(scale * rnd(ds).transpose(-1, -2) @ qd))
    return dk, dv, dk_f, dv_f, lse.float(), delta.float()


def _check(tag, got, exact, floor, unseen=None):
    """`got` against `exact` within FLOOR_FACTOR of the floor's error; `unseen` kv rows exactly 0."""
    got = got.double()
    assert not bool(torch.isnan(got).any()), f"{tag}: {int(torch.isnan(got).sum())} NaN"
    norm = exact.norm().item()
    err = (got - exact).norm().item()
    if norm == 0.0:
        assert err == 0.0, f"{tag}: expected all zero, got norm {err:.3e}"
    else:
        # The absolute term covers gradients that are zero analytically: a lone live
        # key has P = 1, so dS = 0 up to the rounding of dP - Delta.
        fl = (floor - exact).norm().item()
        assert err <= FLOOR_FACTOR * fl + 1e-5 * exact.numel() ** 0.5, \
            f"{tag}: rel RMS {err / norm:.3e} vs floor {fl / norm:.3e}"
    if unseen is not None and bool(unseen.any()):
        stray = got[..., unseen, :].abs().max().item()
        assert stray == 0.0, f"{tag}: KV rows no query sees carry {stray:.3e}"


def _rand(*shape, dt=torch.bfloat16, scale=1.0, uniform=False):
    if uniform:
        return torch.rand(shape, device="cuda", dtype=torch.float32).to(dt)
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * scale).to(dt)


def _run_dense(fn, q, k, v, do, scale, lse, delta, window=None, principal_diagonal=None, **kw):
    b, h, sq, _ = q.shape
    sk = k.shape[2]
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       lse.reshape(b * h, sq).contiguous(),
       delta.reshape(b * h, sq).contiguous(),
       b,
       sq,
       seqlen_k=sk,
       scale=scale,
       window=window,
       principal_diagonal=principal_diagonal,
       **kw)
    torch.cuda.synchronize()
    return dk, dv


def _dense_case(head_dim, sq, sk, h=2, hk=None, window=None, p=0, dtype="bf16", scale=None, qk_scale=1.0,
                uniform=False, causal_type=None):
    hk = hk or h
    dt = DTYPES[dtype]
    if causal_type is None:
        causal_type = 0 if window is None else 3
    fn = _build(head_dim, causal_type=causal_type, dtype=dtype)
    _seed(head_dim, sq, sk, h, hk, window, p, dtype, scale, qk_scale, uniform)
    b = 2
    q = _rand(b, h, sq, head_dim, dt=dt, scale=qk_scale, uniform=uniform)
    k = _rand(b, hk, sk, head_dim, dt=dt, scale=qk_scale, uniform=uniform)
    v = _rand(b, hk, sk, head_dim, dt=dt, uniform=uniform)
    do = _rand(b, h, sq, head_dim, dt=dt, uniform=uniform)
    scale = head_dim ** -0.5 if scale is None else scale
    host_window = window if causal_type == 3 else None
    live = _band(window, p, sq, sk)
    dk_r, dv_r, dk_f, dv_f, lse, delta = _reference(q, k, v, do, scale, dt, live=live)
    pd = None
    if causal_type == 3:
        pd = abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT
    dk, dv = _run_dense(fn, q, k, v, do, scale, lse, delta, window=host_window, principal_diagonal=pd)
    unseen = None if live is None else ~live.any(0)
    tag = f"d{head_dim} {sq}x{sk} h{h}/{hk} w{window} p{p} {dtype}"
    _check(f"dV {tag}", dv, dv_r, dv_f, unseen)
    _check(f"dK {tag}", dk, dk_r, dk_f, unseen)
    return dk, dv, dk_r, dv_r


@pytest.mark.parametrize("shape", [(77, 130), (256, 256), (300, 77), (1, 200), (513, 129)])
@pytest.mark.parametrize("head_dim", (32, 64, 96, 128))
def test_dense(head_dim, shape):
    _dense_case(head_dim, *shape)


@pytest.mark.parametrize("heads", [(4, 2), (6, 1)])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_gqa(head_dim, heads):
    """dK/dV sum over the q heads of a k head."""
    _dense_case(head_dim, 200, 333, h=heads[0], hk=heads[1])


# Causal alignments as sentinels, and numeric gSWA windows: `L` negative, `R`
# past seqlen_k, a band clear of the diagonal, KV columns no row sees.
WINDOWS = ((TL, TL), (BR, BR), (TL, 0), (8, 0), (0, 8), (4, 4), (-1, 3), (40, -20), (300, 300))
CAUSAL_SHAPES = ((48, 160), (128, 128), (160, 48), (1, 200), (301, 190), (257, 519))


@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w[0]}_{w[1]}")
@pytest.mark.parametrize("shape", CAUSAL_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("head_dim", (64, 128))
def test_window(head_dim, shape, window, p):
    _dense_case(head_dim, *shape, window=window, p=p)


@pytest.mark.parametrize("head_dim", (32, 96))
def test_window_other_rungs(head_dim):
    _dense_case(head_dim, 301, 190, window=(BR, BR))
    _dense_case(head_dim, 257, 519, window=(8, 3), p=1)


def test_host_causal_sentinel():
    """causal_type=2 resolves bottom-right on the host path; dead rows included."""
    _dense_case(128, 301, 190, window=(BR, BR), causal_type=2)


@pytest.mark.parametrize("case", ["sharp", "zero_scale", "negative_scale", "f16", "f16_sharp"])
def test_numerics(case):
    """test_common_mistakes' inputs: large logits, sm_scale 0 and negative, and f16."""
    if case == "sharp":
        _dense_case(128, 257, 519, qk_scale=4.0)
    elif case == "zero_scale":
        _dense_case(128, 257, 519, window=(TL, TL), scale=0.0)
    elif case == "negative_scale":
        _dense_case(128, 257, 519, window=(TL, TL), scale=-(128 ** -0.5))
    elif case == "f16":
        _dense_case(64, 300, 777, dtype="f16")
    else:
        _dense_case(128, 257, 519, dtype="f16", qk_scale=4.0, window=(BR, BR))


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_one_signed(head_dim, dtype):
    """U[0, 1) inputs: P must round to nearest, so dV carries no bias (lessons, #16)."""
    _, dv, _, dv_r = _dense_case(head_dim, 143, 579, uniform=True, dtype=dtype)
    eps = 2.0 ** -8 if dtype == "bf16" else 2.0 ** -11
    bias = ((dv.double() - dv_r).mean() / dv_r.abs().mean()).item()
    assert abs(bias) <= eps / 8, f"dV mean signed error {bias:+.2e} of its magnitude, bound {eps / 8:.1e}"


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("window", [None, (BR, BR), (16, 4)], ids=["dense", "br", "w16_4"])
@pytest.mark.parametrize("lse_layout", ["ht", "th"])
@pytest.mark.parametrize("layout", ["compact", "padded"])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_varlen(head_dim, layout, lse_layout, window):
    """Compact packs sequences into (1, H, T, D); padded keeps (B, H, max, D)."""
    h, hk = 4, 2
    dt = torch.bfloat16
    fn = _build(head_dim, causal_type=0 if window is None else 3)
    lq = [a for a, _ in SEQS]
    lk = [b for _, b in SEQS]
    n = len(SEQS)
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device="cuda", dtype=torch.int32)
    lse_bits = abi.VARLEN_LSE_LAYOUT_HT if lse_layout == "ht" else abi.VARLEN_LSE_LAYOUT_TH
    _seed(head_dim, layout, lse_layout, window)
    scale = head_dim ** -0.5
    if layout == "compact":
        tq, tk = sum(lq), sum(lk)
        q = _rand(1, h, tq, head_dim)
        k = _rand(1, hk, tk, head_dim)
        v = _rand(1, hk, tk, head_dim)
        do = _rand(1, h, tq, head_dim)
        varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = 1, n
        lse = torch.full((h, tq) if lse_layout == "ht" else (tq, h), float("nan"), device="cuda")
    else:
        tq = max(lq)
        q = _rand(n, h, tq, head_dim)
        k = _rand(n, hk, max(lk), head_dim)
        v = _rand(n, hk, max(lk), head_dim)
        do = _rand(n, h, tq, head_dim)
        varlen = abi.varlen_padded(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=lse_bits)
        batch, nseq = n, 0
        lse = torch.full((n * h, tq) if lse_layout == "ht" else (n * tq, h), float("nan"), device="cuda")
    delta = torch.full_like(lse, float("nan"))

    def views(z):
        """Q-side and K-side [H, S, D] views of sequence z, and its LSE/Delta [H, S] views."""
        if layout == "compact":
            qs = slice(int(cu_q[z]), int(cu_q[z + 1]))
            ks = slice(int(cu_k[z]), int(cu_k[z + 1]))
            qv = [t[0, :, qs] for t in (q, do)]
            kv = [t[0, :, ks] for t in (k, v)]
            rv = [x[:, qs] if lse_layout == "ht" else x[qs].T for x in (lse, delta)]
        else:
            sq, sk = SEQS[z]
            qv = [t[z, :, :sq] for t in (q, do)]
            kv = [t[z, :, :sk] for t in (k, v)]
            rv = [x[z * h:(z + 1) * h, :sq] if lse_layout == "ht" else x[z * tq:z * tq + sq].T for x in (lse, delta)]
        return qv, kv, rv

    refs = []
    for z, (sq, sk) in enumerate(SEQS):
        (qz, doz), (kz, vz), (lz, dz) = views(z)
        live = _band(window, 0, sq, sk)
        r = _reference(qz, kz, vz, doz, scale, dt, live=live)
        lz.copy_(r[4])
        dz.copy_(r[5])
        refs.append((r, live))
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    fn(q, k, v, do, dk, dv, lse, delta, batch, max(lq), seqlen_k=max(lk), num_seqlens=nseq, scale=scale,
       window=window, varlen=varlen)
    torch.cuda.synchronize()
    for z, (sq, sk) in enumerate(SEQS):
        (r, live) = refs[z]
        if layout == "compact":
            ks = slice(int(cu_k[z]), int(cu_k[z + 1]))
            dkz, dvz = dk[0, :, ks], dv[0, :, ks]
        else:
            dkz, dvz = dk[z, :, :sk], dv[z, :, :sk]
        unseen = None if live is None else ~live.any(0)
        tag = f"d{head_dim} {layout} {lse_layout} w{window} seq{z}=({sq},{sk})"
        _check(f"dV {tag}", dvz, r[1], r[3], unseen)
        _check(f"dK {tag}", dkz, r[0], r[2], unseen)


@pytest.mark.parametrize("shape", [(300, 777), (301, 1001), (1, 1001), (500, 500)])
@pytest.mark.parametrize("head_dim", (64, 128))
def test_bias(head_dim, shape):
    """(B, H, Sq, Sk) bias, an input here; odd pitches start rows mid-dword."""
    sq, sk = shape
    b, h, hk = 2, 4, 2
    dt = torch.bfloat16
    _seed(head_dim, shape, "bias")
    q = _rand(b, h, sq, head_dim)
    k = _rand(b, hk, sk, head_dim)
    v = _rand(b, hk, sk, head_dim)
    do = _rand(b, h, sq, head_dim)
    bias = _rand(b, h, sq, sk, scale=4.0)
    scale = head_dim ** -0.5
    dk_r, dv_r, dk_f, dv_f, lse, delta = _reference(q, k, v, do, scale, dt, bias=bias)
    dk, dv = _run_dense(_build(head_dim, bias=True), q, k, v, do, scale, lse, delta, bias=bias)
    _check(f"dV d{head_dim} bias {shape}", dv, dv_r, dv_f)
    _check(f"dK d{head_dim} bias {shape}", dk, dk_r, dk_f)


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
    """The forward's mask: offset = base + plane*sq*stride + row*stride + col//4, word col % 4."""
    sq, sk = shape
    b, h, hk, p = 2, 4, 2, 0.3
    seed, off1, off2 = 0x123456789ABCDEF0, 1000, 77
    dt = torch.bfloat16
    _seed(head_dim, shape, causal, "dropout")
    q = _rand(b, h, sq, head_dim)
    k = _rand(b, hk, sk, head_dim)
    v = _rand(b, hk, sk, head_dim)
    do = _rand(b, h, sq, head_dim)
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off1 + off2) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = _philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    keep = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk).double()
    window = (BR, BR) if causal else None
    live = _band(window, 0, sq, sk)
    scale = head_dim ** -0.5
    dk_r, dv_r, dk_f, dv_f, lse, delta = _reference(q, k, v, do, scale, dt, live=live, keep=keep, p=p)
    fn = _build(head_dim, causal_type=3 if causal else 0, dropout=True)
    dk, dv = _run_dense(fn, q, k, v, do, scale, lse, delta, window=window,
                        dropout_p=p,
                        philox_seed=seed,
                        philox_offset1=torch.tensor([off1], dtype=torch.int64, device="cuda"),
                        philox_offset2=off2)
    unseen = None if live is None else ~live.any(0)
    _check(f"dV d{head_dim} dropout {shape} causal={causal}", dv, dv_r, dv_f, unseen)
    _check(f"dK d{head_dim} dropout {shape} causal={causal}", dk, dk_r, dk_f, unseen)


def _with_nan_slack(*shape, dt=torch.bfloat16):
    """A (..., hdim) view of a (..., ceil8(hdim)) tensor whose slack is NaN: the 8xD contract."""
    *lead, d = shape
    full = torch.full((*lead, -(-d // 8) * 8), float("nan"), device="cuda", dtype=dt)
    view = full[..., :d]
    view.copy_(torch.randn(*shape, device="cuda", dtype=dt))
    return view


# (rung, hdim_qk, hdim_vo): primes, odd sizes, and QK != VO, each above the rung below.
PADDED = [(32, 7, 7), (32, 23, 23), (64, 37, 37), (64, 53, 61), (96, 67, 67), (96, 83, 83),
          (128, 113, 113), (128, 120, 72), (128, 97, 128)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rung,hqk,hvo", PADDED)
def test_padded_head(rung, hqk, hvo, causal):
    """Columns past hdim are NaN in every input; none may reach a real dK or dV column."""
    b, h, sq, sk = 2, 2, 257, 519
    if causal:
        sq, sk = sk, sq                 # bottom-right with dead rows
    dt = torch.bfloat16
    _seed(rung, hqk, hvo, causal)
    q = _with_nan_slack(b, h, sq, hqk)
    k = _with_nan_slack(b, h, sk, hqk)
    v = _with_nan_slack(b, h, sk, hvo)
    do = _with_nan_slack(b, h, sq, hvo)
    dk = _with_nan_slack(b, h, sk, hqk)
    dv = _with_nan_slack(b, h, sk, hvo)
    window = (BR, BR) if causal else None
    live = _band(window, 0, sq, sk)
    scale = hqk ** -0.5
    dk_r, dv_r, dk_f, dv_f, lse, delta = _reference(q, k, v, do, scale, dt, live=live)
    fn = _build(rung, causal_type=3 if causal else 0, padded_head=True)
    fn(q, k, v, do, dk, dv, lse.reshape(b * h, sq).contiguous(), delta.reshape(b * h, sq).contiguous(), b, sq,
       seqlen_k=sk, window=window)
    torch.cuda.synchronize()
    unseen = None if live is None else ~live.any(0)
    tag = f"rung={rung} hqk={hqk} hvo={hvo} causal={causal}"
    _check(f"dV {tag}", dv, dv_r, dv_f, unseen)
    _check(f"dK {tag}", dk, dk_r, dk_f, unseen)


def test_principal_diagonal_rejects_garbage():
    """The host refuses a principal_diagonal that is not one of the two bit patterns."""
    fn = _build(64, causal_type=3)
    t = _rand(1, 2, 8, 64)
    lse = torch.zeros(2, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="principal_diagonal"):
        fn(t, t, t, t, t.clone(), t.clone(), lse, lse.clone(), 1, 8, window=(0, 0), principal_diagonal=1)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
