#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""flyc dK/dV under `principal_diagonal_choice` (VarlenBits bit 18), JIT.

With `d = seqlen_k - seqlen_q` for each sequence and `p` the bit, a numeric
window bound becomes `L = window_left - p*d`, `R = window_right + p*d`;
sentinels resolve as they always have and are never shifted. Row `i` keeps
column `j` iff `i - L <= j <= i + R`. See `docs/Generalized Sliding Window
Attention.md` A-2.

The bit travels in the VarlenBits word and nowhere else: a dense call passes
`varlen=abi.varlen_dense(principal_diagonal=...)`, a varlen call sets it in
`varlen["bits"]`.

dK/dV is the kernel where a shifted bound is most likely to go wrong, because
it walks the *transpose*: for each KV block it derives the first and last live
Q row from the window, and a shifted band can sit anywhere -- `L` negative,
`R` past `seqlen_k`, a band that touches no diagonal at all, a KV block that
no row sees (whose dK and dV must then be exactly zero, not unwritten). The
cases below cover each of those against an fp32 autograd reference, for MHA
in bf16 and GQA in f16.

LSE and Delta are computed in torch rather than by the forward kernel, so a
failure here is the backward's alone. A row with no live key gets `LSE = +inf`,
the forward's contract (see `test_fwd_masked_lse.py`).

Builds for the first GPU's arch through `tuning_dkdv`; skips on an arch it has
no row for, or without `AOTRITON_FLYDSL_KERNEL_ROOT`.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/tests/test_bwd_dkdv_principal_diagonal.py
"""

import functools
import math
import sys
import zlib
from pathlib import Path

import pytest

import jit_arch
import tuning_dkdv
from fmha_meta import FmhaInputMetadata

torch = pytest.importorskip("torch")
ARCH = jit_arch.require(tuning_dkdv)

import fmha_utils as abi  # noqa: E402

TL = abi.WINDOW_TOPLEFT
BR = abi.WINDOW_BOTRIGHT
_SENTINELS = (TL, BR)

# `(dtype, num_heads, num_kv_heads)`. GQA folds the group into the kernel's Q
# loop; MHA is the one-head group.
CONFIGS = {
    "bf16_mha": ("bf16", 2, 2),
    "f16_gqa": ("f16", 4, 2),
}
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
# bf16: max abs error relative to the reference's max magnitude (floored at 1);
# bf16 inputs and a bf16-packed P/dS feed the GEMMs. f16: norm-relative, which
# measures ~3e-4.
BF16_REL_TOL = 2e-2
F16_RTOL = 2e-3

# `(head_dim, knob variant)`. gfx950 picks its 32-row MFMA family at 64 and the
# 16-row one at 128; 64 is also forced to 16 there, so both families run at
# one width.
BUILDS = {
    "d64": (64, None),
    "d64_mfma_rows16": (64, ("gfx950", {"mfma_rows": 16})),
    "d128": (128, None),
}

# Either sentinel resolves an unbounded left edge the same way; both are here.
WINDOWS = {
    "TL_0": (TL, 0),
    "BR_0": (BR, 0),
    "8_0": (8, 0),
    "0_8": (0, 8),
    "4_4": (4, 4),
    "neg1_3": (-1, 3),
}

# `(seqlen_q, seqlen_k)`: q shorter, equal, longer, decode, and chunked prefills
# whose chunk does not fill a Q tile.
DENSE_SHAPES = [
    (1, 77),
    (1, 200),
    (32, 160),
    (40, 96),
    (48, 160),
    (64, 64),
    (96, 40),
    (128, 128),
    (160, 48),
]

# Packed batches mixing all three orderings, decode and chunked prefill.
VARLEN_BATCHES = {
    "mixed5": ([40, 64, 100, 1, 16], [100, 64, 40, 77, 130]),
    "mixed4": ([40, 64, 96, 3], [96, 64, 40, 130]),
    "decode": ([1, 1, 1], [77, 5, 160]),
    "chunked_prefill": ([48, 16, 33], [160, 80, 33]),
}


def _shape_id(s):
    return f"q{s[0]}_k{s[1]}"


@functools.cache
def _build(build, config):
    head_dim, variant = BUILDS[build]
    dtype_str, num_heads, num_kv_heads = CONFIGS[config]
    meta = FmhaInputMetadata(num_heads=num_heads,
                             num_kv_heads=num_kv_heads,
                             block_dmodel=None,
                             head_dim=head_dim,
                             dtype_str=dtype_str,
                             causal_type=3)
    return jit_arch.build(tuning_dkdv, ARCH, meta, variant)[0]


def _resolve(window, seqlen_q, seqlen_k, p):
    """`(L, R)` for one sequence -- the Python mirror of `fmha.resolve_window`."""
    wl, wr = window
    d = seqlen_k - seqlen_q
    left = seqlen_q if wl in _SENTINELS else wl - p * d
    if wr in _SENTINELS:
        right = 0 if wr == TL else d
    else:
        right = wr + p * d
    return left, right


def _band(seqlen_q, seqlen_k, window, p, device):
    """`[seqlen_q, seqlen_k]` bool: the live elements, by the spec's predicate."""
    i = torch.arange(seqlen_q, device=device)[:, None]
    j = torch.arange(seqlen_k, device=device)[None, :]
    left, right = _resolve(window, seqlen_q, seqlen_k, p)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, window, p, sm_scale):
    """fp32 forward + autograd: `(lse, delta, dk, dv, col_live)`.

    `q`, `k`, `v`, `do` are one sequence, `(1, H, S, D)`. `col_live[j]` says
    whether any row sees KV column `j`; the kernel must write exact zeros for
    the columns that are not.
    """
    seqlen_q, seqlen_k = q.shape[2], k.shape[2]
    rep = q.shape[1] // k.shape[1]
    qf = q.float()
    kf = k.float().requires_grad_()
    vf = v.float().requires_grad_()
    s = (qf @ kf.repeat_interleave(rep, dim=1).transpose(-1, -2)) * sm_scale
    band = _band(seqlen_q, seqlen_k, window, p, q.device)
    s = s.masked_fill(~band, float("-inf"))
    live = band.any(-1)[None, None, :, None]
    prob = torch.where(live, torch.softmax(s, -1), torch.zeros_like(s))
    o = prob @ vf.repeat_interleave(rep, dim=1)
    lse = torch.logsumexp(s, -1)
    lse = torch.where(live.squeeze(-1), lse, torch.full_like(lse, float("inf")))
    dk, dv = torch.autograd.grad(o, (kf, vf), do.float())
    delta = (do.float() * o.detach()).sum(-1)
    return lse.detach(), delta, dk, dv, band.any(0)


def _rel(got, want, floor):
    den = max(want.float().norm().item(), floor, 1e-30)
    return ((got.float() - want.float()).norm() / den).item()


def _assert_close(dk, dv, ref_k, ref_v, dead_k, dtype_str, label):
    """Tolerance on both gradients, and exact zeros on columns no row sees.

    `dead_k` broadcasts over `(B, H, S_k, D)` along `S_k`. Taken from the band,
    not from `ref == 0`: a live column can have an analytically zero dK (a lone
    live key has `P = 1`, so `dS = 0`), which the kernel reproduces only to
    rounding.
    """
    for name, got, ref in (("dk", dk, ref_k), ("dv", dv, ref_v)):
        assert not bool(torch.isnan(got).any()), f"{label}: {name} has NaN (unwritten or poisoned)"
        assert bool(torch.isfinite(got).all()), f"{label}: {name} not finite"
        if dtype_str == "bf16":
            scale = max(1.0, ref.abs().max().item())
            err = (got.float() - ref).abs().max().item()
            assert err <= BF16_REL_TOL * scale, f"{label}: {name} max abs error {err:.3e} against ref scale {scale:.3e}"
        else:
            # A window can make one gradient analytically zero, so judge "zero"
            # on the problem's scale.
            floor = 1e-3 * max(ref_k.norm().item(), ref_v.norm().item())
            r = _rel(got, ref, floor)
            assert r < F16_RTOL, f"{label}: {name} rel={r:.3e}"
        assert (got.masked_select(dead_k) == 0).all(), f"{label}: {name} non-zero on a column no row sees"


def _rand(shape, dtype_str, gen):
    return torch.randn(*shape, dtype=DTYPES[dtype_str], device="cuda", generator=gen)


def _pd(p):
    """The `abi.VARLEN_PRINCIPAL_DIAGONAL_*` value for bit value `p`."""
    return abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT


def _seed(*key):
    return zlib.crc32(repr(key).encode())


def _dense_inputs(build, config, seqlen_q, seqlen_k, batch, seed):
    head_dim, _ = BUILDS[build]
    dtype_str, hq, hk = CONFIGS[config]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = _rand((batch, hq, seqlen_q, head_dim), dtype_str, gen)
    k = _rand((batch, hk, seqlen_k, head_dim), dtype_str, gen)
    v = _rand((batch, hk, seqlen_k, head_dim), dtype_str, gen)
    do = _rand((batch, hq, seqlen_q, head_dim), dtype_str, gen)
    return q, k, v, do


def _dense_ref(q, k, v, do, window, p, sm_scale):
    """Per-batch references: `(lse, delta, ref_k, ref_v, col_live)`, batched."""
    batch, hq, seqlen_q = q.shape[:3]
    lse = torch.empty(batch, hq, seqlen_q, dtype=torch.float32, device="cuda")
    delta = torch.empty_like(lse)
    ref_k = torch.empty(k.shape, dtype=torch.float32, device="cuda")
    ref_v = torch.empty_like(ref_k)
    col_live = None
    for b in range(batch):
        sl = slice(b, b + 1)
        lse[sl], delta[sl], ref_k[sl], ref_v[sl], col_live = _reference(q[sl], k[sl], v[sl], do[sl], window, p, sm_scale)
    return lse, delta, ref_k, ref_v, col_live


def _dense_call(fn, q, k, v, do, lse, delta, window, varlen, scale, lse_layout_th=False):
    """One dense call; `lse`/`delta` are `(B, H, S)`. Returns `(dk, dv)`."""
    batch, hq, seqlen_q = q.shape[:3]
    if lse_layout_th:
        lse = lse.transpose(1, 2).reshape(batch * seqlen_q, hq)
        delta = delta.transpose(1, 2).reshape(batch * seqlen_q, hq)
    else:
        lse = lse.reshape(batch * hq, seqlen_q)
        delta = delta.reshape(batch * hq, seqlen_q)
    # NaN, so an unwritten element cannot pass for a written zero.
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       lse.contiguous(),
       delta.contiguous(),
       batch,
       seqlen_q,
       seqlen_k=k.shape[2],
       scale=scale,
       window=window,
       varlen=varlen)
    torch.cuda.synchronize()
    return dk, dv


@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape", DENSE_SHAPES, ids=_shape_id)
def test_dense(build, config, p, window_name, shape):
    fn = _build(build, config)
    window = WINDOWS[window_name]
    seqlen_q, seqlen_k = shape
    head_dim = BUILDS[build][0]
    sm_scale = 1.0 / math.sqrt(head_dim)
    q, k, v, do = _dense_inputs(build, config, seqlen_q, seqlen_k, 2, _seed(head_dim, p, window, shape))
    lse, delta, ref_k, ref_v, col_live = _dense_ref(q, k, v, do, window, p, sm_scale)
    dk, dv = _dense_call(fn, q, k, v, do, lse, delta, window, abi.varlen_dense(principal_diagonal=_pd(p)), sm_scale)
    dead_k = (~col_live).view(1, 1, seqlen_k, 1)
    _assert_close(dk, dv, ref_k, ref_v, dead_k, CONFIGS[config][0], f"dense {_shape_id(shape)} {window_name} p={p}")


@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("config", list(CONFIGS))
def test_dense_word_th_layout(build, config):
    """A dense word may select the TH LSE layout: (B*S, H) rows, same dK/dV as HT.

    `seqlen_q != num_heads`, so a row check that assumed HT would reject it.
    """
    fn = _build(build, config)
    seqlen_q, seqlen_k = 48, 160
    window, p = (4, 4), 1
    head_dim = BUILDS[build][0]
    sm_scale = 1.0 / math.sqrt(head_dim)
    q, k, v, do = _dense_inputs(build, config, seqlen_q, seqlen_k, 2, _seed(head_dim, "th"))
    lse, delta, ref_k, ref_v, col_live = _dense_ref(q, k, v, do, window, p, sm_scale)
    outs = []
    for layout in (abi.VARLEN_LSE_LAYOUT_HT, abi.VARLEN_LSE_LAYOUT_TH):
        varlen = abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT, lse_layout=layout)
        outs.append(_dense_call(fn,
                                q,
                                k,
                                v,
                                do,
                                lse,
                                delta,
                                window,
                                varlen,
                                sm_scale,
                                lse_layout_th=layout == abi.VARLEN_LSE_LAYOUT_TH))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    dead_k = (~col_live).view(1, 1, seqlen_k, 1)
    _assert_close(*outs[1], ref_k, ref_v, dead_k, CONFIGS[config][0], "dense TH")


@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape", DENSE_SHAPES, ids=_shape_id)
def test_dense_varlen_none_is_word_0(build, config, window_name, shape):
    """`varlen=None` is the dense top-left word: bit-identical to `varlen_dense()`.

    `scale=None` too, so the build's own default scale is exercised.
    """
    fn = _build(build, config)
    window = WINDOWS[window_name]
    seqlen_q, seqlen_k = shape
    head_dim = BUILDS[build][0]
    q, k, v, do = _dense_inputs(build, config, seqlen_q, seqlen_k, 2, _seed(head_dim, window, shape, "word0"))
    lse, delta, _, _, _ = _dense_ref(q, k, v, do, window, 0, 1.0 / math.sqrt(head_dim))
    a = _dense_call(fn, q, k, v, do, lse, delta, window, None, None)
    b = _dense_call(fn, q, k, v, do, lse, delta, window, abi.varlen_dense(), None)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("window_name", [n for n, w in WINDOWS.items() if w[0] not in _SENTINELS])
@pytest.mark.parametrize("shape", DENSE_SHAPES, ids=_shape_id)
def test_dense_p1_equals_host_shift(build, config, window_name, shape):
    """Dense `p = 1` is `p = 0` with `(wl - d, wr + d)`: bitwise, same binary."""
    fn = _build(build, config)
    window = WINDOWS[window_name]
    seqlen_q, seqlen_k = shape
    head_dim = BUILDS[build][0]
    d = seqlen_k - seqlen_q
    shifted = (window[0] - d, window[1] + d)
    q, k, v, do = _dense_inputs(build, config, seqlen_q, seqlen_k, 2, _seed(head_dim, window, shape, "shift"))
    lse, delta, _, _, _ = _dense_ref(q, k, v, do, window, 1, 1.0 / math.sqrt(head_dim))
    a = _dense_call(fn, q, k, v, do, lse, delta, window, abi.varlen_dense(principal_diagonal=_pd(1)), None)
    b = _dense_call(fn, q, k, v, do, lse, delta, shifted, abi.varlen_dense(principal_diagonal=_pd(0)), None)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _cu(lens):
    out = [0]
    for x in lens:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("batch_name", list(VARLEN_BATCHES))
def test_varlen(build, config, p, window_name, batch_name):
    """Packed 1THD, `d` differing per sequence, the bit in `varlen["bits"]`."""
    fn = _build(build, config)
    window = WINDOWS[window_name]
    lens_q, lens_k = VARLEN_BATCHES[batch_name]
    head_dim = BUILDS[build][0]
    dtype_str, hq, hk = CONFIGS[config]
    sm_scale = 1.0 / math.sqrt(head_dim)
    cq, ck = _cu(lens_q), _cu(lens_k)
    n = len(lens_q)
    total_q, total_k = int(cq[-1]), int(ck[-1])
    gen = torch.Generator(device="cuda").manual_seed(_seed(head_dim, p, window, batch_name))
    q = _rand((1, hq, total_q, head_dim), dtype_str, gen)
    k = _rand((1, hk, total_k, head_dim), dtype_str, gen)
    v = _rand((1, hk, total_k, head_dim), dtype_str, gen)
    do = _rand((1, hq, total_q, head_dim), dtype_str, gen)

    lse = torch.empty(hq, total_q, dtype=torch.float32, device="cuda")
    delta = torch.empty_like(lse)
    ref_k = torch.empty(k.shape, dtype=torch.float32, device="cuda")
    ref_v = torch.empty_like(ref_k)
    col_live = torch.empty(total_k, dtype=torch.bool, device="cuda")
    for z in range(n):
        sq = slice(int(cq[z]), int(cq[z + 1]))
        sk = slice(int(ck[z]), int(ck[z + 1]))
        r = _reference(q[:, :, sq], k[:, :, sk], v[:, :, sk], do[:, :, sq], window, p, sm_scale)
        lse[:, sq], delta[:, sq] = r[0][0], r[1][0]
        ref_k[:, :, sk], ref_v[:, :, sk] = r[2], r[3]
        col_live[sk] = r[4]

    varlen = abi.varlen_compact(cq, ck, max(lens_q), max(lens_k), lse_tokens=total_q)
    varlen["bits"] = abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                     abi.VARLEN_COMPACT_SIDE,
                                     principal_diagonal=_pd(p))
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
       num_seqlens=n,
       scale=sm_scale,
       window=window,
       varlen=varlen)
    torch.cuda.synchronize()
    dead_k = (~col_live).view(1, 1, total_k, 1)
    _assert_close(dk, dv, ref_k, ref_v, dead_k, dtype_str, f"varlen {batch_name} {window_name} p={p}")


@pytest.mark.parametrize("key", ("seqinfo_q0", "seqinfo_k0"))
def test_dense_word_rejects_seqinfo(key):
    """A dense word reads no sequence info; handing it some is a caller error."""
    fn = _build("d64", "f16_gqa")
    dtype_str, hq, hk = CONFIGS["f16_gqa"]
    t = torch.zeros((1, hq, 8, 64), dtype=DTYPES[dtype_str], device="cuda")
    tk = torch.zeros((1, hk, 8, 64), dtype=DTYPES[dtype_str], device="cuda")
    rows = torch.zeros((hq, 8), dtype=torch.float32, device="cuda")
    varlen = abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)
    varlen[key] = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="dense VarlenBits word"):
        fn(t,
           tk,
           tk,
           t,
           tk.clone(),
           tk.clone(),
           rows,
           rows.clone(),
           1,
           8,
           window=(4, 4),
           varlen=varlen)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
