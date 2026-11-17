#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`principal_diagonal_choice` (VarlenBits bit 18) in the gfx1201 dK/dV kernel, JIT.

With `p = principal_diagonal_choice` and `d = seqlen_k - seqlen_q` of each
sequence, a **numeric** window bound `w` resolves to `L = w_left - p*d` and
`R = w_right + p*d`; sentinels resolve as before and are never shifted. Row `i`
keeps column `j` iff `i - L <= j <= i + R`. See
`docs/Generalized Sliding Window Attention.md`, A-2.

The bit travels in the VarlenBits word and nowhere else: a dense call passes
`varlen=abi.varlen_dense(principal_diagonal=...)`, a varlen call sets it in
`varlen["bits"]`.

dK/dV is the kernel where a shifted bound is most likely to go wrong, because
it walks the *transpose*: for each KV block it derives the first and last live
Q row from the window, and a shifted band can sit anywhere -- `L` negative,
`R` past `seqlen_k`, a band that touches no diagonal at all, a KV block that
no row sees (whose dK and dV must then be exactly zero, not unwritten). The
cases below cover each of those against an fp32 autograd reference.

LSE and Delta are computed in torch rather than by the forward kernel, so a
failure here is the backward's alone. A row with no live key gets `LSE = +inf`,
the forward's placeholder, which makes `exp(s - lse) = 0`.

Requires a gfx1201 device, `AOTRITON_FLYDSL_KERNEL_ROOT`, and a ROCm the
FlyDSL compiler can link against; skips cleanly without any of them.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
    FLYDSL_EXTRA_SOURCE_DIRS=$PWD/modules/flash/flyc \\
        python -m pytest modules/flash/flyc/test_bwd_dkdv_principal_diagonal_gfx1201.py
"""

import functools
import math
import os
import sys

import pytest

# See test_fwd_masked_lse_gfx950.py: the vendored kernels are this directory,
# which pytest puts on sys.path; the FlyDSL checkout supplies `kernels.common`.
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")


def _is_gfx1201():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.lower().startswith("gfx1201")


pytestmark = pytest.mark.skipif(
    not _KERNEL_ROOT or not _is_gfx1201(),
    reason="needs a gfx1201 device and AOTRITON_FLYDSL_KERNEL_ROOT",
)

# Sentinels, as the kernel compares them. Duplicated rather than imported so
# that collecting this file does not import flydsl on a machine that skips it.
TOPLEFT = -2147483647  # 0x80000001
BOTRIGHT = -2147483646  # 0x80000002
# VarlenBits bit 18.
PD_SHIFT = 18

HEAD_DIM = 64
NUM_HEADS_Q = 4
NUM_HEADS_K = 2  # GQA: the kernel folds the group into its Q loop
DTYPE = torch.float16
# See the FlyDSL parity suite's dK/dV test: f16 measures ~3e-4 relative.
RTOL = 2e-3

WINDOWS = [
    pytest.param((BOTRIGHT, 0), id="sentinel_left-0"),
    pytest.param((8, 0), id="8-0"),
    pytest.param((0, 8), id="0-8"),
    pytest.param((4, 4), id="4-4"),
    pytest.param((-1, 3), id="neg1-3"),
]

# (seqlen_q, seqlen_k): <, ==, >, decode, chunked prefill.
DENSE_SHAPES = [
    pytest.param(40, 96, id="q_lt_k"),
    pytest.param(64, 64, id="q_eq_k"),
    pytest.param(96, 40, id="q_gt_k"),
    pytest.param(1, 77, id="decode"),
    pytest.param(48, 160, id="chunked_prefill"),
]

# Packed batches mixing all three orderings, and the two regression shapes.
VARLEN_BATCHES = [
    pytest.param([40, 64, 96, 3], [96, 64, 40, 130], id="mixed"),
    pytest.param([1, 1, 1], [77, 5, 160], id="decode"),
    pytest.param([48, 16, 33], [160, 80, 33], id="chunked_prefill"),
]


def _resolve(window, seqlen_q, seqlen_k, p):
    """`(L, R)` for one sequence -- the Python mirror of `fmha.resolve_window`."""
    wl, wr = window
    d = seqlen_k - seqlen_q
    left = seqlen_q if wl in (TOPLEFT, BOTRIGHT) else wl - p * d
    if wr in (TOPLEFT, BOTRIGHT):
        right = 0 if wr == TOPLEFT else d
    else:
        right = wr + p * d
    return left, right


def _band(seqlen_q, seqlen_k, window, p, device):
    i = torch.arange(seqlen_q, device=device)[:, None]
    j = torch.arange(seqlen_k, device=device)[None, :]
    left, right = _resolve(window, seqlen_q, seqlen_k, p)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, window, p):
    """fp32 forward + autograd: `(lse, delta, dk, dv, col_live)`.

    `q`, `k`, `v`, `do` are one sequence, `(1, H, S, D)`. `col_live[j]` says
    whether any row sees KV column `j`; the kernel must write exact zeros
    for the columns that are not.
    """
    seqlen_q, seqlen_k = q.shape[2], k.shape[2]
    sm_scale = 1.0 / math.sqrt(q.shape[3])
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


@functools.lru_cache(maxsize=None)
def _exe():
    from fmha_bwd_dkdv_gfx1201_kernel import build_bwd_dkdv_module

    return build_bwd_dkdv_module(
        num_heads=NUM_HEADS_Q,
        head_dim=HEAD_DIM,
        causal=True,
        causal_type=3,
        dtype_str="f16",
    )


def _pd(p):
    """The `abi.VARLEN_PRINCIPAL_DIAGONAL_*` value for bit value `p`."""
    import fmha_abi_gfx1201 as abi

    return abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT


def _dense_word(p):
    """A dense VarlenBits word carrying bit value `p`."""
    import fmha_abi_gfx1201 as abi

    return abi.varlen_dense(principal_diagonal=_pd(p))


def _rand(shape, gen):
    return torch.randn(*shape, dtype=DTYPE, device="cuda", generator=gen)


def _rel(got, want, floor):
    den = max(want.float().norm().item(), floor, 1e-30)
    return ((got.float() - want.float()).norm() / den).item()


def _assert_close(dk, dv, ref_k, ref_v, dead_k, label):
    """Tolerance on both gradients, and exact zeros on columns no row sees."""
    assert torch.isfinite(dk).all(), f"{label}: dk not finite (unwritten or NaN)"
    assert torch.isfinite(dv).all(), f"{label}: dv not finite (unwritten or NaN)"
    # A window can make one gradient analytically zero (e.g. a single live
    # element per row gives dS = 0), so judge "zero" on the problem's scale.
    floor = 1e-3 * max(ref_k.norm().item(), ref_v.norm().item())
    r_dk, r_dv = _rel(dk, ref_k, floor), _rel(dv, ref_v, floor)
    assert r_dk < RTOL, f"{label}: dk rel={r_dk:.3e}"
    assert r_dv < RTOL, f"{label}: dv rel={r_dv:.3e}"
    # `dead_k` broadcasts over (B, H, S_k, D) along S_k.
    assert (dk.masked_select(dead_k) == 0).all(), f"{label}: dk non-zero on a column no row sees"
    assert (dv.masked_select(dead_k) == 0).all(), f"{label}: dv non-zero on a column no row sees"


def _dense_run(seqlen_q, seqlen_k, window, p, varlen, batch=2, seed=0):
    """One dense call. `varlen` (None or a dense word) is what the launcher is passed;
    `p` is the bit value the reference is built for."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = _rand((batch, NUM_HEADS_Q, seqlen_q, HEAD_DIM), gen)
    k = _rand((batch, NUM_HEADS_K, seqlen_k, HEAD_DIM), gen)
    v = _rand((batch, NUM_HEADS_K, seqlen_k, HEAD_DIM), gen)
    do = _rand((batch, NUM_HEADS_Q, seqlen_q, HEAD_DIM), gen)

    lse = torch.empty(batch, NUM_HEADS_Q, seqlen_q, dtype=torch.float32, device="cuda")
    delta = torch.empty_like(lse)
    ref_k = torch.empty(k.shape, dtype=torch.float32, device="cuda")
    ref_v = torch.empty_like(ref_k)
    col_live = None
    for b in range(batch):
        sl = slice(b, b + 1)
        _l, _d, _gk, _gv, col_live = _reference(q[sl], k[sl], v[sl], do[sl], window, p)
        lse[sl], delta[sl], ref_k[sl], ref_v[sl] = _l, _d, _gk, _gv

    # NaN, so an unwritten element cannot pass for a written zero.
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    _exe()(
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse.reshape(batch * NUM_HEADS_Q, seqlen_q).contiguous(),
        delta.reshape(batch * NUM_HEADS_Q, seqlen_q).contiguous(),
        batch,
        seqlen_q,
        seqlen_k=seqlen_k,
        window=window,
        varlen=varlen,
    )
    torch.cuda.synchronize()
    dead_k = (~col_live).view(1, 1, seqlen_k, 1)
    return dk, dv, ref_k, ref_v, dead_k


@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("seqlen_q,seqlen_k", DENSE_SHAPES)
def test_dense(seqlen_q, seqlen_k, window, p):
    dk, dv, ref_k, ref_v, dead_k = _dense_run(seqlen_q, seqlen_k, window, p, _dense_word(p))
    _assert_close(dk, dv, ref_k, ref_v, dead_k, f"dense sq={seqlen_q} sk={seqlen_k} w={window} p={p}")


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("seqlen_q,seqlen_k", DENSE_SHAPES)
def test_dense_varlen_none_is_word_0(seqlen_q, seqlen_k, window):
    """`varlen=None` is the dense top-left word: bit-identical to `varlen_dense()`."""
    a = _dense_run(seqlen_q, seqlen_k, window, 0, None)
    b = _dense_run(seqlen_q, seqlen_k, window, 0, _dense_word(0))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("window", [w for w in WINDOWS if w.values[0][0] not in (TOPLEFT, BOTRIGHT)])
@pytest.mark.parametrize("seqlen_q,seqlen_k", DENSE_SHAPES)
def test_dense_p1_equals_host_shift(seqlen_q, seqlen_k, window):
    """Dense `p = 1` is `p = 0` with `(wl - d, wr + d)`: bitwise, same binary."""
    d = seqlen_k - seqlen_q
    shifted = (window[0] - d, window[1] + d)
    a = _dense_run(seqlen_q, seqlen_k, window, 1, _dense_word(1))
    b = _dense_run(seqlen_q, seqlen_k, shifted, 0, _dense_word(0))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def _cu(lens):
    out = [0]
    for x in lens:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("lens_q,lens_k", VARLEN_BATCHES)
def test_varlen_compact(lens_q, lens_k, window, p):
    """Packed 1THD, `d` differing per sequence, the bit in `varlen["bits"]`."""
    import fmha_abi_gfx1201 as abi

    cq, ck = _cu(lens_q), _cu(lens_k)
    n = len(lens_q)
    total_q, total_k = int(cq[-1]), int(ck[-1])
    mq, mk = max(lens_q), max(lens_k)
    gen = torch.Generator(device="cuda").manual_seed(7)
    q = _rand((1, NUM_HEADS_Q, total_q, HEAD_DIM), gen)
    k = _rand((1, NUM_HEADS_K, total_k, HEAD_DIM), gen)
    v = _rand((1, NUM_HEADS_K, total_k, HEAD_DIM), gen)
    do = _rand((1, NUM_HEADS_Q, total_q, HEAD_DIM), gen)

    lse = torch.empty(NUM_HEADS_Q, total_q, dtype=torch.float32, device="cuda")
    delta = torch.empty_like(lse)
    ref_k = torch.empty(k.shape, dtype=torch.float32, device="cuda")
    ref_v = torch.empty_like(ref_k)
    col_live = torch.empty(total_k, dtype=torch.bool, device="cuda")
    for z in range(n):
        sq = slice(int(cq[z]), int(cq[z + 1]))
        sk = slice(int(ck[z]), int(ck[z + 1]))
        _l, _d, _gk, _gv, _live = _reference(q[:, :, sq], k[:, :, sk], v[:, :, sk], do[:, :, sq], window, p)
        lse[:, sq], delta[:, sq] = _l[0], _d[0]
        ref_k[:, :, sk], ref_v[:, :, sk] = _gk, _gv
        col_live[sk] = _live

    varlen = abi.varlen_compact(cq, ck, mq, mk, lse_tokens=total_q)
    varlen["bits"] = abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                     abi.VARLEN_COMPACT_SIDE,
                                     principal_diagonal=_pd(p))

    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    _exe()(
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse,
        delta,
        1,
        mq,
        seqlen_k=mk,
        num_seqlens=n,
        window=window,
        varlen=varlen,
    )
    torch.cuda.synchronize()
    dead_k = (~col_live).view(1, 1, total_k, 1)
    _assert_close(dk, dv, ref_k, ref_v, dead_k, f"varlen q={lens_q} k={lens_k} w={window} p={p}")


def test_dense_word_rejects_seqinfo():
    """A dense word reads no sequence info; handing it some is a caller error."""
    t = torch.zeros((1, NUM_HEADS_Q, 8, HEAD_DIM), dtype=DTYPE, device="cuda")
    tk = torch.zeros((1, NUM_HEADS_K, 8, HEAD_DIM), dtype=DTYPE, device="cuda")
    rows = torch.zeros((NUM_HEADS_Q, 8), dtype=torch.float32, device="cuda")
    varlen = _dense_word(1)
    varlen["seqinfo_q0"] = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="dense VarlenBits word"):
        _exe()(t,
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
