#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`principal_diagonal_choice` (VarlenBits bit 18) in the gfx1201 dQ kernel, JIT.

With `d = seqlen_k - seqlen_q` of each sequence and `p` the bit, a numeric
window side `w` resolves to `L = w_left - p*d`, `R = w_right + p*d`; a
sentinel side resolves as it always has and is never shifted. Row `i` keeps
column `j` iff `0 <= j < seqlen_k` and `i - L <= j <= i + R`. See
`docs/Generalized Sliding Window Attention.md`, A-1 / A-2.

Two groups of tests:

* **Host plumbing** (`test_launcher_*`). Builds the launcher and intercepts
  `abi.run_compiled`, so nothing is compiled or launched: it checks only that
  `principal_diagonal=` reaches the `varlen_bits` kernel argument, for dense
  calls too, and that `None` keeps what a varlen dict already carries. Needs
  FlyDSL and `AOTRITON_FLYDSL_KERNEL_ROOT`, not a GPU.
* **Numerics** (`test_dq_*`). Runs the kernel on a gfx1201 device against a
  float32 autograd reference over `p in {0, 1}`, dense and varlen, with
  `seqlen_q <`, `==` and `> seqlen_k`, decode and chunked prefill. LSE and
  Delta come from torch, not from the forward kernel, so a failure here is
  this kernel's. Rows that see no key carry LSE `+inf` -- the forward's
  contract (see `test_fwd_masked_lse_gfx950.py`) -- and must get dQ = 0.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_bwd_dq_principal_diagonal_gfx1201.py
"""

import math
import os
import sys
from itertools import accumulate
from pathlib import Path

import pytest

# Bare-name imports of the vendored kernels, as in the other tests here; the
# FlyDSL checkout supplies `kernels.common`.
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")

_needs_flydsl = pytest.mark.skipif(not _KERNEL_ROOT, reason="needs AOTRITON_FLYDSL_KERNEL_ROOT")
_needs_gfx1201 = pytest.mark.skipif(
    not _KERNEL_ROOT
    or not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1201"),
    reason="needs a gfx1201 device and AOTRITON_FLYDSL_KERNEL_ROOT",
)

HEAD_DIM = 64
NUM_HEADS = 2
DTYPE = torch.float16
# f16 inputs, and dS is truncated to f16 for the last GEMM (RDNA4 WMMA has no
# f32 x f32 form), against a float32 reference.
ATOL = 2e-2
RTOL = 2e-2

# The sentinel values, spelled here so that parametrize ids are readable and
# collection does not import FlyDSL. Must equal `fmha_common_gfx1201`'s.
WINDOW_TOPLEFT = -2147483647  # 0x80000001
WINDOW_BOTRIGHT = -2147483646  # 0x80000002
_SENTINELS = (WINDOW_TOPLEFT, WINDOW_BOTRIGHT)

# (window_left, window_right). The first is flash-attention's `(-1, 0)`, an
# unbounded left spelled with a sentinel; with p = 1 it is bottom-right causal.
# `(-1, 3)` is a band that never touches the diagonal. The last pair is all
# sentinels, which p must not move.
WINDOWS = [
    (WINDOW_BOTRIGHT, 0),
    (8, 0),
    (0, 8),
    (4, 4),
    (-1, 3),
    (WINDOW_TOPLEFT, WINDOW_BOTRIGHT),
]

# (seqlen_q, seqlen_k). Several BLOCK_M / BLOCK_N tiles each way, with ragged
# tails. The last two are the report's cases: decode, and a chunked prefill
# whose cached prefix is 192 tokens.
DENSE_SHAPES = [
    (96, 200),
    (160, 160),
    (200, 72),
    (1, 200),
    (64, 256),
]

# One varlen batch mixing all three orders, plus decode and chunked prefill.
VARLEN_SEQLENS = [
    (37, 100),
    (64, 64),
    (130, 50),
    (1, 77),
    (48, 240),
]


def _window_id(w):
    names = {WINDOW_TOPLEFT: "TL", WINDOW_BOTRIGHT: "BR"}
    return "w" + "_".join(names.get(x, str(x)) for x in w)


def _resolve(window, seqlen_q, seqlen_k, p):
    """`(L, R)` per A-2: numeric sides shift by `p * d`, sentinels do not."""
    wl, wr = window
    d = seqlen_k - seqlen_q
    left = seqlen_q if wl in _SENTINELS else wl - p * d
    if wr in _SENTINELS:
        right = 0 if wr == WINDOW_TOPLEFT else d
    else:
        right = wr + p * d
    return left, right


def _keep_mask(window, seqlen_q, seqlen_k, p, device):
    left, right = _resolve(window, seqlen_q, seqlen_k, p)
    i = torch.arange(seqlen_q, device=device).view(-1, 1)
    j = torch.arange(seqlen_k, device=device).view(1, -1)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, do, keep, sm_scale):
    """`(dq, lse, delta)` for one sequence, `[H, S, D]` operands, float32.

    A row with no live key has LSE +inf (the forward's contract) and
    contributes nothing, so its dQ is exactly zero. Its scores are replaced by
    a finite constant before the logsumexp so that the autograd graph never
    sees `-inf - -inf`.
    """
    qf = q.float().detach().requires_grad_(True)
    kf, vf, dof = k.float(), v.float(), do.float()
    s = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    empty = ~keep.any(dim=-1)
    s = s.masked_fill(~keep, float("-inf")).masked_fill(empty.view(1, -1, 1), 0.0)
    lse = torch.logsumexp(s, dim=-1, keepdim=True)
    prob = torch.exp(s - lse).masked_fill(~keep, 0.0)
    o = torch.einsum("hqk,hkd->hqd", prob, vf)
    (dq,) = torch.autograd.grad((o * dof).sum(), qf)
    lse = lse.squeeze(-1).detach().masked_fill(empty.view(1, -1), float("inf"))
    delta = (o.detach() * dof).sum(dim=-1)
    return dq, lse, delta, empty


@pytest.fixture(scope="module")
def dq_fn():
    import fmha_bwd_dq_gfx1201_kernel as kernel

    return kernel.build_bwd_dq_module(num_heads=NUM_HEADS,
                                      head_dim=HEAD_DIM,
                                      causal=True,
                                      causal_type=3,
                                      dtype_str="f16")


@pytest.fixture(scope="module")
def abi():
    import fmha_abi_gfx1201

    assert fmha_abi_gfx1201.CAUSAL_SENTINEL == {1: WINDOW_TOPLEFT, 2: WINDOW_BOTRIGHT}
    return fmha_abi_gfx1201


def _bits(abi, p):
    return abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT


# ---------------------------------------------------------------------------
# Host plumbing: no GPU
# ---------------------------------------------------------------------------

# `varlen_bits` is the fourteenth kernel argument: Q K V B DO DQ DB LSE Delta,
# the four seqinfo pointers, then the word.
_VARLEN_BITS_ARG = 13


def _captured_varlen_bits(abi, dq_fn, monkeypatch, *, varlen=None, principal_diagonal=None):
    captured = []
    monkeypatch.setattr(abi, "run_compiled", lambda cache, exe, *args: captured.append(args))
    sq = 64
    if varlen is None:
        batch, num_seqlens = 1, 0
    else:
        batch, num_seqlens = 1, int(varlen["seqinfo_q0"].numel()) - 1
    q = torch.zeros(batch, NUM_HEADS, sq, HEAD_DIM, dtype=DTYPE)
    rows = torch.zeros(batch * NUM_HEADS if varlen is None else NUM_HEADS, sq)
    dq_fn(q,
          q,
          q,
          q,
          q.clone(),
          rows,
          rows.clone(),
          batch,
          sq,
          seqlen_k=sq,
          num_seqlens=num_seqlens,
          window=(8, 0),
          varlen=varlen,
          principal_diagonal=principal_diagonal)
    (args,) = captured
    return int(args[_VARLEN_BITS_ARG])


@_needs_flydsl
@pytest.mark.parametrize("p", (None, 0, 1))
def test_launcher_dense_sets_bit(abi, dq_fn, monkeypatch, p):
    pd = None if p is None else _bits(abi, p)
    got = _captured_varlen_bits(abi, dq_fn, monkeypatch, principal_diagonal=pd)
    assert got == (abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else 0)


@_needs_flydsl
@pytest.mark.parametrize("carried", (0, 1))
@pytest.mark.parametrize("p", (None, 0, 1))
def test_launcher_varlen_bit(abi, dq_fn, monkeypatch, carried, p):
    """`None` keeps the dict's bit 18; an explicit value replaces it. The
    layout bits are never disturbed."""
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)
    varlen = abi.varlen_compact(cu, cu, 32, 32, lse_tokens=64)
    layout = varlen["bits"]
    varlen["bits"] |= _bits(abi, carried)
    pd = None if p is None else _bits(abi, p)
    got = _captured_varlen_bits(abi, dq_fn, monkeypatch, varlen=varlen, principal_diagonal=pd)
    want = carried if p is None else p
    assert got == layout | _bits(abi, want)


@_needs_flydsl
def test_launcher_rejects_bad_choice(abi, dq_fn, monkeypatch):
    with pytest.raises(ValueError, match="principal_diagonal"):
        _captured_varlen_bits(abi, dq_fn, monkeypatch, principal_diagonal=1)


def test_reference_shift_is_not_vacuous():
    """Every non-sentinel window over a `d != 0` shape must move with p, and
    the all-sentinel one must not -- otherwise the GPU sweep proves nothing."""
    for sq, sk in DENSE_SHAPES:
        for w in WINDOWS:
            same = torch.equal(_keep_mask(w, sq, sk, 0, "cpu"), _keep_mask(w, sq, sk, 1, "cpu"))
            moves = sk != sq and not all(x in _SENTINELS for x in w)
            assert same != moves, (sq, sk, w)


# ---------------------------------------------------------------------------
# Numerics: gfx1201
# ---------------------------------------------------------------------------


def _inputs(shape_q, shape_k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(shape_q, device="cuda", dtype=DTYPE, generator=gen)
    k = torch.randn(shape_k, device="cuda", dtype=DTYPE, generator=gen)
    v = torch.randn(shape_k, device="cuda", dtype=DTYPE, generator=gen)
    do = torch.randn(shape_q, device="cuda", dtype=DTYPE, generator=gen)
    return q, k, v, do


def _check(dq, ref, empty, what):
    assert not bool(torch.isnan(dq).any()), f"{what}: NaN in dQ"
    dead = dq[:, empty]
    assert int((dead != 0).sum()) == 0, f"{what}: {int((dead != 0).sum())} nonzero dQ elements in rows with no key"
    torch.testing.assert_close(dq.float(), ref, atol=ATOL, rtol=RTOL, msg=lambda m: f"{what}: {m}")


@_needs_gfx1201
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window", WINDOWS, ids=_window_id)
@pytest.mark.parametrize("shape", DENSE_SHAPES, ids=lambda s: f"q{s[0]}_k{s[1]}")
def test_dq_dense(abi, dq_fn, shape, window, p):
    sq, sk = shape
    batch = 2
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    q, k, v, do = _inputs((batch, NUM_HEADS, sq, HEAD_DIM), (batch, NUM_HEADS, sk, HEAD_DIM), seed=sq * 1000 + sk)
    keep = _keep_mask(window, sq, sk, p, "cuda")
    refs = [_reference(q[b], k[b], v[b], do[b], keep, sm_scale) for b in range(batch)]
    lse = torch.stack([r[1] for r in refs]).reshape(batch * NUM_HEADS, sq).contiguous()
    delta = torch.stack([r[2] for r in refs]).reshape(batch * NUM_HEADS, sq).contiguous()
    dq = torch.full_like(q, float("nan"))

    dq_fn(q,
          k,
          v,
          do,
          dq,
          lse,
          delta,
          batch,
          sq,
          seqlen_k=sk,
          scale=sm_scale,
          window=window,
          principal_diagonal=_bits(abi, p))
    torch.cuda.synchronize()

    for b, (ref, _, _, empty) in enumerate(refs):
        _check(dq[b], ref, empty, f"dense q{sq} k{sk} {_window_id(window)} p={p} batch {b}")


@_needs_gfx1201
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window", WINDOWS, ids=_window_id)
def test_dq_varlen(abi, dq_fn, window, p):
    """Packed (1, H, T, D), so `d` differs per sequence: the shift has to be
    resolved inside the kernel, once per sequence."""
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    lens_q = [s[0] for s in VARLEN_SEQLENS]
    lens_k = [s[1] for s in VARLEN_SEQLENS]
    tq, tk = sum(lens_q), sum(lens_k)
    q, k, v, do = _inputs((1, NUM_HEADS, tq, HEAD_DIM), (1, NUM_HEADS, tk, HEAD_DIM), seed=17)
    cu_q = torch.tensor(list(accumulate(lens_q, initial=0)), dtype=torch.int32, device="cuda")
    cu_k = torch.tensor(list(accumulate(lens_k, initial=0)), dtype=torch.int32, device="cuda")

    lse = torch.empty(NUM_HEADS, tq, device="cuda", dtype=torch.float32)
    delta = torch.empty_like(lse)
    refs = []
    for z, (sq, sk) in enumerate(VARLEN_SEQLENS):
        rq = slice(int(cu_q[z]), int(cu_q[z + 1]))
        rk = slice(int(cu_k[z]), int(cu_k[z + 1]))
        keep = _keep_mask(window, sq, sk, p, "cuda")
        ref, r_lse, r_delta, empty = _reference(q[0, :, rq], k[0, :, rk], v[0, :, rk], do[0, :, rq], keep, sm_scale)
        lse[:, rq] = r_lse
        delta[:, rq] = r_delta
        refs.append((rq, ref, empty, sq, sk))

    varlen = abi.varlen_compact(cu_q, cu_k, max(lens_q), max(lens_k), lse_tokens=tq)
    dq = torch.full_like(q, float("nan"))
    dq_fn(q,
          k,
          v,
          do,
          dq,
          lse,
          delta,
          1,
          max(lens_q),
          seqlen_k=max(lens_k),
          num_seqlens=len(VARLEN_SEQLENS),
          scale=sm_scale,
          window=window,
          varlen=varlen,
          principal_diagonal=_bits(abi, p))
    torch.cuda.synchronize()

    for rq, ref, empty, sq, sk in refs:
        _check(dq[0, :, rq], ref, empty, f"varlen q{sq} k{sk} {_window_id(window)} p={p}")


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
