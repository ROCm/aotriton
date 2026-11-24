#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Infinities through forward, dQ and dK/dV, JIT.

Every masked score is an infinity in arithmetic: a bias entry of -inf, the
-inf a window or KV-tail mask writes, the +inf LSE the forward hands the
backward for a row with no key. Fast-math flags license the compiler to assume
none of them exist (`ninf`), or that no operation they feed yields NaN
(`nnan`). Such a licence does not fail loudly: a mask goes missing, or a
fully masked row becomes NaN, only on the inputs that reach it. This suite
feeds those inputs on purpose:

- bias with scattered -inf, whole -inf rows and columns, a row with a single
  live key in the KV tail, and a whole -inf (batch, head) plane;
- windows that leave leading or trailing rows and columns with nothing;
- packed varlen batches, so every sequence has its own KV tail;
- head dims that do not fill their tile;
- each of them with and without dropout.

Rows with no key must give O = 0, LSE = +inf and dQ = 0; keys no row sees
must give dK = dV = 0; nothing may be NaN.

The reference is an fp32 softmax and its backward written out by hand, so a
row with no key has P = 0 and no autograd `-inf - -inf`. Dropout has no host
model of the Philox stream, so its keep mask is read back from the forward
under test: with V a block of the identity, O[i, j] is nonzero iff key `c + j`
is kept for row i. A forward that dropped the wrong elements would pass its own
O check that way, but not its LSE check, nor the backward's, which use the mask
as the forward's contract.

Builds for the first GPU's arch through the `tuning_*` dispatchers; skips on an
arch they have no row for, or without `AOTRITON_FLYDSL_KERNEL_ROOT`.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/tests/test_infinite_masks.py
"""

import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import jit_arch
import tuning_dkdv
import tuning_dq
import tuning_fwd
from fmha_meta import CAUSAL_NONE, CAUSAL_WINDOW, FmhaInputMetadata

torch = pytest.importorskip("torch")
ARCH = jit_arch.require(tuning_fwd)
for _dispatcher in (tuning_dq, tuning_dkdv):
    if ARCH not in _dispatcher.arches():
        pytest.skip(f"{_dispatcher.__name__} has no {ARCH} row", allow_module_level=True)

import fmha_utils as abi  # noqa: E402

PASSES = {"fwd": tuning_fwd, "dq": tuning_dq, "dkdv": tuning_dkdv}
# The dQ kernel's dB argument; the name differs by arch.
DB_ARG = {"gfx950": "db"}.get(ARCH, "dbias")

# `(head_dim, dtype)`. 120 does not fill its tile on any arch; f16 there
# because bf16 packs P with a rounding that turns NaN into zero. 256 and 512
# split the head dim across waves on gfx1201, and 512 takes gfx950's wide
# forward.
BUILDS = {
    "d64_bf16": (64, "bf16"),
    "d128_bf16": (128, "bf16"),
    "d120_f16": (120, "f16"),
    "d256_bf16": (256, "bf16"),
    "d512_bf16": (512, "bf16"),
}
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
# Max abs error over the reference's max magnitude, floored at 1.
TOL = {"bf16": 2e-2, "f16": 5e-3}
LSE_ATOL = 1e-2
NUM_HEADS = 2
DROPOUT_P = 0.3
PHILOX_SEED = 0x2B7E1516
PHILOX_OFFSET = 4096

TL = abi.WINDOW_TOPLEFT
BR = abi.WINDOW_BOTRIGHT
# Numeric bounds are measured from the top-left diagonal: row i keeps key j iff
# i - left <= j <= i + right.
WINDOWS = {
    "bottom_right": (BR, BR),  # leading rows dead when seqlen_q > seqlen_k
    "lag": (3, -1),  # row 0 dead
    "lead": (-5, 10),  # trailing rows dead
    "diagonal": (0, 0),  # trailing rows or keys dead, one key per row
}
WINDOW_SHAPES = [(96, 40), (40, 96), (77, 77)]
BIAS_SHAPES = [(64, 64), (100, 77), (33, 200), (128, 257)]
# `(seqlen_q, seqlen_k)` per sequence: every KV length ragged.
VARLEN_SEQS = [(40, 77), (64, 33), (1, 130), (100, 57)]


def _shape_id(s):
    return f"q{s[0]}_k{s[1]}"


@functools.cache
def _build(pass_name, build, causal_type, bias, dropout, max_seqlen_k=None):
    """`max_seqlen_k`: the call's, for the forward's per-call rules."""
    head_dim, dtype_str = BUILDS[build]
    meta = FmhaInputMetadata(num_heads=NUM_HEADS,
                             block_dmodel=None,
                             head_dim=head_dim,
                             dtype_str=dtype_str,
                             causal_type=causal_type,
                             bias=bias,
                             dropout=dropout)
    runtime = {} if max_seqlen_k is None else {"max_seqlen_k": max_seqlen_k}
    return jit_arch.build(PASSES[pass_name], ARCH, meta, **runtime)[0]


@dataclass
class Problem:
    """One call's tensors and how to split them into sequences.

    Dense tensors are `(B, H, S, D)` and each batch is a sequence. Packed ones
    are `(1, H, T, D)`, cut into sequences by the cumulative lengths `cu_q` and
    `cu_k`, which are None for a dense call.
    """

    q: object
    k: object
    v: object
    do: object
    bias: object
    window: tuple | None
    cu_q: list | None
    cu_k: list | None
    dropout: bool

    @property
    def dense(self):
        return self.cu_q is None

    @property
    def num_seqs(self):
        return self.q.shape[0] if self.dense else len(self.cu_q) - 1

    def seq(self, t, z, k_side=False):
        """Sequence `z` of a `(B|1, H, S|T, ...)` tensor, as `(H, S, ...)`."""
        if self.dense:
            return t[z]
        cu = self.cu_k if k_side else self.cu_q
        return t[0, :, cu[z]:cu[z + 1]]

    def rows(self, t, z):
        """Sequence `z` of an LSE-shaped tensor, as `(H, S)`."""
        if self.dense:
            return t.view(self.q.shape[0], NUM_HEADS, -1)[z]
        return t[:, self.cu_q[z]:self.cu_q[z + 1]]

    def row_tensor(self, fill):
        n = self.q.shape[0] * NUM_HEADS if self.dense else NUM_HEADS
        return torch.full((n, self.q.shape[2]), fill, device="cuda", dtype=torch.float32)

    def call_args(self):
        """`(batch, seqlen_q, kwargs)` shared by the three launchers."""
        kw = {}
        if self.window is not None:
            kw["window"] = self.window
        if self.dense:
            return self.q.shape[0], self.q.shape[2], dict(kw, seqlen_k=self.k.shape[2])
        lens_q = [b - a for a, b in zip(self.cu_q, self.cu_q[1:])]
        lens_k = [b - a for a, b in zip(self.cu_k, self.cu_k[1:])]
        varlen = abi.varlen_compact(torch.tensor(self.cu_q, device="cuda", dtype=torch.int32),
                                    torch.tensor(self.cu_k, device="cuda", dtype=torch.int32),
                                    max(lens_q),
                                    max(lens_k),
                                    lse_tokens=self.cu_q[-1])
        return 1, max(lens_q), dict(kw, seqlen_k=max(lens_k), varlen=varlen, num_seqlens=len(lens_q))


def _rand(shape, dtype_str, gen):
    return torch.randn(shape, device="cuda", dtype=DTYPES[dtype_str], generator=gen)


def _problem(dtype_str, head_dim, seed, dropout, shape=None, seqs=None, bias=False, window=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if seqs is None:
        sq, sk = shape
        shape_q, shape_k = (2, NUM_HEADS, sq, head_dim), (2, NUM_HEADS, sk, head_dim)
        cu_q = cu_k = None
    else:
        cu_q, cu_k = [0], [0]
        for a, b in seqs:
            cu_q.append(cu_q[-1] + a)
            cu_k.append(cu_k[-1] + b)
        shape_q, shape_k = (1, NUM_HEADS, cu_q[-1], head_dim), (1, NUM_HEADS, cu_k[-1], head_dim)
    q, k, v, do = (_rand(s, dtype_str, gen) for s in (shape_q, shape_k, shape_k, shape_q))
    b = _bias(shape_q[0], shape_q[2], shape_k[2], dtype_str, gen) if bias else None
    return Problem(q, k, v, do, b, window, cu_q, cu_k, dropout)


def _bias(batch, sq, sk, dtype_str, gen):
    """`(B, H, Sq, Sk)` finite noise, then -inf in every pattern a mask takes.

    (0, 0): rows 0 and sq-1 all -inf; row 1 -inf but for its last key, which
    sits in the KV tail tile. (0, 1): keys 0, sk//3 and sk-1 -inf for every row.
    (1, 0): a quarter of the entries -inf at random. (1, 1): all -inf.
    """
    ninf = float("-inf")
    b = _rand((batch, NUM_HEADS, sq, sk), dtype_str, gen) * 0.5
    b[0, 0, 0] = ninf
    b[0, 0, sq - 1] = ninf
    if sq > 2:
        b[0, 0, 1, :sk - 1] = ninf
    b[0, 1, :, [0, sk // 3, sk - 1]] = ninf
    if batch > 1:
        scatter = torch.rand((sq, sk), device="cuda", generator=gen) < 0.25
        b[1, 0][scatter] = ninf
        b[1, 1] = ninf
    return b


def _band(window, sq, sk):
    """`(sq, sk)` bool: the keys each row keeps, by the spec's predicate."""
    i = torch.arange(sq, device="cuda")[:, None]
    j = torch.arange(sk, device="cuda")[None, :]
    if window is None:
        return torch.ones((sq, sk), dtype=torch.bool, device="cuda")
    wl, wr = window
    left = sq if wl in (TL, BR) else wl
    if wr in (TL, BR):
        right = 0 if wr == TL else sk - sq
    else:
        right = wr
    return (j >= i - left) & (j <= i + right)


@dataclass
class Reference:
    """fp32 forward and backward of one sequence, `(H, ...)` throughout."""

    o: object
    lse: object
    delta: object
    dq: object
    dk: object
    dv: object
    ds: object
    live: object  # (H, Sq, Sk): the finite scores
    dead_rows: object  # (H, Sq)
    dead_keys: object  # (H, Sk)


def _reference(pb, z, sm_scale, keep):
    """`keep` is `(H, Sq, Sk)` bool, the dropout mask, or None."""
    q, do = pb.seq(pb.q, z).float(), pb.seq(pb.do, z).float()
    k, v = pb.seq(pb.k, z, k_side=True).float(), pb.seq(pb.v, z, k_side=True).float()
    sq, sk = q.shape[1], k.shape[1]
    s = torch.einsum("hqd,hkd->hqk", q, k) * sm_scale
    if pb.bias is not None:
        s = s + pb.bias[z].float()
    s = s.masked_fill(~_band(pb.window, sq, sk), float("-inf"))
    live = torch.isfinite(s)
    dead_rows = ~live.any(-1)
    m = s.amax(-1, keepdim=True).masked_fill(dead_rows[..., None], 0.0)
    e = torch.exp(s - m)
    p = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
    lse = (m.squeeze(-1) + torch.log(e.sum(-1))).masked_fill(dead_rows, float("inf"))
    pd = p if keep is None else p * keep / (1.0 - DROPOUT_P)
    o = torch.einsum("hqk,hkd->hqd", pd, v)
    dp = torch.einsum("hqd,hkd->hqk", do, v)
    if keep is not None:
        dp = dp * keep / (1.0 - DROPOUT_P)
    delta = (do * o).sum(-1)
    ds = p * (dp - delta[..., None])
    dq = torch.einsum("hqk,hkd->hqd", ds, k) * sm_scale
    dk = torch.einsum("hqk,hqd->hkd", ds, q) * sm_scale
    dv = torch.einsum("hqk,hqd->hkd", pd, do)
    return Reference(o, lse, delta, dq, dk, dv, ds, live, dead_rows, ~live.any(1))


def _forward(fn, pb, v, sm_scale):
    """`(out, lse)` of one forward call with `v` in place of `pb.v`."""
    batch, seqlen_q, kw = pb.call_args()
    if pb.bias is not None:
        kw["bias"] = pb.bias
    out = torch.full_like(pb.q, float("nan"))
    lse = pb.row_tensor(float("nan"))
    fn(pb.q,
       pb.k,
       v,
       out,
       batch,
       seqlen_q,
       scale=sm_scale,
       lse=lse,
       **kw,
       **_dropout_args(pb.dropout))
    torch.cuda.synchronize()
    return out, lse


def _dropout_args(dropout):
    if not dropout:
        return {}
    return dict(dropout_p=DROPOUT_P, philox_seed=PHILOX_SEED, philox_offset2=PHILOX_OFFSET)


def _keep_masks(fwd, pb, sm_scale):
    """Per sequence, `(H, Sq, Sk)` bool: the forward's dropout keep mask.

    V is a block of the identity, so O column j is P * keep at key c + j.
    """
    width = pb.v.shape[-1]
    nk = [pb.seq(pb.k, z, k_side=True).shape[1] for z in range(pb.num_seqs)]
    masks = [torch.zeros((NUM_HEADS, pb.seq(pb.q, z).shape[1], n), dtype=torch.bool, device="cuda")
             for z, n in enumerate(nk)]
    for c in range(0, max(nk), width):
        v = torch.zeros_like(pb.v)
        for z, n in enumerate(nk):
            for j in range(min(width, max(n - c, 0))):
                pb.seq(v, z, k_side=True)[:, c + j, j] = 1
        out, _ = _forward(fwd, pb, v, sm_scale)
        for z, n in enumerate(nk):
            span = min(width, max(n - c, 0))
            masks[z][:, :, c:c + span] = pb.seq(out, z)[:, :, :span] != 0
    return masks


def _close(got, want, dtype_str, what):
    assert not bool(torch.isnan(got).any()), f"{what}: {int(torch.isnan(got).sum())} NaN"
    assert bool(torch.isfinite(got).all()), f"{what}: non-finite"
    scale = max(1.0, want.abs().max().item())
    err = (got.float() - want).abs().max().item()
    assert err <= TOL[dtype_str] * scale, f"{what}: max abs error {err:.3e} against scale {scale:.3e}"


def _zero(got, where, what):
    """`got` is exactly 0 wherever `where` (broadcast over trailing axes) holds."""
    bad = int((got.masked_select(where) != 0).sum())
    assert bad == 0, f"{what}: {bad} nonzero elements where nothing is attended"


def _run(pass_name, build, pb, causal_type, label):
    """Run `pass_name` on `pb` with and without the build's dropout; check it."""
    head_dim, dtype_str = BUILDS[build]
    sm_scale = 1.0 / math.sqrt(head_dim)
    bias = pb.bias is not None
    fwd = _build("fwd", build, causal_type, bias, pb.dropout, max_seqlen_k=pb.call_args()[2]["seqlen_k"])
    keeps = _keep_masks(fwd, pb, sm_scale) if pb.dropout else [None] * pb.num_seqs
    refs = [_reference(pb, z, sm_scale, keeps[z]) for z in range(pb.num_seqs)]
    if pb.dropout:
        kept = torch.cat([keep.masked_select(ref.live) for ref, keep in zip(refs, keeps)]).float()
        frac = kept.mean().item()
        assert abs(frac - (1 - DROPOUT_P)) < 0.05, f"{label}: dropout kept {frac:.3f} of the live elements"

    if pass_name == "fwd":
        out, lse = _forward(fwd, pb, pb.v, sm_scale)
        for z, ref in enumerate(refs):
            what = f"{label} seq {z}"
            o, l = pb.seq(out, z), pb.rows(lse, z)
            _zero(o, ref.dead_rows[..., None], f"{what} O")
            n_inf = int((l.masked_select(ref.dead_rows) == float("inf")).sum())
            assert n_inf == int(ref.dead_rows.sum()), (
                f"{what}: {n_inf}/{int(ref.dead_rows.sum())} dead rows have LSE +inf "
                f"({int(torch.isnan(l).sum())} NaN)"
            )
            _close(o, ref.o, dtype_str, f"{what} O")
            live = ~ref.dead_rows
            assert bool(torch.isfinite(l.masked_select(live)).all()), f"{what}: live LSE not finite"
            err = (l - ref.lse).masked_select(live).abs().max().item() if bool(live.any()) else 0.0
            assert err <= LSE_ATOL, f"{what}: live LSE off by {err:.3e}"
        return

    lse = pb.row_tensor(0.0)
    delta = pb.row_tensor(0.0)
    for z, ref in enumerate(refs):
        pb.rows(lse, z).copy_(ref.lse)
        pb.rows(delta, z).copy_(ref.delta)
    batch, seqlen_q, kw = pb.call_args()
    kw.update(_dropout_args(pb.dropout))
    fn = _build(pass_name, build, causal_type, bias, pb.dropout)
    if bias:
        kw["bias"] = pb.bias
    if pass_name == "dq":
        dq = torch.full_like(pb.q, float("nan"))
        if bias:
            kw[DB_ARG] = torch.full_like(pb.bias, float("nan"))
        fn(pb.q,
           pb.k,
           pb.v,
           pb.do,
           dq,
           lse,
           delta,
           batch,
           seqlen_q,
           scale=sm_scale,
           **kw)
        torch.cuda.synchronize()
        for z, ref in enumerate(refs):
            what = f"{label} seq {z}"
            got = pb.seq(dq, z)
            _zero(got, ref.dead_rows[..., None], f"{what} dQ")
            _close(got, ref.dq, dtype_str, f"{what} dQ")
            if bias:
                db = kw[DB_ARG][z]
                _zero(db, ~torch.isfinite(pb.bias[z]), f"{what} dB at -inf bias")
                _close(db, ref.ds, dtype_str, f"{what} dB")
        return

    dk = torch.full_like(pb.k, float("nan"))
    dv = torch.full_like(pb.v, float("nan"))
    fn(pb.q,
       pb.k,
       pb.v,
       pb.do,
       dk,
       dv,
       lse,
       delta,
       batch,
       seqlen_q,
       scale=sm_scale,
       **kw)
    torch.cuda.synchronize()
    for z, ref in enumerate(refs):
        what = f"{label} seq {z}"
        for name, got, want in (("dK", pb.seq(dk, z, True), ref.dk), ("dV", pb.seq(dv, z, True), ref.dv)):
            _zero(got, ref.dead_keys[..., None], f"{what} {name}")
            _close(got, want, dtype_str, f"{what} {name}")


@pytest.mark.parametrize("pass_name", list(PASSES))
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("dropout", (False, True), ids=("nodrop", "drop"))
@pytest.mark.parametrize("shape", BIAS_SHAPES, ids=_shape_id)
def test_bias(pass_name, build, dropout, shape):
    head_dim, dtype_str = BUILDS[build]
    pb = _problem(dtype_str, head_dim, sum(shape) * 7 + head_dim, dropout, shape=shape, bias=True)
    _run(pass_name, build, pb, CAUSAL_NONE, f"{pass_name} {build} bias {_shape_id(shape)} dropout={dropout}")


@pytest.mark.parametrize("pass_name", list(PASSES))
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("dropout", (False, True), ids=("nodrop", "drop"))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape", WINDOW_SHAPES, ids=_shape_id)
def test_window(pass_name, build, dropout, window_name, shape):
    head_dim, dtype_str = BUILDS[build]
    pb = _problem(dtype_str, head_dim, sum(shape) * 11 + head_dim, dropout, shape=shape, window=WINDOWS[window_name])
    _run(pass_name,
         build,
         pb,
         CAUSAL_WINDOW,
         f"{pass_name} {build} {window_name} {_shape_id(shape)} dropout={dropout}")


@pytest.mark.parametrize("pass_name", list(PASSES))
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("dropout", (False, True), ids=("nodrop", "drop"))
@pytest.mark.parametrize("window_name", ("none", "bottom_right", "lag"))
def test_varlen(pass_name, build, dropout, window_name):
    head_dim, dtype_str = BUILDS[build]
    window = WINDOWS.get(window_name)
    pb = _problem(dtype_str, head_dim, head_dim * 13 + len(window_name), dropout, seqs=VARLEN_SEQS, window=window)
    causal_type = CAUSAL_NONE if window is None else CAUSAL_WINDOW
    _run(pass_name, build, pb, causal_type, f"{pass_name} {build} varlen {window_name} dropout={dropout}")


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
