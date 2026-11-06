#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`principal_diagonal_choice` (VarlenBits bit 18) in `flyc_attn_fwd`, JIT.

The bit says which diagonal a *numeric* window bound is measured from. With
`d = seqlen_k - seqlen_q` of the sequence being processed and `p` the bit:

    L = sentinel(left)  ? seqlen_q            : left  - p * d
    R = sentinel(right) ? (resolved as today) : right + p * d

and row `i` keeps column `j` iff `0 <= j < seqlen_k` and `i - L <= j <= i + R`
(docs/Generalized Sliding Window Attention.md, A-2). Sentinels are never
shifted. `p = 0` is the behavior before the bit existed.

The bit travels in the VarlenBits word and nowhere else: a dense call passes
`varlen=abi.varlen_dense(principal_diagonal=...)`, a varlen call sets it in
`varlen["bits"]`.

The reference is a float32 masked softmax built straight from that predicate,
per sequence, so a varlen batch whose sequences have different `d` checks that
the shift is resolved per sequence rather than once per call.

A row whose band holds no key has no defined softmax. Only its O is compared
(it must be 0); its LSE is a placeholder and is not read.

Requires a gfx950 device, `AOTRITON_FLYDSL_KERNEL_ROOT`, and a ROCm the FlyDSL
compiler can link against; skips cleanly without any of them. Set
`FLYDSL_EXTRA_SOURCE_DIRS` to this directory after editing a kernel source, or
the JIT disk cache returns the previous binary.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
    FLYDSL_EXTRA_SOURCE_DIRS=$PWD/modules/flash/flyc \\
        python -m pytest modules/flash/flyc/test_fwd_principal_diagonal_gfx950.py
"""

import functools
import math
import os
import sys
from pathlib import Path

import pytest

_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")

_SKIP = (not _KERNEL_ROOT
         or not torch.cuda.is_available()
         or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"))
_SKIP_REASON = "needs a gfx950 device and AOTRITON_FLYDSL_KERNEL_ROOT"
# Skip the module *before* the imports below: a skipif mark does not stop
# module-level imports, and these pull in FlyDSL and the kernel root's
# `kernels.common`, so a host without them would fail collection instead.
if _SKIP:
    pytest.skip(_SKIP_REASON, allow_module_level=True)

import fmha_abi_gfx1201 as abi  # noqa: E402
import fmha_common_gfx1201 as fmha  # noqa: E402

# 64 and 128 take the dual-wave kernel, 512 the wide one. Both resolve the
# window in `ParityKernelContext.init_tile_bounds`, but they walk the tiles
# with different bodies.
HEAD_DIMS = (64, 512)
NUM_HEADS = 2
DTYPE = torch.bfloat16
ATOL = 1.5e-2

TL = fmha.WINDOW_TOPLEFT
BR = fmha.WINDOW_BOTRIGHT

WINDOWS = {
    "sentinel_left_0": (TL, 0),
    "8_0": (8, 0),
    "0_8": (0, 8),
    "4_4": (4, 4),
    "neg1_3": (-1, 3),
    # A pure-sentinel window: must ignore `p` entirely.
    "sentinel_br": (BR, BR),
}

PRINCIPAL_DIAGONAL = {
    0: abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT,
    1: abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT,
}

# (seqlen_q, seqlen_k). Chosen off the tile sizes so ragged edges are hit.
DENSE_SHAPES = {
    "q_lt_k": (37, 300),
    "q_eq_k": (130, 130),
    "q_gt_k": (300, 37),
    "decode": (1, 300),
    "chunked_prefill": (64, 320),
}

# One packed batch mixing all three relations, decode and a chunk.
VARLEN_SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@functools.cache
def _build(head_dim):
    """A causal window build returning LSE, compiled in-process."""
    from flash_attn_func_gfx950 import (
        build_flash_attn_func_gfx950_module_primary as build_primary,
    )
    from fmha_tuning_gfx950 import FmhaInputMetadata, fmha_knobs

    meta = FmhaInputMetadata(num_heads=NUM_HEADS,
                             head_dim=head_dim,
                             causal=True,
                             window=True,
                             dtype_str="bf16")
    return build_primary(meta, fmha_knobs("gfx950", return_lse=True).resolve(meta))


def _resolve(side_value, sentinel_value, numeric_shift):
    """One side of A-2: a sentinel resolves to its own value, a number shifts."""
    if side_value in (TL, BR):
        return sentinel_value
    return side_value + numeric_shift


def _keep_mask(window, p, sq, sk, device):
    """`[sq, sk]` boolean keep mask from the A-2 predicate."""
    wl, wr = window
    d = sk - sq
    left = _resolve(wl, sq, -p * d)
    right = _resolve(wr, d if wr == BR else 0, p * d)
    i = torch.arange(sq, device=device).view(sq, 1)
    j = torch.arange(sk, device=device).view(1, sk)
    return (j >= i - left) & (j <= i + right)


def _reference(q, k, v, keep, sm_scale):
    """fp32 masked softmax over `[H, sq, D]` x `[H, sk, D]`. Returns (O, live rows)."""
    s = torch.einsum("hqd,hkd->hqk", q.float(), k.float()) * sm_scale
    s = s.masked_fill(~keep.unsqueeze(0), float("-inf"))
    live = keep.any(dim=-1)
    w = torch.softmax(s, dim=-1)
    w = torch.where(live.view(1, -1, 1), w, torch.zeros_like(w))
    return torch.einsum("hqk,hkd->hqd", w, v.float()), live


def _check(tag, got, ref, live):
    """Compare one sequence's O: live rows to `ref`, dead rows to exactly 0."""
    assert not bool(torch.isnan(got).any()), f"{tag}: {int(torch.isnan(got).sum())} NaN in O"
    if bool(live.any()):
        err = (got[:, live].float() - ref[:, live]).abs().max().item()
        assert err <= ATOL, f"{tag}: live rows off by {err:.3e}"
    dead = got[:, ~live]
    assert int((dead != 0).sum()) == 0, (
        f"{tag}: {int((dead != 0).sum())} nonzero elements in rows that see no key "
        f"(rows {(~live).nonzero().flatten().tolist()[:8]}...)"
    )


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("shape_name", list(DENSE_SHAPES))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_dense(head_dim, shape_name, window_name, p):
    fn = _build(head_dim)
    sq, sk = DENSE_SHAPES[shape_name]
    window = WINDOWS[window_name]
    batch = 2
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(sq * 1000 + sk)
    q = torch.randn((batch, NUM_HEADS, sq, head_dim), device="cuda", dtype=DTYPE)
    k = torch.randn((batch, NUM_HEADS, sk, head_dim), device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((batch * NUM_HEADS, sq), float("nan"), device="cuda", dtype=torch.float32)

    fn(q,
       k,
       v,
       out,
       batch,
       sq,
       seqlen_k=sk,
       scale=sm_scale,
       lse=lse,
       window=window,
       varlen=abi.varlen_dense(principal_diagonal=PRINCIPAL_DIAGONAL[p]))
    torch.cuda.synchronize()

    keep = _keep_mask(window, p, sq, sk, q.device)
    for b in range(batch):
        ref, live = _reference(q[b], k[b], v[b], keep, sm_scale)
        _check(f"hd={head_dim} {shape_name} {window_name} p={p} b={b}", out[b], ref, live)


@pytest.mark.parametrize("p", sorted(PRINCIPAL_DIAGONAL))
@pytest.mark.parametrize("window_name", list(WINDOWS))
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_varlen_mixed(head_dim, window_name, p):
    """Packed (1, H, T, D), compact on both sides, one `d` per sequence.

    `p` travels in `varlen["bits"]`, next to the compact side bytes.
    """
    fn = _build(head_dim)
    window = WINDOWS[window_name]
    sm_scale = 1.0 / math.sqrt(head_dim)
    lens_q = [s[0] for s in VARLEN_SEQS]
    lens_k = [s[1] for s in VARLEN_SEQS]
    cu_q = [0]
    cu_k = [0]
    for a, b in VARLEN_SEQS:
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + b)
    tq, tk = cu_q[-1], cu_k[-1]

    torch.manual_seed(head_dim + p)
    q = torch.randn((1, NUM_HEADS, tq, head_dim), device="cuda", dtype=DTYPE)
    k = torch.randn((1, NUM_HEADS, tk, head_dim), device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((NUM_HEADS, tq), float("nan"), device="cuda", dtype=torch.float32)
    cu_q_t = torch.tensor(cu_q, device="cuda", dtype=torch.int32)
    cu_k_t = torch.tensor(cu_k, device="cuda", dtype=torch.int32)
    varlen = {
        "bits": abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                abi.VARLEN_COMPACT_SIDE,
                                principal_diagonal=PRINCIPAL_DIAGONAL[p]),
        "seqinfo_q0": cu_q_t,
        "seqinfo_k0": cu_k_t,
        "max_seqlen_q": max(lens_q),
        "max_seqlen_k": max(lens_k),
    }

    fn(q,
       k,
       v,
       out,
       1,
       max(lens_q),
       seqlen_k=max(lens_k),
       scale=sm_scale,
       lse=lse,
       window=window,
       varlen=varlen,
       num_seqlens=len(VARLEN_SEQS))
    torch.cuda.synchronize()

    for z, (sq, sk) in enumerate(VARLEN_SEQS):
        qs = slice(cu_q[z], cu_q[z + 1])
        ks = slice(cu_k[z], cu_k[z + 1])
        keep = _keep_mask(window, p, sq, sk, q.device)
        ref, live = _reference(q[0, :, qs], k[0, :, ks], v[0, :, ks], keep, sm_scale)
        _check(f"hd={head_dim} varlen seq{z}=({sq},{sk}) {window_name} p={p}", out[0, :, qs], ref, live)


def test_varlen_none_is_the_dense_top_left_word():
    """`varlen=None` and `varlen_dense()` are the same word (0): bit-identical O and LSE."""
    fn = _build(HEAD_DIMS[0])
    sq, sk = DENSE_SHAPES["q_lt_k"]
    torch.manual_seed(7)
    q = torch.randn((2, NUM_HEADS, sq, HEAD_DIMS[0]), device="cuda", dtype=DTYPE)
    k = torch.randn((2, NUM_HEADS, sk, HEAD_DIMS[0]), device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    assert abi.varlen_dense()["bits"] == 0
    outs = []
    for varlen in (None, abi.varlen_dense()):
        out = torch.full_like(q, float("nan"))
        lse = torch.full((2 * NUM_HEADS, sq), float("nan"), device="cuda", dtype=torch.float32)
        fn(q,
           k,
           v,
           out,
           2,
           sq,
           seqlen_k=sk,
           lse=lse,
           window=(4, 4),
           varlen=varlen)
        torch.cuda.synchronize()
        outs.append((out, lse))
    assert torch.equal(outs[0][0], outs[1][0])
    assert torch.equal(outs[0][1], outs[1][1])


def test_dense_word_rejects_seqinfo():
    """A dense word reads no sequence info; handing it some is a caller error."""
    fn = _build(HEAD_DIMS[0])
    q = torch.zeros((1, NUM_HEADS, 8, HEAD_DIMS[0]), device="cuda", dtype=DTYPE)
    varlen = abi.varlen_dense(principal_diagonal=abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)
    varlen["seqinfo_q0"] = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="dense VarlenBits word"):
        fn(q,
           q,
           q,
           torch.empty_like(q),
           1,
           8,
           lse=torch.empty((NUM_HEADS, 8), device="cuda", dtype=torch.float32),
           window=(0, 0),
           varlen=varlen)


def test_varlen_dense_rejects_other_values():
    """Only the two `VARLEN_PRINCIPAL_DIAGONAL_*` words; a bare 1 is not bit 18."""
    with pytest.raises(ValueError, match="principal_diagonal"):
        abi.varlen_dense(principal_diagonal=1)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
