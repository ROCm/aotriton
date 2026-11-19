# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the gfx942 JIT tests share: the import/skip bootstrap, the host Philox,
NaN-slack operands, the A-2 keep masks, the principal-diagonal cases and the
forward's fp64 reference.

A test module imports `torch`, `np` and `pytestmark` from here first; the
kernel modules import after it, since it puts the FlyDSL kernel tree on sys.path.
Without a gfx942 device or `AOTRITON_FLYDSL_KERNEL_ROOT` the importing module
is skipped before anything imports flydsl.
"""

import os
import sys

import pytest

KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if KERNEL_ROOT and KERNEL_ROOT not in sys.path:
    sys.path.append(KERNEL_ROOT)

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

if (not KERNEL_ROOT
        or not torch.cuda.is_available()
        or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942")):
    pytest.skip("needs a gfx942 device and AOTRITON_FLYDSL_KERNEL_ROOT", allow_module_level=True)
pytestmark = []

import fmha_utils as fmha  # noqa: E402

TL = fmha.WINDOW_TOPLEFT
BR = fmha.WINDOW_BOTRIGHT
# Forward tolerances: O against fp64, and LSE on rows that see a key.
FWD_ATOL = 1.5e-2
FWD_LSE_ATOL = 1e-4

_M32 = 0xFFFFFFFF


def philox_4x32_10(seed, offsets):
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


def with_nan_slack(*shape, dt=torch.bfloat16):
    """A (..., hdim) view of a (..., ceil8(hdim)) tensor whose slack is NaN: the 8xD contract."""
    *lead, d = shape
    full = torch.full((*lead, -(-d // 8) * 8), float("nan"), device="cuda", dtype=dt)
    view = full[..., :d]
    view.copy_(torch.randn(*shape, device="cuda", dtype=dt))
    return view


def resolve(side_value, sentinel_value, numeric_shift):
    """One side of A-2: a sentinel resolves to its own value, a number shifts."""
    if side_value in (TL, BR):
        return sentinel_value
    return side_value + numeric_shift


def keep_mask(window, p, sq, sk, device="cuda"):
    """`[sq, sk]` boolean keep mask from the A-2 predicate (docs/Generalized Sliding
    Window Attention.md); `p` 1 measures numeric windows from the bottom-right diagonal."""
    wl, wr = window
    d = sk - sq
    left = resolve(wl, sq, -p * d)
    right = resolve(wr, d if wr == BR else 0, p * d)
    i = torch.arange(sq, device=device).view(sq, 1)
    j = torch.arange(sk, device=device).view(1, sk)
    return (j >= i - left) & (j <= i + right)


def bottom_right(sq, sk, device="cuda"):
    """`[sq, sk]` keep mask of bottom-right causal."""
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return j <= i + (sk - sq)


def fwd_reference(q, k, v, scale, bias=None, keep=None, p=0.0, mask=None):
    """fp64 `(O, LSE)` over `[..., sq, D]`; a row that sees no key has O = 0, LSE = -inf.
    `mask` is a keep mask, `keep` dropout's."""
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    w = torch.softmax(s, dim=-1).nan_to_num()
    if keep is not None:
        w = w * keep / (1 - p)
    return w @ v.double(), torch.logsumexp(s, dim=-1)


def check_fwd(tag, out, lse, ref, ref_lse, atol=FWD_ATOL, lse_atol=FWD_LSE_ATOL):
    """Live rows against fp64; rows that see no key: O exactly 0, LSE +inf. `ref_lse`
    None skips the rows check, `lse` None the LSE."""
    assert not bool(torch.isnan(out).any()), f"{tag}: {int(torch.isnan(out).sum())} NaN in O"
    err = (out.double() - ref).abs().max().item()
    assert err <= atol, f"{tag}: O off by {err:.3e}"
    if ref_lse is None:
        return
    dead = ~torch.isfinite(ref_lse)
    n = int((out[dead] != 0).sum())
    assert n == 0, f"{tag}: {n} nonzero elements in rows that see no key"
    if lse is None:
        return
    live = ~dead
    if bool(live.any()):
        lse_err = (lse.double() - ref_lse)[live].abs().max().item()
        assert lse_err <= lse_atol, f"{tag}: LSE off by {lse_err:.3e}"
    assert bool(torch.isinf(lse[dead]).all()) and bool((lse[dead] > 0).all()), f"{tag}: dead-row LSE is not +inf"


# Principal-diagonal cases, as tests/test_fwd_principal_diagonal.py's.
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
    0: fmha.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT,
    1: fmha.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT,
}

# (seqlen_q, seqlen_k). Chosen off the tile sizes so ragged edges are hit.
# With p = 1, q_lt_k shifts the band right by 263 (R far past the first
# tiles, L negative), q_gt_k shifts it left by 263 (whole leading Q blocks see
# no key).
DENSE_SHAPES = {
    "q_lt_k": (37, 300),
    "q_eq_k": (130, 130),
    "q_gt_k": (300, 37),
    "decode": (1, 300),
    "chunked_prefill": (64, 320),
}

# One packed batch mixing all three relations, decode and a chunk.
VARLEN_SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))
