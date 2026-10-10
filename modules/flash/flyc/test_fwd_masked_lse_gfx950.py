#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What `flyc_attn_fwd` writes to LSE for a Q row that attends nothing, JIT.

Bottom-right causal with `seqlen_k < seqlen_q` leaves the leading
`seqlen_q - seqlen_k` rows with no live key. Those rows are not a corner case
the caller can avoid -- any cross-attention shape produces them -- and their
LSE is read by the backward, which subtracts it from `qk`. The contract is
therefore `+inf`, so that `exp(qk - lse)` is zero for exactly the rows that
must contribute nothing. `modules/flash/kernel/fwd_kernel.py` writes `+inf` at
its fully-masked early exit and says why; gfx1201's FlyDSL kernel writes it at
its LSE store.

**Two code paths reach those rows, and only one of them is the epilogue.** A Q
block whose every row is masked never enters the kernel body -- `active` is
`causal_end_raw > 0` -- and is served entirely by
`ParityStoreHelper.zero_o_block_if_needed`. A block that straddles the boundary
*is* active and carries its masked rows through the ordinary epilogue. They
fail independently: at the pinned FlyDSL the early exit wrote O and never
touched LSE at all, while the epilogue wrote `m * ln2 + log(0)`, which under
`fastmath<fast>` is poison rather than an infinity. Both come back as whatever
the caller allocated, so both read as "the store faulted" and neither is
distinguishable from the other without splitting the rows the way
`_row_groups` does below.

**Why this is a JIT test and lives here.** The AOT suite covers this from the
other end (`modules/flash/tests/test_backward.py::test_bottom_right_fully_masked_rows`),
but reaching it costs a full `ninja install` plus a by-hand deletion of the
flyc hsaco/aks2/zip outputs, because no ninja edge depends on these sources.
Building one kernel in-process instead turns that into seconds, and it is the
only way to sweep `head_dim` -- the AOT path compiles whatever the tuning
database selected, so it cannot ask for the wide kernel on purpose.

Requires a gfx950 device, `AOTRITON_FLYDSL_KERNEL_ROOT`, and a ROCm the
FlyDSL compiler can link against; skips cleanly without any of them.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_fwd_masked_lse_gfx950.py
"""

import math
import os
import sys
from pathlib import Path

import pytest

# The vendored kernels are this file's own directory; importing them by bare
# name is what every other consumer does (`devtools/`, the AOT descriptions),
# and pytest's rootless import mode has already put the directory on sys.path.
# The FlyDSL checkout has to be added, for the `kernels.attention` that
# `gfx950_standalone` re-exports as `dualwave`.
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not _KERNEL_ROOT
    or not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="needs a gfx950 device and AOTRITON_FLYDSL_KERNEL_ROOT",
)

# head_dim 512 takes the wide kernel (`fmha_wide_gfx950.py`), a different
# algorithm with its own store helper; 64 and 128 take the dual-wave one. The
# early exit is shared -- `zero_o_block_if_needed` is called outside the
# WIDE/else split -- so covering both is what says the shared call is reached
# from both, which is not visible from either alone.
HEAD_DIMS = (64, 128, 512)

BATCH = 2
NUM_HEADS = 4
SEQLEN_K = 64
DTYPE = torch.bfloat16
# bf16 inputs and an fp32 reference disagree in the last bits of the dot
# product; this bound is on the live rows only, where the value is O(1).
LIVE_ATOL = 8e-3


def _build(head_dim):
    """A causal, LSE-returning forward for `head_dim`, compiled in-process."""
    from flash_attn_func_gfx950 import (
        build_flash_attn_func_gfx950_module_primary as build_primary,
    )
    from fmha_tuning_gfx950 import FmhaInputMetadata, fmha_knobs

    meta = FmhaInputMetadata(
        num_heads=NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="bf16"
    )
    # `return_lse=True` pinned, exactly as `modules/flash/aot/flyc_attn_fwd.py`
    # pins it: the policy's default is False, which deletes the store outright
    # and would make every assertion below read the allocation.
    return build_primary(meta, fmha_knobs("gfx950", return_lse=True).resolve(meta))


def _row_groups(block_m):
    """`(seqlen_q, whole_block, straddling)` for a shape that arms both paths.

    The masked prefix is `1.5 * BLOCK_M`, which is the smallest multiple of
    half a block that produces one of each: block 0 is masked end to end and
    takes the early exit, block 1 straddles the boundary and runs the body
    with its first half masked. Derived from the built kernel's own BLOCK_M
    rather than hardcoded -- the tile width varies with head_dim and with
    whatever the tuning policy decides next, and a fixed prefix would quietly
    stop covering one of the two paths.
    """
    n_masked = block_m + block_m // 2
    return n_masked + SEQLEN_K, slice(0, block_m), slice(block_m, n_masked)


def _reference_lse(q, k, sm_scale):
    """fp32 bottom-right causal LSE, `[B, H, Sq]`, natural log, scale folded."""
    qf, kf = q.float(), k.float()
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * sm_scale
    sq, sk = q.shape[2], k.shape[2]
    qi = torch.arange(sq, device=q.device).view(sq, 1)
    ki = torch.arange(sk, device=q.device).view(1, sk)
    scores = scores.masked_fill((ki > (qi + (sk - sq))).view(1, 1, sq, sk), float("-inf"))
    return torch.logsumexp(scores, dim=-1)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_fully_masked_rows_get_positive_inf_lse(head_dim):
    fn = _build(head_dim)
    seqlen_q, whole, straddling = _row_groups(fn.traits.BLOCK_M)
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(head_dim)
    shape_q = (BATCH, NUM_HEADS, seqlen_q, head_dim)
    shape_k = (BATCH, NUM_HEADS, SEQLEN_K, head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=DTYPE)
    k = torch.randn(shape_k, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    # NaN rather than zeros: an unwritten element must be distinguishable from
    # a correctly written one, and 0.0 is a legitimate LSE and the *required*
    # Out. This is what the AOT harness's `fillnan` does, for the same reason.
    out = torch.full(shape_q, float("nan"), device="cuda", dtype=DTYPE)
    lse = torch.full((BATCH * NUM_HEADS, seqlen_q), float("nan"),
                     device="cuda", dtype=torch.float32)

    fn(q, k, v, out, BATCH, seqlen_q, seqlen_k=SEQLEN_K, scale=sm_scale, lse=lse)
    torch.cuda.synchronize()

    lse = lse.view(BATCH, NUM_HEADS, seqlen_q)
    n_masked = straddling.stop
    for name, rows in (("wholly-masked block", whole), ("straddling block", straddling)):
        got = lse[:, :, rows]
        n_inf = int((got == float("inf")).sum())
        assert n_inf == got.numel(), (
            f"head_dim={head_dim} BLOCK_M={fn.traits.BLOCK_M}: {name} rows "
            f"{rows.start}:{rows.stop} -- {n_inf}/{got.numel()} are +inf; "
            f"{int(torch.isnan(got).sum())} NaN (never written, or poison), "
            f"{int((got == float('-inf')).sum())} -inf (written with the wrong sign)"
        )

    # The masked rows above are satisfied by a kernel that writes +inf to
    # everything, so pin the rest: Out is zero where nothing was attended, and
    # the live rows carry the real logsumexp.
    masked_out = out[:, :, :n_masked]
    assert int((masked_out != 0).sum()) == 0, (
        f"head_dim={head_dim}: {int((masked_out != 0).sum())} nonzero elements in "
        f"fully-masked Out rows (NaN count {int(torch.isnan(masked_out).sum())})"
    )
    live = lse[:, :, n_masked:]
    ref = _reference_lse(q, k, sm_scale)[:, :, n_masked:]
    assert bool(torch.isfinite(live).all()), "live rows must not be infinite"
    diff = (live - ref).abs().max().item()
    assert diff <= LIVE_ATOL, f"head_dim={head_dim}: live LSE off by {diff:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
