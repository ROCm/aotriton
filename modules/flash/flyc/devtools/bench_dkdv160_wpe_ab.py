#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Interleaved A/B of head_dim 160 dK/dV at `waves_per_eu` 1 against 2.

Why this exists. `_GEOMETRY[160]` in `fmha_tuning_bwd_dkdv_gfx950.py` asks for
`waves_per_eu=2`, and the backend cannot grant it: the workgroup takes 87040 B
of LDS, so two of them need 174080 B against a 163840 B cap and occupancy is
pinned at 1 before the register allocator gets a say. Every head_dim 160 build
therefore emits

    failed to meet occupancy target given by 'amdgpu-waves-per-eu'
    in 'flyc_bwd_dkdv': desired occupancy was 2, final occupancy is 1

The hint is not always inert when unmet -- it doubles as a register budget for
the scheduler, and the tuning table records a case where the hint alone moved a
rung from 403 to 721 TFLOP/s. So "can we just set it to 1 and silence the
warning" was a measurement, not a cleanup.

**The answer, for this rung, is that there is nothing to measure**, and this
file is kept for how that was established rather than for its numbers. Building
both ways and comparing the artifacts gives 24 byte-identical hsacos: at
occupancy 1 the register budget is already the whole file, so asking for 2
constrains nothing. Whereupon note the trap below.

Shape and accounting are FlyDSL's own backward benchmark's, so the numbers land
on the same scale as the tuning tables': B=2 H=8 S=4096 bf16 non-causal, nominal
FLOPs, dK/dV counted as four GEMMs (`8*B*H*Sq*Sk*d`). That benchmark lives in
the FlyDSL tree and is not vendored here; the constants below are the whole of
what is borrowed from it.

Two configurations reach head_dim 160 and BOTH ask for 2, so both are measured:

  * `_GEOMETRY[160]`                      -> 32 MFMA rows, bias=False builds
  * `_FEATURE_OVERRIDES[(160,False,True)]` -> 16 MFMA rows, bias=True builds

**Interleaved, not sequential.** `perf_ab.py` records the reason: running one
variant to completion and then the other puts minutes between the two
measurements of a point, and the board drifts over minutes -- a ~5% noise floor
with outliers to +19% on kernel selections that were provably identical. Here
both variants live in one process, so the gap is one repetition rather than one
sweep, and the alternation is what cancels the drift. Nothing is reloaded
between rounds.

**Interleaving is not enough on its own, which is the trap this file exists to
record.** Run in the default order it reports the dense arm 5.5% faster at
wpe=1, reproducibly, over two independent runs -- against binaries already
known to be byte-identical. The 5.5% is position in the round: every arm here
runs second in its pair, straight after one that streams a 512 MB bias tensor
through L2. `WPE_ORDER=12` swaps the pair order and the same 5.5% moves to the
other arm. A drift-cancelling design still leaves a *fixed* per-slot bias, so
alternate the order too, or compare only slots that saw the same predecessor.

Needs gfx950 hardware, torch, and a flydsl matching the wheel the kernels were
built with. It needs no AOTriton build: it imports the vendored kernels
directly and JIT-builds them. Set, in the environment:

    ROCM_PATH                     a directory with llvm/bin/ld.lld under it
    AOTRITON_FLYDSL_KERNEL_ROOT   a FlyDSL source checkout

`python/flyc_bootstrap.py`'s `resolve_rocm_path()` will find and print the
first if you do not know it.

Usage:

    ulimit -c 0
    HIP_VISIBLE_DEVICES=5 \\
    PYTHONPATH=<build>/venv/lib/python3.13/site-packages \\
    python \\
        modules/flash/flyc/devtools/bench_dkdv160_wpe_ab.py [head_dim ...]

Set `WPE_ORDER=12` to run wpe=1 before wpe=2 in each round; any effect that
does not survive both orders is the slot, not the knob.

PYTHONPATH is how a locally built flydsl is used together with an interpreter
that has torch: the build virtual environment has the compiler and deliberately
no torch, and putting its site-packages first shadows only `flydsl`.
"""

import os
import statistics
import sys
import time
from pathlib import Path

# AOTriton's VENDORED kernels, not FlyDSL's parity copies -- these are what
# ships, and the two trees have diverged (the varlen build axis is gone here).
# The kernels are this directory's parent, and that is the whole of what this
# tool knows about the repository layout.
# It must come first, so that `fmha_traits_gfx950` and friends resolve to
# the vendored siblings rather than to FlyDSL's own `parity/` copies.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ...and the FlyDSL checkout, for the `kernels.attention.flash_attn_utils` that
# `gfx950_standalone` re-exports as `dualwave`. Required, with no default: the
# only honest default would be one person's build directory.
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if not _KERNEL_ROOT:
    sys.exit(
        "AOTRITON_FLYDSL_KERNEL_ROOT is unset. Point it at a FlyDSL source "
        "checkout (a build directory's own `flydsl` clone will do) so that "
        "`kernels.attention.flash_attn_utils` resolves."
    )
sys.path.append(_KERNEL_ROOT)

# ROCM_PATH, required from the environment. Without a correct one flydsl fails
# at link time with a bare `error: lld invocation failed` that says nothing
# about the toolchain, so it is checked up front rather than discovered.
_ROCM_PATH = os.environ.get("ROCM_PATH")
if not _ROCM_PATH or not (Path(_ROCM_PATH) / "llvm" / "bin" / "ld.lld").is_file():
    sys.exit(
        "ROCM_PATH must name a directory containing llvm/bin/ld.lld at exactly "
        "that relative path; it is "
        + (f"set to {_ROCM_PATH!r}, which does not." if _ROCM_PATH else "unset.")
        + " AOTriton's own resolver, python/flyc_bootstrap.py's "
        "resolve_rocm_path(), knows where to look and will print the answer."
    )
print(f"ROCM_PATH={_ROCM_PATH}")

import torch  # noqa: E402

from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build_dkdv  # noqa: E402

DT = torch.bfloat16
B, H, S = 2, 8, 4096
HDIMS = [160]

# One timed sample is `REP` launches back to back; `ROUNDS` samples per variant,
# taken round-robin. 3 rounds is what FlyDSL's own non-interleaved best-of
# uses; more here because the whole point is to resolve a difference smaller
# than the drift.
REP = 20
ROUNDS = 15
WARMUP = 10


# The feature combinations that reach head_dim 160. `causal` and `bias` are
# mutually exclusive by construction (`make_traits` enforces it), and `dropout`
# is included because it is the arm that spills: the worst head_dim 160 build in
# the shipped set is CAUSAL=0 DROPOUT=1 BIAS=0 at 447 VGPRs. Only the bias arm
# takes `_FEATURE_OVERRIDES`; the other three all take `_GEOMETRY[160]`, so a
# change there has to be right for all of them.
FEATURES = (
    {"causal": False, "bias": False, "dropout": False},
    {"causal": True, "bias": False, "dropout": False},
    {"causal": False, "bias": False, "dropout": True},
    {"causal": False, "bias": True, "dropout": False},
)


def make_inputs(d, bias):
    q, k, v, do = (torch.randn(B, H, S, d, device="cuda", dtype=DT) for _ in range(4))
    dk, dv = (torch.empty_like(q) for _ in range(2))
    lse = torch.zeros(B * H, S, device="cuda", dtype=torch.float32)
    delta = torch.zeros(B * H, S, device="cuda", dtype=torch.float32)
    bias_t = torch.randn(B, H, S, S, device="cuda", dtype=DT) if bias else None
    return q, k, v, do, dk, dv, lse, delta, bias_t


def make_call(fn, t, d, dropout):
    q, k, v, do, dk, dv, lse, delta, bias_t = t
    scale = d**-0.5
    extra = (
        {"dropout_p": 0.5, "philox_seed": 0x1BF52, "philox_offset1": 0x1D4B42}
        if dropout
        else {}
    )

    def call():
        fn(
            q, k, v, do, dk, dv, lse, delta, B, S,
            seqlen_k=S, scale=scale, bias=bias_t, **extra,
        )

    return call


def sample(call, rep=REP):
    """One timed sample: mean seconds per launch over `rep` back-to-back launches."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


def run(d):
    base = B * H * S * S * d * 1e-12

    variants = []
    for feat in FEATURES:
        tensors = make_inputs(d, feat["bias"])
        name = "".join(k[0] for k in ("causal", "bias", "dropout") if feat[k]) or "dense"
        # Order within a round is a confound, not a detail: a variant that
        # always runs straight after the bias arm inherits an L2 flushed by a
        # 512 MB bias read, and one that always runs second in its pair
        # inherits a warm one. WPE_ORDER flips it so the asymmetry can be told
        # apart from the knob.
        for wpe in ((1, 2) if os.environ.get("WPE_ORDER") == "12" else (2, 1)):
            fn = build_dkdv(
                num_heads=H, num_kv_heads=H, head_dim=d, waves_per_eu=wpe, **feat
            )
            kn = fn.knobs
            assert kn.waves_per_eu == wpe, f"override ignored: got {kn.waves_per_eu}"
            variants.append(
                {
                    "label": f"{name:<8} rows={kn.mfma_rows} wpe={wpe}",
                    "feat": name,
                    "wpe": wpe,
                    "knobs": kn,
                    "call": make_call(fn, tensors, d, feat["dropout"]),
                    "t": [],
                }
            )

    for v in variants:
        for _ in range(WARMUP):
            v["call"]()
    torch.cuda.synchronize()

    # The alternation. Round-robin over variants so that any drift over the
    # run lands on both arms of each comparison equally.
    for r in range(ROUNDS):
        for v in variants:
            v["t"].append(sample(v["call"]))

    print(f"\n=== head_dim {d} -- B={B} H={H} S={S} bf16, {ROUNDS} interleaved rounds x {REP} launches ===")
    hdr = f"{'variant':<24} {'best TF':>8} {'med TF':>8} {'spread':>7}   {'BLOCK_KV':>8} {'block_q':>7} {'tight':>5}"
    print(hdr)
    print("-" * len(hdr))
    for v in variants:
        ts = v["t"]
        best, med = 8 * base / min(ts), 8 * base / statistics.median(ts)
        kn = v["knobs"]
        print(
            f"{v['label']:<24} {best:8.0f} {med:8.0f} {100*(max(ts)-min(ts))/min(ts):6.1f}%"
            f"   {kn.block_kv:>8} {kn.block_q:>7} {str(kn.tight_registers):>5}"
        )

    print()
    for feat in FEATURES:
        name = "".join(k[0] for k in ("causal", "bias", "dropout") if feat[k]) or "dense"
        arms = [v for v in variants if v["feat"] == name]
        a2 = next(v for v in arms if v["wpe"] == 2)
        a1 = next(v for v in arms if v["wpe"] == 1)
        # Paired: each round measured both arms about a second apart, so the
        # per-round ratio is the drift-free comparison and its median is the
        # answer. The best-of ratio is reported beside it as a cross-check.
        ratios = [t2 / t1 for t2, t1 in zip(a2["t"], a1["t"])]
        med_ratio = statistics.median(ratios)
        best_ratio = min(a2["t"]) / min(a1["t"])
        table = "_FEATURE_OVERRIDES" if feat["bias"] else "_GEOMETRY[160]"
        verdict = "wpe=1 FASTER" if med_ratio > 1.005 else (
            "wpe=2 faster" if med_ratio < 0.995 else "tie")
        print(
            f"  {name:<8} ({a2['knobs'].mfma_rows}-row, {table:<18}): wpe=1 is "
            f"{100*(med_ratio-1):+.1f}% (paired median) / {100*(best_ratio-1):+.1f}% (best-of) "
            f"-> {verdict}"
        )


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or HDIMS
    print(f"GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
    import importlib.metadata as md

    print(f"flydsl {md.version('flydsl')}, kernels from "
          f"{Path(__file__).resolve().parent.parent}")
    for d in dims:
        run(d)
