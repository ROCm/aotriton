#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sweep `waves_per_eu` 1 against 2 for `flyc_attn_fwd`, every shipped rung.

Why this exists. `_GFX950_FALLBACK` in `fmha_tuning_gfx950.py` sets
`waves_per_eu=2` for the whole forward -- it sits among "defaults the policy
has no shape-dependent opinion about", and every one of the 216 shipped
`flyc_attn_fwd` builds carries it. The hardware grants it at the narrow rungs
and refuses it at the wide ones, which is 120 of the 216 builds emitting

    failed to meet occupancy target given by 'amdgpu-waves-per-eu'
    in 'flyc_attn_fwd': desired occupancy was 2, final occupancy is 1

The split is LDS, and it is not close (`LDS_CAP_BYTES` is 163840):

    head_dim    LDS/workgroup   two of them   VGPR+AGPR   2 waves/EU?
        32             17024         34048      124-167   yes
        64             34048         68096      164-219   yes
        96             51072        102144      204-237   yes
       128             68096        136192      244-256   yes
       160             85120        170240      326-428   no -- LDS
       192            102144        204288      416-696   no -- LDS
       224            119168        238336      544-768   no -- LDS
       256            136192        272384      650-768   no -- LDS
       512            136192        272384      444-724   no -- LDS

At 160 and wider the second workgroup cannot land whatever the registers do,
so the request is refused before the allocator gets a say. What that costs, if
anything, is the question this answers -- separately from the warning, because
an unmet hint is still a register budget for the scheduler.

**Most of the answer is in the artifacts, not here.** Building the 120 refused
kernels both ways gives 118 byte-identical and 2 differing -- and rebuilding
all 216 at *unchanged* settings also gives 2 differing, because this backend is
not bit-reproducible. One kernel appears in both sets (head_dim 256 bf16
`CAUSAL_TYPE=3 ENABLE_DROPOUT=True BIAS_TYPE=0 PADDED_HEAD=False`), alternating
between two forms at identical register counts and identical size. So at the
refused rungs the signal is 2 against a floor of 2, and this run exists to
confirm there is nothing there rather than to discover it.

The narrow rungs are the live question, since 2 is granted there and dropping
it really would change the schedule. head_dim 96 is where it does: 17 of 24
builds change, 13 raising VGPRs, all in the same direction. Note that this
harness does *not* reach those -- it builds at an exact head_dim with no bias,
so `PADDED_HEAD=False BIAS_TYPE=0`, and the head_dim 96 builds that move are
the padded and bias ones. Reading a tie at 96 here as "the knob does not
matter at 96" would be reading it off the one variant that does not change.

**Order-alternation, not just interleaving.** The pair order flips every round
(`(2,1)` on even rounds, `(1,2)` on odd), so each arm spends half its samples
in each slot. Interleaving alone cancels drift over minutes but not a *fixed*
per-slot bias, and that bias is worth ~5% here: an arm that always runs second
inherits its neighbour's warm L2 every single round. That trap produced a
reproducible, two-runs-agreeing +5.5% on byte-identical `flyc_bwd_dkdv`
binaries -- see `bench_dkdv160_wpe_ab.py`. `--fixed_order` reproduces the
broken methodology on demand; the per-slot columns below are the check that it
is not happening.

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
        modules/flash/flyc/devtools/bench_fwd_wpe_sweep.py [--head_dims 256 512] [...]

The GPU is shared, so the run refuses to start on a board that is already busy
-- see `preflight`. Pin it with HIP_VISIBLE_DEVICES.
"""

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# AOTriton's VENDORED kernels, not FlyDSL's parity copies -- these are what
# ships, and the two trees have diverged (the varlen build axis is gone here).
# The kernels are this directory's parent, and that is the whole of what this
# tool knows about the repository layout.
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

from flash_attn_func_gfx950 import (  # noqa: E402
    build_flash_attn_func_gfx950_module_primary as build_primary,
)
from fmha_tuning_gfx950 import FmhaInputMetadata, fmha_knobs  # noqa: E402

# Every head_dim with a shipped `flyc_attn_fwd` build. 384 is deliberately
# absent: it is a supported geometry in the tuning policy but not a rung the
# operator instantiates, so measuring it would tune something nothing loads.
HEAD_DIMS = (32, 64, 96, 128, 160, 192, 224, 256, 512)

# The feature points. `dense` is the reporting shape the tuning tables are
# written against; `causal+dropout` is carried because it is the only point
# where the wpe setting was observed to change the emitted code at all, at
# head_dim 256. bias is left out -- it is a separate build family whose wpe
# comes from the same single fallback, so it moves with whatever this decides.
FEATURES = (
    {"causal": False, "dropout": False},
    {"causal": True, "dropout": True},
)

REP = 20
ROUNDS = 16  # even, so the order flip leaves each arm the same slot count
WARMUP = 10


def shape_for(hd):
    """Batch/heads/seqlen, from `ab_knobs_gfx950.py` -- shrinking the batch as
    the head widens keeps the working set roughly constant across rungs."""
    return (4, 8, 4096) if hd <= 128 else (2, 8, 4096) if hd <= 256 else (1, 8, 4096)


def preflight(threshold_pct=5, threshold_vram_gb=2.0):
    """Refuse to benchmark a board somebody else is using.

    The machine is shared and a neighbouring test run is not visible from
    inside the process -- it shows up only as numbers that are quietly wrong,
    which is worse than a crash. HIP_VISIBLE_DEVICES renumbers the device to 0
    for us, so the check has to go through the *physical* index.
    """
    phys = os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        out = subprocess.run(["amd-smi", "monitor"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as e:
        print(f"WARNING: amd-smi unavailable ({e}); skipping the busy-GPU check")
        return
    for line in out.splitlines():
        f = line.split()
        if not f or f[0] != phys:
            continue
        # `amd-smi monitor` separates value and unit, so the row tokenises as
        # ... '158' 'MHz' '0' '%' '0' '%' 'N/A' '0' '%' '0.3/288.0' 'GB'.
        # GFX% is therefore the token after 'MHz', not one ending in '%'.
        try:
            gfx = float(f[f.index("MHz") + 1])
            vram = float(next(x for x in f if "/" in x and x != "N/A").split("/")[0])
        except (StopIteration, ValueError, IndexError):
            print(f"WARNING: could not parse amd-smi row for GPU {phys}: {line!r}")
            return
        print(f"preflight: physical GPU {phys} at {gfx:.0f}% busy, {vram:.1f} GB VRAM in use")
        if gfx > threshold_pct or vram > threshold_vram_gb:
            sys.exit(
                f"REFUSING TO RUN: GPU {phys} is in use ({gfx:.0f}% busy, {vram:.1f} GB). "
                f"Pick an idle board with HIP_VISIBLE_DEVICES."
            )
        return
    print(f"WARNING: GPU {phys} not found in amd-smi output; skipping the busy-GPU check")


def make_inputs(hd, B, H, S, dt):
    q, k, v = (torch.randn(B, H, S, hd, device="cuda", dtype=dt) for _ in range(3))
    o = torch.empty(B, H, S, hd, device="cuda", dtype=dt)
    return q, k, v, o


def make_call(fn, t, B, S, dropout):
    q, k, v, o = t
    extra = (
        {"dropout_p": 0.5, "philox_seed": 0x1BF52, "philox_offset1": 0x1D4B42}
        if dropout
        else {}
    )

    def call():
        fn(q, k, v, o, B, S, seqlen_k=S, scale=None, lse=None, **extra)

    return call


def sample(call, rep=REP):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


def run(hd, dt, fixed_order, rounds):
    B, H, S = shape_for(hd)
    results = []
    for feat in FEATURES:
        name = "+".join(k for k in ("causal", "dropout") if feat[k]) or "dense"
        tensors = make_inputs(hd, B, H, S, dt)
        meta = FmhaInputMetadata(
            num_heads=H, head_dim=hd, causal=feat["causal"], dropout=feat["dropout"],
            dtype_str={torch.bfloat16: "bf16", torch.float16: "fp16"}[dt],
        )
        arms = {}
        for wpe in (2, 1):
            try:
                knobs = fmha_knobs("gfx950", waves_per_eu=wpe).resolve(meta)
                fn = build_primary(meta, knobs)
                assert knobs.waves_per_eu == wpe, f"override ignored: {knobs.waves_per_eu}"
                c = make_call(fn, tensors, B, S, feat["dropout"])
                c()
                # `slot[i]` collects samples taken in round-position i, so a
                # per-slot gap can be read off directly rather than inferred.
                arms[wpe] = {"call": c, "t": [], "slot": ([], [])}
            except Exception as e:  # noqa: BLE001
                print(f"hd={hd:>4} {name:<14} wpe={wpe}: BUILD {type(e).__name__}: {e}")
        if len(arms) != 2:
            continue

        torch.cuda.synchronize()
        for _ in range(WARMUP):
            for a in arms.values():
                a["call"]()
        torch.cuda.synchronize()

        for r in range(rounds):
            order = (2, 1) if (fixed_order or r % 2 == 0) else (1, 2)
            for slot, wpe in enumerate(order):
                s = sample(arms[wpe]["call"])
                arms[wpe]["t"].append(s)
                arms[wpe]["slot"][slot].append(s)

        # Causal halves the work; dropout does not change the FLOP count.
        flops = 2.0 * 2 * B * H * S * S * hd * (0.5 if feat["causal"] else 1.0)
        results.append((name, B, H, S, arms, flops))
        del tensors
        torch.cuda.empty_cache()
    return results


def report(hd, results):
    for name, B, H, S, arms, flops in results:
        tf = lambda ts: flops / statistics.median(ts) * 1e-12  # noqa: E731
        m2, m1 = tf(arms[2]["t"]), tf(arms[1]["t"])
        sp = lambda ts: 100 * (max(ts) - min(ts)) / statistics.median(ts)  # noqa: E731
        # Slot gap: how much the round-position alone is worth. If this is
        # comparable to the wpe gap, the wpe gap is not real.
        slot_gap = []
        for wpe in (2, 1):
            a, b = arms[wpe]["slot"]
            if a and b:
                slot_gap.append(100 * (statistics.median(a) / statistics.median(b) - 1))
        delta = 100 * (m1 / m2 - 1)
        verdict = "wpe=1" if delta > 1.0 else ("wpe=2" if delta < -1.0 else "tie")
        print(
            f"hd={hd:>4} {name:<14} B={B} S={S}  "
            f"wpe2 {m2:7.1f} TF (spr {sp(arms[2]['t']):4.1f}%)  "
            f"wpe1 {m1:7.1f} TF (spr {sp(arms[1]['t']):4.1f}%)  "
            f"{delta:+6.1f}%  slot-bias "
            + "/".join(f"{g:+.1f}%" for g in slot_gap)
            + f"  -> {verdict}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--head_dims", type=int, nargs="+", default=list(HEAD_DIMS))
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument(
        "--fixed_order", action="store_true",
        help="do NOT alternate the pair order -- reproduces the slot-biased methodology",
    )
    p.add_argument("--skip_preflight", action="store_true")
    args = p.parse_args()

    if not args.skip_preflight:
        preflight()

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    print(f"GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
    import importlib.metadata as md

    print(f"flydsl {md.version('flydsl')}, kernels from "
          f"{Path(__file__).resolve().parent.parent}")
    print(
        f"{args.dtype}, {args.rounds} rounds x {REP} launches, "
        f"pair order {'FIXED (2,1) -- slot-biased' if args.fixed_order else 'alternating'}\n"
    )
    for hd in args.head_dims:
        report(hd, run(hd, dt, args.fixed_order, args.rounds))


if __name__ == "__main__":
    main()
