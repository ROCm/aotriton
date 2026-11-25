#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sweep `flyc_attn_fwd` over (seqlen_q, seqlen_k) at fixed FLOPs, JIT-built.

Short sequences are measured at large B*H so that launch overhead and an
under-filled GPU do not stand in for kernel inefficiency: every point does the
same work as the reference shape (B=4 H=8 S=4096), with B grown to compensate.
The heads axis stays fixed at `--heads`, since it is the fastest grid axis and
an L2-locality lever on gfx950.

Environment, as for `bench_fwd_wpe_sweep.py`:

    ROCM_PATH                     a directory with llvm/bin/ld.lld under it
    AOTRITON_FLYDSL_KERNEL_ROOT   a FlyDSL source checkout matching the wheel
"""

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if not _KERNEL_ROOT:
    sys.exit("AOTRITON_FLYDSL_KERNEL_ROOT is unset")
sys.path.append(_KERNEL_ROOT)

import torch  # noqa: E402

import tuning_fwd  # noqa: E402
from fmha_meta import CAUSAL_BOTTOM_RIGHT, CAUSAL_NONE, FmhaInputMetadata  # noqa: E402

REF_WORK = 4 * 8 * 4096 * 4096  # B*H*Sq*Sk of the reference shape


def parse_overrides(items):
    out = {}
    for item in items or ():
        k, v = item.split("=", 1)
        out[k] = eval(v, {}, {})  # noqa: S307 -- developer tool, literal values
    return out


def time_call(call, iters, warmup, per_sample=10):
    """Median seconds per launch, and host seconds per launch.

    Each sample is `per_sample` back-to-back launches between one event pair,
    so the queue stays fed and host-side argument packing hides behind the
    kernel whenever the kernel is the longer of the two. The host figure says
    when it is not.
    """
    import time

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    host = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        t0 = time.perf_counter()
        for _ in range(per_sample):
            call()
        host.append((time.perf_counter() - t0) / per_sample)
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1e-3 / per_sample)
    return statistics.median(samples), statistics.median(host)


def max_err(q, k, v, o, causal, nb=2):
    """Max |o - ref| over the first `nb` batches, against an fp32 reference.

    Rows a bottom-right causal mask leaves empty (Sq > Sk) have no defined
    answer and are skipped.
    """
    q, k, v, o = (x[:nb].float() for x in (q, k, v, o))
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    sq, sk, hd = q.shape[2], k.shape[2], q.shape[3]
    s = q @ k.transpose(-1, -2) / hd**0.5
    rows = torch.arange(sq, device=q.device)
    if causal:
        keep = torch.arange(sk, device=q.device)[None, :] <= rows[:, None] + (sk - sq)
        s = s.masked_fill(~keep, float("-inf"))
        live = keep.any(-1)
    else:
        live = torch.ones(sq, dtype=torch.bool, device=q.device)
    ref = torch.softmax(s, -1).nan_to_num() @ v
    return (o - ref)[:, :, live].abs().max().item()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv_heads", type=int, default=None, help="GQA: number of K/V heads (default: --heads)")
    p.add_argument("--seqlens_q", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--seqlens_k", type=int, nargs="+", default=None,
                   help="default: same list as --seqlens_q")
    p.add_argument("--diagonal_only", action="store_true", help="only Sq == Sk")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--work_scale", type=float, default=1.0, help="multiply the per-point FLOPs")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--check", action="store_true", help="report max |err| against an fp32 reference")
    p.add_argument("--knob", action="append", help="NAME=VALUE knob override, repeatable")
    args = p.parse_args()

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    hd, H = args.head_dim, args.heads
    HK = args.kv_heads or H
    meta = FmhaInputMetadata(num_heads=H,
                             num_kv_heads=HK,
                             block_dmodel=None,
                             head_dim=hd,
                             causal_type=CAUSAL_BOTTOM_RIGHT if args.causal else CAUSAL_NONE,
                             dtype_str={"bf16": "bf16", "fp16": "f16"}[args.dtype])
    # No LSE store unless asked: `plan` keeps it on, as AOT binaries do.
    overrides = {"return_lse": False, **parse_overrides(args.knob)}
    print(f"# {torch.cuda.get_device_name(0)}  hd={hd} H={H} {args.dtype} causal={args.causal} "
          f"overrides={overrides}")
    print(f"{'Sq':>6} {'Sk':>6} {'B':>6} {'WGs':>7} {'us':>9} {'TFLOPs':>8} {'host_us':>8} {'build':>5}"
          + (f" {'max_err':>8}" if args.check else ""))

    # The policy plans every cell, per-call rules included; cells whose knobs
    # agree share a build. --knob pins win over the rules.
    built = {}

    def build_for(sq, sk):
        planned, knobs = tuning_fwd.plan("gfx950", meta, overrides=overrides, max_seqlen_q=sq, max_seqlen_k=sk)
        key = json.dumps(asdict(knobs), sort_keys=True)
        if key not in built:
            fn = tuning_fwd.builder("gfx950", planned, knobs)
            t = fn.traits
            built[key] = (len(built), fn)
            print(f"# build {len(built) - 1}: waves={t.NUM_WAVES} BLOCK_M={t.BLOCK_M} BLOCK_N={t.BLOCK_N} "
                  f"wpe={knobs.waves_per_eu} exact_kv_tiles={knobs.exact_kv_tiles} "
                  f"hdim_qk_floor={knobs.hdim_qk_floor}", flush=True)
        return built[key]

    sks = args.seqlens_k or args.seqlens_q
    for sq in args.seqlens_q:
        for sk in sks:
            if args.diagonal_only and sq != sk:
                continue
            build_index, fn = build_for(sq, sk)
            t = fn.traits
            B = max(1, round(REF_WORK * args.work_scale / (H * sq * sk)))
            B = min(B, 65535)
            q = torch.randn(B, H, sq, hd, device="cuda", dtype=dt)
            k = torch.randn(B, HK, sk, hd, device="cuda", dtype=dt)
            v = torch.randn(B, HK, sk, hd, device="cuda", dtype=dt)
            o = torch.empty_like(q)

            def call():
                fn(q, k, v, o, B, sq, seqlen_k=sk, scale=None, lse=None)

            sec, host = time_call(call, args.iters, args.warmup)
            # A non-window causal build is bottom-right aligned
            # (`_resolve_window_args`): row i keeps keys j <= i + sk - sq.
            if args.causal:
                pairs = sum(max(0, min(i + 1 + sk - sq, sk)) for i in range(sq))
            else:
                pairs = sq * sk
            flops = 4.0 * B * H * pairs * hd
            wgs = B * H // t.Q_HEADS_PER_WG * -(-sq // t.BLOCK_M)
            print(f"{sq:>6} {sk:>6} {B:>6} {wgs:>7} {sec * 1e6:>9.1f} {flops / sec * 1e-12:>8.1f} {host * 1e6:>8.1f}"
                  f" {build_index:>5}"
                  + (f" {max_err(q, k, v, o, args.causal):>8.1e}" if args.check else ""), flush=True)
            del q, k, v, o
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
