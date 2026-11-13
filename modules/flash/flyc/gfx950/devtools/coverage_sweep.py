#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage sweep of the gfx950 flyc attention kernels (JIT): correctness and speed per build.

For every group `(kernel, head_dim, dtype, mask, feature)` in the shard, and
every candidate build the policy could legitimately pick, three passes:

    compile   warm the FlyDSL JIT disk cache (FLYDSL_COMPILE_ONLY=1, parallel)
    check     dense and packed-varlen cells vs an fp32 torch reference
    time      dense cells at fixed FLOPs, candidates interleaved

Each group runs in its own worker process (`coverage_worker.py`). A GPU fault
kills that process; the driver attributes it to the launch the worker had
announced (`@@BEGIN`), writes a FAULT record with the tail of the worker's
stderr (where the ROCr message is), and restarts the worker, which skips every
record already written. A worker silent for `--stall_timeout` seconds is killed
and its pending launch recorded as a FAULT (hang). Rerunning the same command
resumes: every record already in `--out_dir` is skipped.

Shard by head_dim (and kernel) across GPUs; pick the GPU with
HIP_VISIBLE_DEVICES. Each driver appends to its own records file, so several
may share one `--out_dir`.

    source ~/scratch/coverage/env.sh            # ROCM_PATH, kernel root, JIT key
    HIP_VISIBLE_DEVICES=6 python coverage_sweep.py --out_dir ~/scratch/coverage/run1 \\
        --head_dims 128 --kernels fwd bwd --phases compile check time
    python coverage_summary.py ~/scratch/coverage/run1

Required environment: AOTRITON_FLYDSL_KERNEL_ROOT (the kernel tree
`third_party/flydsl-kernel.txt` pins), FLYDSL_EXTRA_SOURCE_DIRS
(`modules/flash/flyc`, or the JIT cache serves stale binaries), ROCM_PATH.
"""

import argparse
import concurrent.futures as cf
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_space as space  # noqa: E402

WORKER = str(Path(__file__).resolve().parent / "coverage_worker.py")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_worker(argv, env, log_path, stall_timeout, compile_timeout):
    """Run one worker; stream its combined output to `log_path`.

    Returns `(returncode, lines, stalled)`. The stall clock is
    `compile_timeout` while the worker is between `@@COMPILE` and `@@COMPILED`,
    `stall_timeout` otherwise.
    """
    lines = []
    stalled = False
    with open(log_path, "a") as lf:
        lf.write(f"\n### {time.ctime()} {' '.join(argv)}\n")
        lf.flush()
        proc = subprocess.Popen(argv,
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                                bufsize=0)
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        last = time.time()
        compiling = False
        buf = b""
        while True:
            events = sel.select(timeout=5.0)
            if events:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                lf.write(chunk.decode(errors="replace"))
                lf.flush()
                buf += chunk
                *done, buf = buf.split(b"\n")
                for raw in done:
                    line = raw.decode(errors="replace")
                    lines.append(line)
                    if line.startswith("@@COMPILE "):
                        compiling = True
                    elif line.startswith("@@COMPILED"):
                        compiling = False
                last = time.time()
            limit = compile_timeout if compiling else stall_timeout
            if time.time() - last > limit:
                stalled = True
                os.killpg(proc.pid, signal.SIGKILL)
                lf.write(f"\n### killed: no output for {limit}s\n")
                break
        if buf:
            lines.append(buf.decode(errors="replace"))
        rc = proc.wait()
    return rc, lines, stalled


def pending_marker(lines):
    """The last `@@BEGIN` key or `@@COMPILE` id with no matching close, as (tag, payload)."""
    open_begin = None
    open_compile = None
    for line in lines:
        if line.startswith("@@BEGIN "):
            open_begin = line[len("@@BEGIN "):]
        elif line.startswith("@@END"):
            open_begin = None
        elif line.startswith("@@COMPILE "):
            open_compile = line[len("@@COMPILE "):]
        elif line.startswith("@@COMPILED"):
            open_compile = None
    if open_begin is not None:
        return "BEGIN", open_begin
    if open_compile is not None:
        return "COMPILE", open_compile
    return None, None


def stderr_tail(lines, n=40):
    keep = [ln for ln in lines if not ln.startswith("@@")]
    return "\n".join(keep[-n:])


def run_group(group, pass_name, args, env, out_path, log_dir):
    """Run one group's pass to completion, restarting the worker after each crash."""
    tag = space.group_tag(group)
    log_path = os.path.join(log_dir, f"{tag}.{pass_name}.log")
    crashed = []
    attempts = 0
    no_progress = 0
    while True:
        attempts += 1
        n_before = len(space.load_records(args.out_dir))
        argv = [sys.executable,
                WORKER,
                "--group",
                json.dumps(group),
                "--pass_name",
                pass_name,
                "--out_dir",
                args.out_dir,
                "--out",
                out_path,
                "--crashed_builds",
                json.dumps(crashed),
                "--seqlens",
                *map(str, args.seqlens)]
        t0 = time.time()
        rc, lines, stalled = run_worker(argv, env, log_path, args.stall_timeout, args.compile_timeout)
        if rc == 0 and not stalled:
            return attempts, time.time() - t0
        what, payload = pending_marker(lines)
        tail = stderr_tail(lines)
        if what == "BEGIN":
            key = json.loads(payload)
            names = ("kernel", "hd", "dtype", "mask", "feature", "layout", "sq", "sk", "candidate", "pass")
            rec = dict(zip(names, key))
            rec.update(status="FAULT",
                       error=("hang: no output for %ds" % args.stall_timeout) if stalled else f"worker exit code {rc}",
                       stderr=tail,
                       ts=time.time())
            space.append_record(out_path, rec)
            log(f"  FAULT {tag} {rec['layout']} {rec['sq']}x{rec['sk']} {rec['candidate']} ({rec['error']})")
        elif what == "COMPILE":
            crashed.append(payload)
            log(f"  build {payload} crashed the worker (rc {rc}); marking BUILD_ERROR")
        else:
            log(f"  worker for {tag} exited {rc} with no pending launch; tail:\n{tail}")
        n_after = len(space.load_records(args.out_dir))
        no_progress = 0 if n_after > n_before else no_progress + 1
        if no_progress >= 3 or attempts >= args.max_restarts:
            log(f"  giving up on {tag} {pass_name} after {attempts} attempts")
            return attempts, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[1:]))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--kernels", nargs="+", choices=space.KERNELS, default=list(space.KERNELS))
    p.add_argument("--head_dims", type=int, nargs="+", default=list(space.HEAD_DIMS))
    p.add_argument("--dtypes", nargs="+", choices=space.DTYPES, default=list(space.DTYPES))
    p.add_argument("--masks", nargs="+", choices=space.MASKS, default=list(space.MASKS))
    p.add_argument("--features", nargs="+", choices=space.FEATURES, default=list(space.FEATURES))
    p.add_argument("--seqlens", type=int, nargs="+", default=list(space.SEQLENS))
    p.add_argument("--phases", nargs="+", choices=space.PASSES, default=list(space.PASSES))
    p.add_argument("--compile_jobs", type=int, default=32, help="parallel workers in the compile phase")
    p.add_argument("--stall_timeout", type=int, default=300, help="seconds without output before a hang is declared")
    p.add_argument("--compile_timeout", type=int, default=1800, help="the same, while a build compiles")
    p.add_argument("--max_restarts", type=int, default=200)
    args = p.parse_args()

    for var in ("AOTRITON_FLYDSL_KERNEL_ROOT", "FLYDSL_EXTRA_SOURCE_DIRS", "ROCM_PATH"):
        if not os.environ.get(var):
            sys.exit(f"{var} is unset; source ~/scratch/coverage/env.sh (see --help)")
    if not os.environ.get("HIP_VISIBLE_DEVICES"):
        sys.exit("set HIP_VISIBLE_DEVICES to the one GPU this shard owns")
    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    gpu = os.environ["HIP_VISIBLE_DEVICES"].replace(",", "_")
    out_path = os.path.join(args.out_dir, f"records-{socket.gethostname()}-gpu{gpu}.jsonl")

    groups = space.groups(args.kernels, args.head_dims, args.dtypes, args.masks, args.features)
    log(f"{len(groups)} groups, phases {args.phases}, records -> {out_path}")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    timings = {}
    for phase in args.phases:
        t0 = time.time()
        if phase == "compile":
            cenv = dict(env, FLYDSL_COMPILE_ONLY="1")
            with cf.ThreadPoolExecutor(max_workers=args.compile_jobs) as ex:
                futs = {ex.submit(run_group, g, phase, args, cenv, out_path, log_dir): g for g in groups}
                for i, fut in enumerate(cf.as_completed(futs), 1):
                    g = futs[fut]
                    att, dt = fut.result()
                    log(f"compile {i}/{len(groups)} {space.group_tag(g)} {dt:.0f}s ({att} attempt(s))")
        else:
            for i, g in enumerate(groups, 1):
                att, dt = run_group(g, phase, args, env, out_path, log_dir)
                log(f"{phase} {i}/{len(groups)} {space.group_tag(g)} {dt:.0f}s ({att} attempt(s))")
        timings[phase] = time.time() - t0
        log(f"phase {phase} done in {timings[phase]:.0f}s")
    with open(os.path.join(args.out_dir, f"phase-times-{socket.gethostname()}-gpu{gpu}.jsonl"), "a") as f:
        f.write(json.dumps(dict(ts=time.time(), argv=sys.argv, timings=timings)) + "\n")


if __name__ == "__main__":
    main()
