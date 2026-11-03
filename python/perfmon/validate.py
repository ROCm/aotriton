# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Phase 1 driver: runs a list of measurements through the native runner of one
GPU of this host, and writes one JSON line per measurement.

It needs no database, node or queue. Run it inside a perfmon node
environment (.perfmon/host/node_env.sh: AOTRITON_WORKDIR, AOTRITON_SRC and a
venv):

    python -m aotriton.perfmon.validate --jobs jobs.jsonl --out out.jsonl \\
                                        --arch gfx950 --gpu 0 \\
                                        --methods hipgraph_ev100,stream_ev \\
                                        --reps 5 --new_process_per_rep

A job is a JSON object (one per line of the jobs file; see validate_plans.py
for generators):

    {"subject": "aotriton-0.14.2b+rocm10.1.0",
     "entry": {<flash entry fields>},           # or "entry_pon": "<pon>"
     "iface": "op_attn_fwd",
     "backend": "-" | "<name>" | "all",         # "all": "-" and every
     "label": "..."}                             #   backend `enumerate` lists

"all" asks the runner (`enumerate`) for the backends of the entry and iface.
Each result line carries the job's fields plus method, rep, status ("ok",
"unavailable" or "error"), stats (the runner's reply, times in microseconds)
or message.

The runner is the same executable perfmon's node drives through
ExaidPerfmonRunner; this driver speaks its line protocol directly so that it
can set PERFMON_TIMING_METHOD per process and start fresh processes at will.
Standard library only (plus aotriton.utils.pon).
"""

import argparse
import json
import os
import selectors
import statistics
import subprocess
import sys
import time
from pathlib import Path

from aotriton.utils.pon import render_pon

METHODS = ('hipgraph_ev100', 'stream_ev')
UNFORCED = '-'
KEEPALIVE_PREFIX = 'OVERHEATING:'
ENUMERATE_TIMEOUT = 120.0


def job_pon(job: dict) -> str:
    if 'entry_pon' in job:
        return job['entry_pon']
    return render_pon(job['entry'])


def parse_reply(line: str) -> dict:
    """One runner reply line -> {'status': 'ok'|'unavailable'|'error', ...}.
    'ok' carries 'data' (the parsed JSON), the others a 'message'."""
    line = line.rstrip('\n')
    if line.startswith('OK '):
        try:
            return {'status': 'ok', 'data': json.loads(line[3:])}
        except json.JSONDecodeError as e:
            return {'status': 'error', 'message': f'bad OK payload ({e}): {line[:200]}'}
    if line == 'OK':
        return {'status': 'ok', 'data': {}}
    if line.startswith('UNAVAILABLE'):
        return {'status': 'unavailable', 'message': line[len('UNAVAILABLE'):].strip()}
    return {'status': 'error', 'message': line}


def read_jobs(lines) -> list[dict]:
    jobs = []
    for n, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        job = json.loads(line)
        for key in ('subject', 'iface', 'backend'):
            if key not in job:
                raise ValueError(f'jobs line {n}: missing "{key}"')
        if 'entry' not in job and 'entry_pon' not in job:
            raise ValueError(f'jobs line {n}: needs "entry" or "entry_pon"')
        jobs.append(job)
    return jobs


def group_by_subject(jobs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Consecutive jobs of one subject share a runner process; the order of
    first appearance is kept."""
    groups: list[tuple[str, list[dict]]] = []
    for job in jobs:
        if groups and groups[-1][0] == job['subject']:
            groups[-1][1].append(job)
        else:
            groups.append((job['subject'], [job]))
    return groups


def summarize(values: list[float]) -> dict:
    """Spread of a list of medians: min, max, mean and the relative spread
    (max - min) / min, plus the coefficient of variation in percent."""
    if not values:
        return {}
    lo, hi = min(values), max(values)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {'n': len(values),
            'min': lo,
            'max': hi,
            'mean': mean,
            'spread_pct': (hi - lo) / lo * 100.0 if lo else float('nan'),
            'cv_pct': sd / mean * 100.0 if mean else float('nan')}


class RunnerProcess:
    """One native runner on one GPU, spoken to over its line protocol."""

    def __init__(self, launcher: Path, subject: str, arch: str, gpu: int, method: str, stderr_path: Path):
        self._argv = ['bash', str(launcher), '--subject', subject, '--module', 'flash', '--gpu', str(gpu),
                      '--arch', arch]
        self._env = dict(os.environ)
        self._env['PERFMON_TIMING_METHOD'] = method
        self._stderr_path = stderr_path
        self._spawn()

    def _spawn(self):
        self._stderr = open(self._stderr_path, 'ab')
        self._proc = subprocess.Popen(self._argv,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=self._stderr,
                                      env=self._env,
                                      text=True,
                                      bufsize=1)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._proc.stdout, selectors.EVENT_READ)
        #: OVERHEATING keep-alives seen since the last command.
        self.keepalives = 0
        #: Set when a command found the process gone or stuck.
        self.dead = False

    def restart(self):
        self.close()
        self._spawn()

    def command(self, text: str, timeout: float) -> dict:
        """Send one command; return the parsed reply. The deadline restarts at
        each OVERHEATING keep-alive, like the node's."""
        self.keepalives = 0
        reply = self._command(text, timeout)
        if reply.get('dead'):
            self.dead = True
        return reply

    def _command(self, text: str, timeout: float) -> dict:
        try:
            self._proc.stdin.write(text + '\n')
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return {'status': 'error', 'message': f'runner is gone: {e}', 'dead': True}
        while True:
            if not self._sel.select(timeout):
                return {'status': 'timeout', 'message': f'timeout after {timeout}s', 'dead': True}
            line = self._proc.stdout.readline()
            if line == '':
                return {'status': 'error', 'message': f'runner exited with {self._proc.poll()}', 'dead': True}
            if line.startswith(KEEPALIVE_PREFIX):
                self.keepalives += 1
                continue
            return parse_reply(line)

    def close(self):
        try:
            if self._proc.poll() is None and self.dead:
                self._proc.kill()       # stuck in a long timed loop: exit would wait for it
                self._proc.wait()
            elif self._proc.poll() is None:
                try:
                    self._proc.stdin.write('exit\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=20)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self._proc.kill()
                    self._proc.wait()
        finally:
            self._sel.close()
            self._stderr.close()


def run_job(runner, job: dict, timeout: float, probe: bool = False) -> list[dict]:
    """Run one job on a live runner; one record per measured backend.

    With `probe`, `timeout` is how long a measure may take before it counts as
    "started": the entry was built, its buffers allocated and its first launch
    accepted (a failure of any of these replies at once), and the long timed
    loop is not waited for. The runner is then dead, and the caller relaunches
    it."""
    pon = job_pon(job)
    iface = job['iface']
    backend = job['backend']
    base = {k: v for k, v in job.items() if k not in ('backend', 'entry_pon')}
    records = []
    if backend == 'all':
        listing = runner.command(f'enumerate {pon} {iface}', max(timeout, ENUMERATE_TIMEOUT))
        if listing['status'] != 'ok':
            listing.pop('dead', None)
            listing['status'] = 'error' if listing['status'] == 'timeout' else listing['status']
            return [dict(base, backend='all', **listing)]
        backends = [UNFORCED] + list(listing['data'].get('backends', []))
    else:
        backends = [backend]
    for name in backends:
        t0 = time.time()
        reply = runner.command(f'measure {pon} {iface} {name}', timeout)
        status = reply['status']
        if status == 'timeout':
            status = 'started' if probe else 'error'
        record = dict(base, backend=name, status=status, wall_s=round(time.time() - t0, 3))
        if reply['status'] == 'ok':
            record['stats'] = reply['data']
        else:
            record['message'] = reply['message']
        if runner.keepalives:
            record['keepalives'] = runner.keepalives
        records.append(record)
        if reply.get('dead'):
            if probe and status == 'started' and hasattr(runner, 'restart'):
                runner.restart()    # killed on purpose: the next backend needs a runner
                continue
            break           # the process is gone or stuck: the caller relaunches
    return records


def run(jobs: list[dict], out, args, launcher: Path) -> int:
    failures = 0
    stderr_path = Path(args.out + '.stderr')
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f'--methods: unknown timing method {method!r}; one of {METHODS}')
    live: tuple[str, RunnerProcess] | None = None

    def emit(job, records, method, rep):
        nonlocal failures
        for r in records:
            r.update(method=method, rep=rep, arch=args.arch, gpu=args.gpu)
            if r['status'] == 'error':
                failures += 1
            out.write(json.dumps(r) + '\n')
            out.flush()

    def close_live():
        nonlocal live
        if live is not None:
            live[1].close()
            live = None

    for method in methods:
        for rep in range(args.reps):
            for subject, group in group_by_subject(jobs):
                for job in group:
                    if live is None or live[0] != subject:
                        close_live()
                        live = (subject, RunnerProcess(launcher, subject, args.arch, args.gpu, method, stderr_path))
                    timeout = args.probe_seconds or args.timeout
                    records = run_job(live[1], job, timeout, probe=bool(args.probe_seconds))
                    emit(job, records, method, rep)
                    if live[1].dead:
                        close_live()        # relaunch for the next job
                if args.new_process_per_rep:
                    close_live()
            close_live()                    # a method change needs a new environment
    return failures


def find_launcher() -> Path:
    for root in (os.environ.get('AOTRITON_SRC'), os.getcwd()):
        if root:
            path = Path(root) / 'perfmon' / 'launch_runner.sh'
            if path.is_file():
                return path
    raise SystemExit('perfmon/launch_runner.sh not found; set AOTRITON_SRC to the checkout')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--jobs', required=True, help='JSON lines of jobs; - for stdin')
    p.add_argument('--out', required=True, help='results, JSON lines (appended)')
    p.add_argument('--arch', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--methods', default='hipgraph_ev100', help='comma list of PERFMON_TIMING_METHOD values')
    p.add_argument('--reps', type=int, default=1, help='repeat the whole job list this many times')
    p.add_argument('--new_process_per_rep', action='store_true',
                   help='start a fresh runner process for every repetition')
    p.add_argument('--probe_seconds', type=float, default=0.0,
                   help='allocation probe: a measure still running after this many seconds counts as '
                        '"started" and its runner is killed (for entries whose timed loop takes minutes)')
    p.add_argument('--timeout', type=float, default=900.0, help='seconds per reply line')
    args = p.parse_args(argv)
    lines = sys.stdin if args.jobs == '-' else open(args.jobs, encoding='utf-8')
    jobs = read_jobs(lines)
    launcher = find_launcher()
    with open(args.out, 'a', encoding='utf-8') as out:
        failures = run(jobs, out, args, launcher)
    print(f'{len(jobs)} jobs, {failures} errors; results in {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
