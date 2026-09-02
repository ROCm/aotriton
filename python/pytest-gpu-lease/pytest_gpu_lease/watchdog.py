# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detect a wedged pytest-xdist worker from outside its process, and kill it.

Run beside pytest, not inside it: ``python -m pytest_gpu_lease.watchdog --lockfile
... --workers N``. It polls the lease lock file that ``plugin.py`` already
maintains -- one 4096-byte page per GPU, write-locked by whichever worker owns
that GPU for the run -- and reads the 8-byte deadline written there: an initial
one set by ``gpu_id`` the moment it takes the lease, refreshed before every
test and zeroed after by a ``pytest_runtest_protocol`` hookwrapper.

Why the lock file is the right substrate, in one line: ``fcntl(F_GETLK)`` fills
``l_pid`` with the pid holding a byte range, and the kernel releases a record
lock automatically when its owner dies. So this module never remembers a pid --
it asks the kernel again immediately before every signal. A worker that already
died on its own between one poll and the next is simply unlocked by the time we
look, and gets nothing sent to it: pid reuse is not a hazard here, because a
dead process holds no lock to be mistaken for a live one.

Escalation is SIGTERM, a grace period, then SIGKILL if the page is still locked.
``plugin.py`` registers SIGTERM with ``faulthandler`` on every leasing worker, so
the SIGTERM here is not a death sentence by itself -- it is a request for a
stack dump, C-level and GIL-free, naming whatever frame the wedge is in. Only a
worker that ignores that and is still holding its page after ``--grace`` seconds
gets SIGKILLed.

Shares ``PAGE_SIZE`` and ``STRUCT_FLOCK`` with ``plugin.py`` by importing them
rather than restating them, since a copy that drifts from the writer's layout
would silently misread every deadline.
"""

import argparse
import fcntl
import os
import signal
import struct
import sys
import time

from .plugin import PAGE_SIZE, STRUCT_FLOCK, _DEFAULT_BUDGET_S

# Default cadence: frequent enough that a 600s budget is caught within a small
# fraction of itself, infrequent enough that polling 4-8 pages is noise next to
# a 15-22h pass.
_DEFAULT_POLL_INTERVAL_S = 5.0

# Default grace between SIGTERM and SIGKILL: long enough for faulthandler to
# flush a stack dump to stderr (a syscall, not instant under load) and for a
# process that was already exiting on its own to finish doing so, short enough
# that a genuinely wedged worker is not left idle for long after being caught.
_DEFAULT_GRACE_S = 30.0

# Consecutive empty polls (no page locked at all) *after having seen at least
# one page locked* before the watchdog exits on its own. This is the fallback
# for the case run-test.sh itself cannot reach its own trap -- e.g. it is
# killed with SIGKILL, which a trap cannot catch -- so a watchdog started for
# one run does not outlive it and go on signalling pids that by then belong to
# somebody else. At the default poll interval this is one minute of a
# completely idle lock file, counted only once the run has actually gone idle
# -- i.e. its last worker released its page -- not before the run has started.
# The "at least one" qualifier is load bearing, not a nicety: run-test.sh
# starts the watchdog before pytest even begins collecting, specifically so a
# wedge during collection is covered too, and a Level-3 pass's collection
# (~330k tests, a conftest that imports torch) takes minutes -- far longer
# than idle_polls * poll_interval at the defaults. A watchdog that started
# counting immediately would exit on its own well before the first worker ever
# takes a lease, and every wedge for the following ~22h would go uncaught,
# silently: the "no page locked ... exiting" line looks identical whether it
# fires because the run is genuinely over or because it never got a chance to
# start watching.
_DEFAULT_IDLE_POLLS = 12


def _getlk(fd: int, page: int) -> tuple[bool, int]:
    """Whether `page` is currently write-locked, and by which pid.

    The pid is meaningless when the first element is False -- F_GETLK leaves
    only `l_type` defined in that case (see fcntl(2)) -- and callers must not
    use it.
    """
    probe = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET,
                        PAGE_SIZE * page, PAGE_SIZE, 0)
    result = fcntl.fcntl(fd, fcntl.F_GETLK, probe)
    lock_type, _, _, _, pid = struct.unpack(STRUCT_FLOCK, result)
    return lock_type != fcntl.F_UNLCK, pid


def _read_deadline(fd: int, page: int) -> int:
    """The 8-byte deadline at `page`'s base, or 0 if never written / zeroed.

    A plain `os.pread` at a page-aligned offset -- no torn-read handling needed
    for the same reason the writer needs none: 8 bytes at an aligned offset is
    within the platform's atomic write granularity.

    The lock file is never pre-sized (see plugin.py's `_gpu_lease_lockfile`), so
    a page whose worker took the lease but has not yet reached its first test's
    heartbeat write is genuinely short of 8 bytes there, not zero-filled -- a
    plain `os.pread` returns fewer than 8 bytes rather than padding with
    zeroes. Treated the same as an explicit zero: nothing has expired yet.
    """
    raw = os.pread(fd, 8, PAGE_SIZE * page)
    if len(raw) < 8:
        return 0
    return struct.unpack('<Q', raw)[0]


def _send(pid: int, sig: signal.Signals, reason: str) -> None:
    """Signal `pid`, tolerating a process that is already gone.

    The F_GETLK check immediately before every call site is the liveness proof
    (see module docstring); the only race left is the process exiting in the
    interval between that check and this call, which is exactly what
    ProcessLookupError reports -- not a bug to guard against, just the same
    race resolving itself one step later.
    """
    try:
        os.kill(pid, sig)
        print(f'pytest_gpu_lease.watchdog: sent {sig.name} to pid {pid} ({reason})',
              file=sys.stderr, flush=True)
    except ProcessLookupError:
        print(f'pytest_gpu_lease.watchdog: pid {pid} already gone, no {sig.name} needed ({reason})',
              file=sys.stderr, flush=True)


def _poll_once(fd: int, workers: int, threshold_ns: int, grace_ns: int,
              pending: dict[int, int]) -> bool:
    """One sweep of every page. Sends at most one signal -- see module docstring
    on staggering kills -- and returns whether any page is currently locked, for
    the caller's idle/self-exit counter.

    `pending` maps a page already SIGTERMed to the `monotonic_ns()` that signal
    went out, and is mutated in place across calls so escalation survives
    between polls without any state living outside this loop.
    """
    now = time.monotonic_ns()
    any_locked = False

    # Escalations in flight take priority over freshly-discovered ones: a kill
    # already staged should not be starved, poll after poll, by a steady trickle
    # of newly-expired pages elsewhere in the pool.
    for page in list(pending):
        locked, pid = _getlk(fd, page)
        if not locked:
            del pending[page]  # worker exited on its own; nothing left to escalate
            continue
        any_locked = True
        if now - pending[page] >= grace_ns:
            _send(pid, signal.SIGKILL, 'grace period expired, still holding its page')
            del pending[page]
            return any_locked  # one signal per poll

    for page in range(workers):
        if page in pending:
            continue
        locked, pid = _getlk(fd, page)
        if not locked:
            continue
        any_locked = True
        deadline = _read_deadline(fd, page)
        if deadline == 0:
            continue  # between tests -- see plugin.py's gpu_id and pytest_runtest_protocol
        if now > deadline:
            running_s = (now - (deadline - threshold_ns)) / 1e9
            _send(pid, signal.SIGTERM,
                  f'running ~{running_s:.1f}s, past its ~{threshold_ns / 1e9:.0f}s budget')
            pending[page] = now
            return any_locked  # one signal per poll

    return any_locked


def watch(lockfile: str, workers: int, threshold_s: float, grace_s: float,
         poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
         idle_polls: int = _DEFAULT_IDLE_POLLS) -> None:
    """Poll `lockfile` until `idle_polls` consecutive sweeps find no page locked
    -- but only start that countdown once a page has actually been seen locked
    at least once.

    Before that first sighting, `idle` still increments every empty poll (there
    is no reason not to track it), but the loop condition ignores it: a
    watchdog that has never seen a worker take a lease is not stray, it is
    early -- started deliberately ahead of pytest itself, including ahead of
    collection, which for a large suite can run for minutes. Counting idle
    polls from process start would let the watchdog exit on its own before
    collection even finished, and every wedge for the rest of that run would
    then go uncaught. See `_DEFAULT_IDLE_POLLS` for the numbers this matters
    most for.

    Once armed, a worker's page stays locked for the whole of that worker's
    session, so in steady state the idle countdown only ever starts once every
    worker has finished and released its page -- i.e. once the run is actually
    over -- which is the behaviour this fallback exists for in the first place.
    """
    threshold_ns = int(threshold_s * 1_000_000_000)
    grace_ns = int(grace_s * 1_000_000_000)
    pending: dict[int, int] = {}
    idle = 0
    armed = False
    # O_CREAT so the watchdog can be started before pytest ever touches the
    # lock file -- run-test.sh starts it first specifically so a wedge that
    # happens during collection would still be caught. plugin.py's own
    # lockfile fixture is equally permissive for the mirror-image reason.
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while not armed or idle < idle_polls:
            if _poll_once(fd, workers, threshold_ns, grace_ns, pending):
                armed = True
                idle = 0
            else:
                idle += 1
            time.sleep(poll_interval_s)
    finally:
        os.close(fd)
    print(f'pytest_gpu_lease.watchdog: no page locked for {idle_polls} consecutive polls, exiting',
          file=sys.stderr, flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m pytest_gpu_lease.watchdog',
        description='Escalate SIGTERM -> SIGKILL on a pytest_gpu_lease worker whose '
                    'per-test deadline has passed.')
    parser.add_argument('--lockfile', default=os.getenv('GPU_LEASE_LOCKFILE'),
                        help='Shared lease lock file to watch. Defaults to $GPU_LEASE_LOCKFILE, '
                             'the same variable run-test.sh exports for pytest itself.')
    parser.add_argument('--workers', type=int, required=True,
                        help='Size of the GPU pool (the -n given to pytest); pages '
                             '0..workers-1 are watched.')
    parser.add_argument('--threshold', type=float, default=float(_DEFAULT_BUDGET_S),
                        help='Seconds a test may legitimately run (default %(default)s, '
                             'see plugin.py). The actual cutoff is always the deadline the '
                             'worker itself wrote using its own budget (GPU_LEASE_BUDGET_S) '
                             '-- this value never gates that decision, it only sizes the '
                             '"running ~Ns" figure in the log line, so keep the two in sync '
                             'if you change one.')
    parser.add_argument('--grace', type=float, default=_DEFAULT_GRACE_S,
                        help='Seconds to wait after SIGTERM before re-checking the lock '
                             'and escalating to SIGKILL if it is still held.')
    parser.add_argument('--poll_interval', type=float, default=_DEFAULT_POLL_INTERVAL_S,
                        help='Seconds between sweeps of the lock file.')
    parser.add_argument('--idle_polls', type=int, default=_DEFAULT_IDLE_POLLS,
                        help='Consecutive empty polls before exiting on its own, so a stray '
                             'watchdog cannot outlive the run that started it.')
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.lockfile:
        parser.error('--lockfile is required (or set GPU_LEASE_LOCKFILE)')
    watch(args.lockfile, args.workers, args.threshold, args.grace,
         args.poll_interval, args.idle_polls)


if __name__ == '__main__':
    main()
