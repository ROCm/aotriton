# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detect a wedged pytest-xdist worker from outside its process, and kill it.

Run beside pytest, not inside it: ``python -m pytest_gpu_lease.watchdog --lockfile
... --workers N``. It polls the lease lock file that ``plugin.py`` already
maintains -- one 4096-byte page per GPU, write-locked by whichever worker owns
that GPU for the run -- and reads the 8-byte ``time.monotonic_ns()`` stamp
written there: an initial one set by ``gpu_id`` the moment it takes the lease,
refreshed before every test and zeroed after by a ``pytest_runtest_protocol``
hookwrapper. A zero means the worker is between tests and is skipped.

**This module owns the timeout.** The worker reports only when it was last
seen alive; ``--threshold`` here is the sole definition of how stale that may
get, so there is no second copy of the policy to fall out of sync with, and
changing the timeout means changing one command line rather than a command
line and an environment variable that have to agree.

Why the lock file is the right substrate, in one line: ``fcntl(F_GETLK)`` fills
``l_pid`` with the pid holding a byte range, and the kernel releases a record
lock automatically when its owner dies. So the lock answers "does this still
need killing?" on its own: a worker that died between one poll and the next is
simply unlocked by the time we look, and gets nothing sent to it.

It does not answer "who is this?", and pids are not stable identities -- once a
process is reaped its number can be reissued. Every signal therefore goes
through a pidfd (see ``_pidfd_open``), which names one specific process for as
long as the fd is open. The two questions stay separate: the lock decides
whether to signal, the pidfd decides who receives it.

Escalation is SIGTERM, a grace period, then SIGKILL if the page is still locked.
``plugin.py`` registers SIGTERM with ``faulthandler`` on every leasing worker, so
the SIGTERM here is not a death sentence by itself -- it is a request for a
stack dump, C-level and GIL-free, naming whatever frame the wedge is in. Only a
worker that ignores that and is still holding its page after ``--grace`` seconds
gets SIGKILLed.

Shares ``PAGE_SIZE`` and ``STRUCT_FLOCK`` with ``plugin.py`` by importing them
rather than restating them, since a copy that drifts from the writer's layout
would silently misread every stamp.
"""

import argparse
import fcntl
import os
import signal
import struct
import sys
import time
from typing import NamedTuple

from .plugin import PAGE_SIZE, STRUCT_FLOCK

# The timeout, and the only one in the tree: how long a worker's page may go
# without a fresh stamp before it is treated as wedged. Because the worker
# rewrites its stamp before every test, this is exactly one thing -- the
# longest a single test may legitimately take. Not the suite length, not the
# worker count.
#
# A real pass ran 265,282 tests in 227,112 worker-seconds, 0.86s/test mean, so
# 600s is about the tail, not the average -- and the tail is plausibly the
# torch reference materialising an 8192x8192 score matrix, not the kernel
# under test, so sizing this off kernel cost alone would guess far too low.
# The cost of being wrong is asymmetric: too high and a wedge idles a worker
# for the excess (a handful of tests a pass); too low and a slow-but-passing
# test is recorded as a crash, a restart is burned, and the scheduler goes
# through the worker-teardown path again. Refine it with
# `pytest --durations=50 --durations-min=10` on an idle GPU.
_DEFAULT_THRESHOLD_S = 600

# Default cadence: frequent enough that a 600s threshold is caught within a
# small fraction of itself, infrequent enough that polling 4-8 pages is noise
# next to a 15-22h pass.
_DEFAULT_POLL_INTERVAL_S = 5.0

# Default grace between SIGTERM and SIGKILL: long enough for faulthandler to
# flush a stack dump to stderr (a syscall, not instant under load) and for a
# process that was already exiting on its own to finish doing so, short enough
# that a genuinely wedged worker is not left idle for long after being caught.
_DEFAULT_GRACE_S = 30.0



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


def _read_last_activity(fd: int, page: int) -> int:
    """The 8-byte `monotonic_ns()` stamp at `page`'s base, or 0 if the worker
    is between tests (never written, or zeroed after the last one).

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


class _Staged(NamedTuple):
    """A page whose holder has been SIGTERMed and may still need SIGKILL."""

    pid: int
    sent_ns: int
    pidfd: int | None  # None only where pidfd is unavailable; see _pidfd_open


def _pidfd_open(pid: int) -> int | None:
    """A handle to *this* process, or None where that is not available.

    A pid is not a stable identity: the moment its process is reaped the
    number can be handed to somebody else, so every `os.kill(pid, ...)` is a
    bet that nothing has changed since whatever check justified it. A pidfd
    refers to one specific process for as long as the fd is open -- signalling
    through it after that process dies raises ProcessLookupError rather than
    reaching whoever inherited the number.

    None on a kernel older than 5.3 or an interpreter older than 3.9, where
    callers fall back to `os.kill` and the race is merely narrow rather than
    closed.
    """
    try:
        return os.pidfd_open(pid)
    except (AttributeError, OSError):
        return None


def _discard(pending: dict[int, _Staged], page: int) -> None:
    """Drop a staged escalation, closing its pidfd."""
    staged = pending.pop(page, None)
    if staged is not None and staged.pidfd is not None:
        os.close(staged.pidfd)


def _send(pid: int, sig: signal.Signals, reason: str, pidfd: int | None = None) -> None:
    """Signal `pid`, tolerating a process that is already gone.

    The F_GETLK check immediately before every call site is the liveness proof
    (see module docstring); the only race left is the process exiting in the
    interval between that check and this call, which is exactly what
    ProcessLookupError reports -- not a bug to guard against, just the same
    race resolving itself one step later.

    `pid > 0` is checked because `l_pid` is not always a local pid: F_GETLK
    reports -1 for an open-file-description lock, and a lock held over NFS can
    report a pid that means nothing on this host. `os.kill` reads 0 as "the
    whole process group" and -1 as "every process this uid may signal", so a
    single unexpected `l_pid` would take out the pass, the shell that started
    it, and everything else the user owns.
    """
    if pid <= 0:
        print(f'pytest_gpu_lease.watchdog: refusing to signal pid {pid} ({reason}); '
              f'l_pid is not a local pid (OFD lock, or a lock held over NFS)',
              file=sys.stderr, flush=True)
        return
    try:
        if pidfd is None:
            os.kill(pid, sig)
        else:
            signal.pidfd_send_signal(pidfd, sig)
        print(f'pytest_gpu_lease.watchdog: sent {sig.name} to pid {pid} ({reason})',
              file=sys.stderr, flush=True)
    except ProcessLookupError:
        print(f'pytest_gpu_lease.watchdog: pid {pid} already gone, no {sig.name} needed ({reason})',
              file=sys.stderr, flush=True)


def _poll_once(fd: int, workers: int, threshold_ns: int, grace_ns: int,
              pending: dict[int, _Staged]) -> bool:
    """One sweep of every page. Sends at most one signal -- see module docstring
    on staggering kills -- and returns whether any page is currently locked
    (unused by `watch`, which never stops; the tests assert on it).

    `pending` maps a page already SIGTERMed to the `_Staged` record for that
    kill, and is mutated in place across calls so escalation survives between
    polls without any state living outside this loop.
    """
    now = time.monotonic_ns()
    any_locked = False

    # Escalations in flight take priority over freshly-discovered ones: a kill
    # already staged should not be starved, poll after poll, by a steady trickle
    # of newly-expired pages elsewhere in the pool.
    for page in list(pending):
        staged = pending[page]
        locked, pid = _getlk(fd, page)
        # "Still locked" is not enough to escalate: the page has to still be
        # held by the *same* process. A SIGTERMed worker that dies anyway (the
        # HIP runtime aborting on a fault is the common way) is replaced by
        # xdist, and the replacement takes the lowest free page -- very often
        # the one just vacated -- well inside a 30s grace period. Escalating on
        # the page alone would SIGKILL that healthy replacement, then drop the
        # entry and be free to do it again. Dropping the entry here instead
        # hands the page back to the heartbeat scan below, where a replacement
        # is judged on its own stamp like any other worker.
        if not locked or pid != staged.pid:
            _discard(pending, page)
            continue
        any_locked = True
        if now - staged.sent_ns >= grace_ns:
            # Through the pidfd, so this cannot land on anyone else. The lock
            # check above answers the policy question -- is it still wedged and
            # still leasing -- but not the identity one: between that F_GETLK
            # and this call the process may exit and its pid be reissued. That
            # window is small and the payload is SIGKILL, which is precisely
            # the combination not to leave to chance.
            _send(staged.pid, signal.SIGKILL,
                  'grace period expired, still holding its page', staged.pidfd)
            _discard(pending, page)
            return any_locked  # one signal per poll

    for page in range(workers):
        if page in pending:
            continue
        locked, pid = _getlk(fd, page)
        if not locked:
            continue
        any_locked = True
        last_activity = _read_last_activity(fd, page)
        if last_activity == 0:
            continue  # between tests -- see plugin.py's gpu_id and pytest_runtest_protocol
        stale_ns = now - last_activity
        if stale_ns > threshold_ns:
            # Take the handle first, then re-read the lock. If the pid was
            # reissued between the F_GETLK above and the pidfd_open, the
            # original holder is dead, its lock is released, and this second
            # read sees either an unlocked page or a different holder -- so the
            # handle we are about to signal through is the process we measured.
            pidfd = _pidfd_open(pid)
            locked_now, pid_now = _getlk(fd, page)
            if not locked_now or pid_now != pid:
                if pidfd is not None:
                    os.close(pidfd)
                continue
            _send(pid, signal.SIGTERM,
                  f'no heartbeat for {stale_ns / 1e9:.1f}s, past the '
                  f'{threshold_ns / 1e9:.0f}s threshold', pidfd)
            pending[page] = _Staged(pid, now, pidfd)
            return any_locked  # one signal per poll

    return any_locked


def watch(lockfile: str, workers: int, threshold_s: float, grace_s: float,
         poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S) -> None:
    """Poll `lockfile` forever. Returns only when signalled.

    Deliberately has no idea whether a pass is running, finishing, or finished:
    it watches a file, and whoever started it decides when that stops being
    useful (`run-test.sh` does, in a trap). Every rule for inferring "the run
    is over" from the file itself is a guess, and a wrong guess is expensive
    and silent -- the watchdog exits, `main` unlinks the lock file, and the
    rest of a 22h pass runs unprotected with no line in any log saying so.

    An earlier revision guessed from consecutive polls with no page locked.
    That is not the same question: a worker this watchdog just killed is
    replaced, and the replacement holds no lease until it has re-imported torch
    and re-collected -- minutes, with the whole pool in that state at once
    after a multi-worker wedge. The first threshold was 60s and fired during
    exactly that; raising it only moved the guess.

    The cost of never stopping is an orphan if the starting shell is SIGKILLed
    and its trap never runs. That orphan is inert: `fd` is opened once here, so
    once the file is unlinked it polls a deleted inode, and a later run gets a
    new path and a new inode it can never see. No locks observed means no
    pidfds opened and nothing signalled. It shows up in `ps`, and its lock file
    is left behind in /dev/shm.
    """
    threshold_ns = int(threshold_s * 1_000_000_000)
    grace_ns = int(grace_s * 1_000_000_000)
    pending: dict[int, _Staged] = {}
    # O_CREAT so the watchdog can be started before pytest ever touches the
    # lock file: run-test.sh starts it first so that it is already watching by
    # the time the first worker takes a lease. plugin.py's own lockfile fixture
    # is equally permissive for the mirror-image reason.
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        while True:
            _poll_once(fd, workers, threshold_ns, grace_ns, pending)
            time.sleep(poll_interval_s)
    finally:
        for page in list(pending):
            _discard(pending, page)  # close any pidfd still staged
        os.close(fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m pytest_gpu_lease.watchdog',
        description='Escalate SIGTERM -> SIGKILL on a pytest_gpu_lease worker whose '
                    'page has gone unstamped for too long.')
    parser.add_argument('--lockfile', required=True,
                        help='Shared lease lock file to watch. Required, rather than '
                             'defaulted from $GPU_LEASE_LOCKFILE, so that `ps` shows which '
                             'file each watchdog is on -- that is the only way to tell a '
                             'stale watchdog from a live one. Workers still find the same '
                             'file through the environment variable; only this process '
                             'insists on being told.')
    parser.add_argument('--workers', type=int, required=True,
                        help='Size of the GPU pool (the -n given to pytest); pages '
                             '0..workers-1 are watched.')
    parser.add_argument('--threshold', type=float, default=float(_DEFAULT_THRESHOLD_S),
                        help='Seconds a test may legitimately run before its worker is '
                             'treated as wedged (default %(default)s). This is the timeout: '
                             'workers only report when they were last alive, so nothing '
                             'else in the system has an opinion about it.')
    parser.add_argument('--grace', type=float, default=_DEFAULT_GRACE_S,
                        help='Seconds to wait after SIGTERM before re-checking the lock '
                             'and escalating to SIGKILL if it is still held.')
    parser.add_argument('--poll_interval', type=float, default=_DEFAULT_POLL_INTERVAL_S,
                        help='Seconds between sweeps of the lock file.')
    return parser


def _exit_on_signal(signum, _frame):
    """Turn a termination signal into a normal unwind.

    SIGTERM and SIGHUP would otherwise kill the process outright, skipping the
    `finally` in `main` that removes the lock file. SIGINT already unwinds, and
    is handled here only so all three exit the same way.
    """
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    try:
        watch(args.lockfile, args.workers, args.threshold, args.grace,
             args.poll_interval)
    finally:
        # The watchdog owns the lock file, so whoever started it needs no
        # cleanup of its own -- which matters because an untrapped SIGTERM or
        # SIGHUP skips a starting script's EXIT trap entirely (measured).
        # SIGKILL is the gap: nothing runs here, and the file is left behind.
        #
        # `watch()` never returns on its own, so reaching this means we were
        # told to stop and the pass is over -- the file cannot be pulled out
        # from under a running one. Deliberately not inside `watch()`, which
        # stays a pure poll loop callable from tests against a file they own.
        try:
            os.unlink(args.lockfile)
        except FileNotFoundError:
            pass
        print('pytest_gpu_lease.watchdog: stopped', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
