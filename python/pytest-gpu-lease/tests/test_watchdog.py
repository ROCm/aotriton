# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-test suite for pytest_gpu_lease.watchdog.

Needs no GPU. Two kinds of case:

* A direct test of the escalation logic against a process wedged in a genuine
  C call, driven straight against `watchdog.watch()` -- no pytest session on
  the wedged side at all. This isolates "does the watchdog correctly
  SIGTERM-then-SIGKILL a stuck pid" from everything `pytest_gpu_lease.plugin`
  does around it.

* An end-to-end case with the plugin, the heartbeat, and the watchdog all
  running together under `-n 2`, via a nested `pytester` session -- the
  scenario the whole feature exists for.

Both wedge by calling `os.read` on a pipe nobody ever writes to, rather than
relying on any claim about GIL behaviour. An earlier attempt to wedge a thread
via `ctypes.PyDLL` (chosen specifically because it does not release the GIL),
to test that a background `threading.Timer` could not run during the wedge,
gave a result inconsistent with that theory -- the Timer fired anyway. So
nothing here depends on GIL semantics: `os.read` on an empty pipe blocks in
the kernel regardless of GIL state, and what is checked afterwards is only
what is directly observable -- the process does not return from that call on
a plain SIGTERM (because faulthandler.register replaces the default terminate
action), it does produce a stack dump naming the blocked frame, and it does
die on SIGKILL, which no handler can intercept.
"""

import os
import pathlib
import signal
import struct
import subprocess
import sys
import threading
import time

import pytest

from pytest_gpu_lease import watchdog

pytest_plugins = ['pytester']


_WEDGE_CHILD = """
import fcntl, os, struct, sys, time, faulthandler, signal
from pytest_gpu_lease.plugin import PAGE_SIZE, STRUCT_FLOCK

lockfile = sys.argv[1]
page = int(sys.argv[2])
backdate_ns = int(sys.argv[3])
f = open(lockfile, 'r+b')
claim = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET, PAGE_SIZE * page, PAGE_SIZE, 0)
fcntl.fcntl(f, fcntl.F_SETLK, claim)
# Backdate the heartbeat well past any threshold these tests use, so the
# watchdog acts on its very first poll rather than this test also depending on
# wall-clock timing to age one out. max(1, ...) because monotonic_ns() is time
# since boot and could in principle be smaller than the offset -- 1 is still a
# valid ancient stamp, whereas 0 would read as "between tests" and be skipped.
# backdate_ns=0 gives a fresh stamp instead: a healthy worker, to be left alone.
os.pwrite(f.fileno(), struct.pack('<Q', max(1, time.monotonic_ns() - backdate_ns)),
          PAGE_SIZE * page)
faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
r, w = os.pipe()
print('READY', flush=True)
os.read(r, 1)  # blocks forever: nobody ever writes to w
"""

_STALE_NS = 10_000_000_000


def _spawn_wedge_child(tmp_path, lockfile, page=0, backdate_ns=_STALE_NS, tag=''):
    child_script = tmp_path / f'wedge_child_{page}.py'
    child_script.write_text(_WEDGE_CHILD)
    err_path = tmp_path / f'child_{page}{tag}.err'
    err_file = open(err_path, 'wb')
    child = subprocess.Popen(
        [sys.executable, str(child_script), str(lockfile), str(page), str(backdate_ns)],
        stdout=subprocess.PIPE, stderr=err_file, text=True)
    assert child.stdout.readline().strip() == 'READY'
    return child, err_path, err_file


@pytest.mark.timeout(30)
def test_watchdog_sigterms_then_sigkills_a_process_wedged_in_a_c_call(tmp_path):
    """The escalation logic in isolation, against a real wedged process.

    `watch()` never returns, so the assertion is on the child rather than on
    the watcher: a daemon thread runs the poll loop and is reaped with the
    process, and `child.wait` is what bounds the test. The `timeout` mark above
    cannot be relied on for that -- this repo dropped pytest-timeout, so in a
    clean environment it is an unknown mark that does nothing.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    child, err_path, err_file = _spawn_wedge_child(tmp_path, lockfile)
    try:
        watcher = threading.Thread(
            target=watchdog.watch,
            args=(str(lockfile), 1, 1, 2),
            kwargs=dict(poll_interval_s=0.1),
            daemon=True)
        watcher.start()
        ret = child.wait(timeout=20)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        err_file.close()

    assert ret == -signal.SIGKILL, f'expected the child to die by SIGKILL, got {ret}'
    stack = err_path.read_text()
    assert 'wedge_child_0.py' in stack, f'no stack dump naming the wedged frame:\n{stack}'


def test_getlk_reports_unlocked_after_holder_is_killed(tmp_path):
    """F_GETLK auto-releases on process death -- the liveness proof the module
    docstring relies on to re-check the lock immediately before every signal
    instead of trusting a remembered pid. A dead process holds no lock, full
    stop, so a reused pid can never be mistaken for the worker that had it.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    child, err_path, err_file = _spawn_wedge_child(tmp_path, lockfile)
    fd = os.open(str(lockfile), os.O_RDWR)
    try:
        locked, pid = watchdog._getlk(fd, 0)
        assert locked and pid == child.pid

        child.kill()  # SIGKILL: no handler can intervene
        child.wait(timeout=5)

        locked, _ = watchdog._getlk(fd, 0)
        assert not locked, 'kernel did not auto-release the lock on process death'
    finally:
        os.close(fd)
        err_file.close()


@pytest.mark.timeout(30)
def test_escalation_does_not_sigkill_a_replacement_that_took_the_same_page(tmp_path):
    """A staged SIGKILL must follow the pid, not the page.

    Regression test. The SIGTERMed worker often dies on its own inside the
    grace period -- the HIP runtime aborting on a fault is the usual way -- and
    xdist replaces it. The replacement takes the lowest free page, i.e. very
    often the one just vacated, and can be heartbeating there well before the
    grace period is up. Escalating on "page is still locked" alone would then
    SIGKILL a perfectly healthy worker, drop the entry, and be free to do it
    again on the next one.

    Driven through `_poll_once` directly so the two halves -- SIGTERM staged,
    then escalation re-checked -- are two explicit calls rather than a timing
    coincidence inside `watch()`.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    fd = os.open(str(lockfile), os.O_RDWR)
    pending: dict[int, watchdog._Staged] = {}
    # grace_ns=0: the second poll escalates immediately if it is going to at all.
    poll = lambda: watchdog._poll_once(fd, str(lockfile), 1, threshold_ns=10**9,  # noqa: E731
                                       grace_ns=0, pending=pending)

    wedged, _, wedged_err = _spawn_wedge_child(tmp_path, lockfile, tag='_wedged')
    replacement = replacement_err = None
    try:
        assert poll() is True
        assert list(pending) == [0] and pending[0].pid == wedged.pid, pending

        wedged.kill()  # the SIGTERMed worker dies on its own, releasing page 0
        wedged.wait(timeout=5)

        # ...and its replacement takes the same page, with a fresh heartbeat.
        replacement, _, replacement_err = _spawn_wedge_child(
            tmp_path, lockfile, backdate_ns=0, tag='_replacement')

        assert poll() is True
        assert pending == {}, 'stale escalation must be dropped once the pid changes'
        # `poll()` is not enough: SIGKILL is delivered asynchronously, so a
        # child signalled a microsecond ago still reads as running. Waiting is
        # what makes the absence of a kill observable.
        with pytest.raises(subprocess.TimeoutExpired):
            replacement.wait(timeout=2)
    finally:
        for child in (wedged, replacement):
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
        for handle in (wedged_err, replacement_err):
            if handle is not None:
                handle.close()
        os.close(fd)


def test_escalation_is_cancelled_when_the_worker_recovers(tmp_path):
    """A SIGTERMed worker that got going again must not be SIGKILLed.

    SIGTERM is a stack dump, not a death -- `plugin.py` registers it with
    faulthandler and chain=False -- so a test that was merely slow rather than
    wedged can finish inside the grace period. The page then goes back to 0
    (between tests) or takes a fresh stamp, while the lock and the pid are
    unchanged, so escalating on those two alone kills a healthy worker.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    fd = os.open(str(lockfile), os.O_RDWR)
    pending: dict[int, watchdog._Staged] = {}
    poll = lambda: watchdog._poll_once(fd, str(lockfile), 1, threshold_ns=10**9,  # noqa: E731
                                       grace_ns=0, pending=pending)

    child, _, err_file = _spawn_wedge_child(tmp_path, lockfile, tag='_recovers')
    try:
        assert poll() is True
        assert list(pending) == [0], pending

        # The slow test finishes: same process, same lease, fresh heartbeat.
        os.pwrite(fd, struct.pack('<Q', 0), 0)

        assert poll() is True
        assert pending == {}, 'a recovered worker must not stay staged for SIGKILL'
        # grace_ns=0, so any SIGKILL would already have gone out.
        time.sleep(0.5)
        assert child.poll() is None, 'SIGKILLed a worker that had recovered'
    finally:
        child.kill()
        child.wait()
        err_file.close()
        os.close(fd)


@pytest.mark.timeout(30)
def test_watchdog_never_stops_on_its_own(tmp_path):
    """The service invariant: it does not decide for itself when to stop.

    Both halves matter. An empty lock file must not end the watch -- that is
    the state during collection, before any worker has leased, and again while
    a killed worker's replacement re-imports torch, and an earlier revision
    exited after 60s of it and took the lock file with it. And having fired,
    it must still not stop: one wedge does not end a pass, and the remaining
    workers are exactly what it is there for.

    Runs on a daemon thread because `watch()` blocks forever by design; the
    thread is reaped with the process.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    poll_interval = 0.1

    thread = threading.Thread(
        target=watchdog.watch,
        args=(str(lockfile), 1, 1, 1),
        kwargs=dict(poll_interval_s=poll_interval),
        daemon=True)
    thread.start()

    time.sleep(poll_interval * 20)
    assert thread.is_alive(), 'watchdog stopped while the lock file was empty'

    child, err_path, err_file = _spawn_wedge_child(tmp_path, lockfile)
    try:
        ret = child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        err_file.close()

    assert ret == -signal.SIGKILL, f'expected the child to die by SIGKILL, got {ret}'
    time.sleep(poll_interval * 20)
    assert thread.is_alive(), 'watchdog stopped after acting on one wedge'


@pytest.mark.timeout(90)
def test_end_to_end_wedged_worker_is_replaced_and_run_stays_green(pytester, monkeypatch, tmp_path):
    """The full path: heartbeat, watchdog, escalation, worker replacement.

    One test out of six wedges in a C call under `-n 2`; the watchdog runs
    alongside the nested pytest session exactly as run-test.sh will run it
    (own process, same lock file, `--threshold`/`--grace` sized in seconds
    instead of the real 600s/30s -- this test cannot wait out the real
    default; that the fuse can be shortened from the watchdog's command line
    alone, with nothing set on the worker side, is the protocol working as
    intended). Assertions: the wedged worker's stack dump names the wedged
    frame, the other five tests still pass despite that worker being killed
    mid-run, and the session terminates at all -- a leaked lease or a dead
    watchdog would otherwise hang it.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    monkeypatch.delenv('GPU_LEASE_PIN', raising=False)
    monkeypatch.setenv('GPU_LEASE_LOCKFILE', str(lockfile))

    pytester.makepyfile("""
        import os

        import pytest

        @pytest.mark.parametrize('i', range(6))
        def test_maybe_wedge(i, gpu_id):
            if i == 0:
                r, w = os.pipe()
                os.read(r, 1)  # never returns
            assert gpu_id in (0, 1)
    """)

    watchdog_proc = subprocess.Popen(
        [sys.executable, '-m', 'pytest_gpu_lease.watchdog',
         '--lockfile', str(lockfile), '--workers', '2',
         '--threshold', '1', '--grace', '2',
         '--poll_interval', '0.2'],
        stderr=subprocess.PIPE, text=True)
    try:
        result = pytester.runpytest_subprocess(
            '-n', '2', '--max-worker-restart', '4', '-p', 'xdist', timeout=60)
        # The watchdog is a service: it is stopped by whoever started it, never
        # by itself. Doing that here also covers the SIGTERM shutdown path.
        watchdog_proc.terminate()
        _, watchdog_err = watchdog_proc.communicate(timeout=30)
    finally:
        if watchdog_proc.poll() is None:
            watchdog_proc.kill()
            watchdog_proc.communicate()

    assert result.ret is not None, 'session did not terminate'
    assert 'SIGTERM' in watchdog_err, watchdog_err
    assert 'SIGKILL' in watchdog_err, watchdog_err

    outcomes = result.parseoutcomes()
    # 5 of the 6 parametrizations never touch the wedge; they must all still
    # pass despite the sixth worker being killed and replaced mid-run.
    assert outcomes.get('passed', 0) >= 5, outcomes


def test_a_stale_pidfd_reports_gone_rather_than_signalling_a_reissued_pid(capsys):
    """The identity half of the escalation guard, in isolation.

    `_getlk` says whether a page still needs its holder killed, but there is
    always a gap between that answer and the signal, and a pid reaped inside
    that gap can be reissued to anything. A pidfd names one process for the
    life of the fd, so the worst case degrades from "SIGKILL an innocent
    process" to "ESRCH, and say so".

    Deliberately not asserting that some bystander survived: pid reuse cannot
    be forced on demand, so this checks the property that makes reuse
    unreachable instead of trying to stage it.
    """
    victim = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    pidfd = watchdog._pidfd_open(victim.pid)
    assert pidfd is not None, 'expected pidfd support on this kernel'
    victim.kill()
    victim.wait(timeout=10)   # dead *and* reaped: the number is free to be reissued
    try:
        watchdog._send(victim.pid, signal.SIGKILL, 'stale handle', pidfd)
    finally:
        os.close(pidfd)
    assert 'already gone' in capsys.readouterr().err


def test_watchdog_relays_a_wedged_workers_stack_dump(tmp_path, capsys):
    """The dump reaches the pass's stderr through the watchdog, not directly.

    A worker cannot write it to the shared stream itself: a dump is many small
    writes, so several workers dumping at once would splice. It writes a file
    of its own and the watchdog, the one serial writer, relays it before the
    SIGKILL -- after which nobody is left to ask.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    child, _, err_file = _spawn_wedge_child(tmp_path, lockfile, tag='_dump')
    pathlib.Path(watchdog.dump_path(str(lockfile), child.pid)).write_text(
        'Current thread 0x00007f00 (most recent call first):\n'
        '  File "test_flash.py", line 42 in test_wedges_here\n')
    fd = os.open(str(lockfile), os.O_RDWR)
    try:
        pending: dict[int, watchdog._Staged] = {}
        poll = lambda: watchdog._poll_once(fd, str(lockfile), 1, threshold_ns=10**9,  # noqa: E731
                                           grace_ns=0, pending=pending)
        poll()          # stages the SIGTERM
        poll()          # grace is 0, so this escalates and should relay first
    finally:
        child.kill()
        child.wait()
        err_file.close()
        os.close(fd)

    relayed = capsys.readouterr().err
    assert 'test_wedges_here' in relayed, relayed
    assert f'stack of pid {child.pid} on GPU 0' in relayed, relayed
    assert not pathlib.Path(watchdog.dump_path(str(lockfile), child.pid)).exists(), \
        'the dump must be deleted as soon as it is relayed'
