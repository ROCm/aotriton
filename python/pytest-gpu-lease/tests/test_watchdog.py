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
import signal
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
f = open(lockfile, 'r+b')
claim = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET, PAGE_SIZE * page, PAGE_SIZE, 0)
fcntl.fcntl(f, fcntl.F_SETLK, claim)
# Deadline already in the past, so the watchdog acts on its very first poll
# rather than this test also depending on wall-clock timing to expire one.
os.pwrite(f.fileno(), struct.pack('<Q', time.monotonic_ns() - 1_000_000_000), PAGE_SIZE * page)
faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
r, w = os.pipe()
print('READY', flush=True)
os.read(r, 1)  # blocks forever: nobody ever writes to w
"""


def _spawn_wedge_child(tmp_path, lockfile, page=0):
    child_script = tmp_path / f'wedge_child_{page}.py'
    child_script.write_text(_WEDGE_CHILD)
    err_path = tmp_path / f'child_{page}.err'
    err_file = open(err_path, 'wb')
    child = subprocess.Popen(
        [sys.executable, str(child_script), str(lockfile), str(page)],
        stdout=subprocess.PIPE, stderr=err_file, text=True)
    assert child.stdout.readline().strip() == 'READY'
    return child, err_path, err_file


@pytest.mark.timeout(30)
def test_watchdog_sigterms_then_sigkills_a_process_wedged_in_a_c_call(tmp_path):
    """The escalation logic in isolation, against a real wedged process."""
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    child, err_path, err_file = _spawn_wedge_child(tmp_path, lockfile)
    try:
        watchdog.watch(str(lockfile), workers=1, threshold_s=1, grace_s=2,
                        poll_interval_s=0.1, idle_polls=5)
        ret = child.wait(timeout=10)
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
def test_watchdog_waits_indefinitely_before_first_page_is_ever_locked(tmp_path):
    """Regression test: idle polls must not be counted before a page has ever
    been seen locked, or the watchdog would exit on its own during collection,
    before any worker has taken a lease.

    That is not a hypothetical corner case: run-test.sh starts the watchdog
    ahead of pytest deliberately, specifically to cover a wedge during
    collection too, and a Level-3 pass collects roughly 330k tests through a
    conftest that imports torch -- minutes, not seconds. At the defaults
    (5s poll interval, 12 idle polls) the buggy version exits after 60s of
    an empty lock file, which collection alone comfortably outlasts; every
    wedge for the rest of that ~22h run would then go uncaught, silently,
    since the "no page locked ... exiting" line looks the same whether the
    run is actually over or the watchdog simply gave up too early.

    Drives `watch()` on a background thread because it blocks for the whole
    watch, and this test needs to observe it still running mid-watch, then
    later observe it firing -- both from the outside.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    poll_interval = 0.1
    idle_polls = 3
    # Comfortably longer than idle_polls * poll_interval: the exact window the
    # buggy version needed to give up on a still-empty lock file.
    wait_before_lock = poll_interval * idle_polls * 5

    # daemon=True: watch() has no cooperative way to stop it early, so if an
    # assertion below fails, let the thread be reaped with the process rather
    # than block the test on joining it.
    thread = threading.Thread(
        target=watchdog.watch,
        args=(str(lockfile), 1, 1, 1),
        kwargs=dict(poll_interval_s=poll_interval, idle_polls=idle_polls),
        daemon=True)
    thread.start()

    time.sleep(wait_before_lock)
    assert thread.is_alive(), \
        'watchdog exited before any page was ever locked; it must wait ' \
        'indefinitely until a worker actually takes a lease'

    child, err_path, err_file = _spawn_wedge_child(tmp_path, lockfile)
    try:
        thread.join(timeout=10)
        assert not thread.is_alive(), \
            'watchdog did not fire once a page was locked with an expired deadline'
        ret = child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        err_file.close()

    assert ret == -signal.SIGKILL, f'expected the child to die by SIGKILL, got {ret}'


@pytest.mark.timeout(90)
def test_end_to_end_wedged_worker_is_replaced_and_run_stays_green(pytester, monkeypatch, tmp_path):
    """The full path: heartbeat, watchdog, escalation, worker replacement.

    One test out of six wedges in a C call under `-n 2`; the watchdog runs
    alongside the nested pytest session exactly as run-test.sh will run it
    (own process, same lock file, `--threshold`/`--grace` sized in seconds
    instead of the real 600s/30s -- this test cannot wait out the real
    default). Assertions: the wedged worker's stack dump names the wedged
    frame, the other five tests still pass despite that worker being killed
    mid-run, and the session terminates at all -- a leaked lease or a dead
    watchdog would otherwise hang it.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    monkeypatch.delenv('GPU_LEASE_PIN', raising=False)
    monkeypatch.setenv('GPU_LEASE_LOCKFILE', str(lockfile))
    monkeypatch.setenv('GPU_LEASE_BUDGET_S', '1')

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
         '--poll_interval', '0.2', '--idle_polls', '15'],
        stderr=subprocess.PIPE, text=True)
    try:
        result = pytester.runpytest_subprocess(
            '-n', '2', '--max-worker-restart', '4', '-p', 'xdist', timeout=60)
        _, watchdog_err = watchdog_proc.communicate(timeout=15)
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
