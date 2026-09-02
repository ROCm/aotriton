# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lease exactly one GPU to each pytest-xdist worker.

Workers of a single pytest run coordinate through POSIX record locks on one shared
file: GPU *n* is represented by the byte range [n*PAGE_SIZE, (n+1)*PAGE_SIZE). A
worker walks the range round-robin and takes the first page it can write-lock; it
holds that lock for its whole session and releases it at teardown.

Three modes, selected by environment:

* ``GPU_LEASE_PIN=<n>`` -- every worker pinned to GPU n, no locking. Checked
  first, so it works with or without xdist. For single-GPU reruns and for
  bisecting a failure onto a known-good device.
* not running under xdist (no ``config.workerinput``) -- GPU 0, no locking.
* otherwise -- lease as described above, one page per worker.

The mapping is deliberately 1:1 worker-to-GPU with no oversubscription knob: running
several tests concurrently on one GPU invites memory pressure and runtime / driver /
firmware / VBIOS races.
"""

import fcntl
import faulthandler
import itertools
import os
import signal
import struct
import sys
import time
from pathlib import Path
from types import MappingProxyType

import pytest

STRUCT_FLOCK = 'hhqqi'
PAGE_SIZE = 4096
_RETRY_INTERVAL = 0.05

# How long a single test may legitimately run before the watchdog (see
# watchdog.py, which imports this) treats the worker holding it as wedged.
# Not the suite length, not the worker count -- the heartbeat below is
# rewritten before every test, so this number is exactly one thing: the
# longest one test may take. A real pass ran 265,282 tests in 227,112
# worker-seconds, 0.86s/test mean, so 600s is about the tail, not the
# average -- and the tail is plausibly the torch reference materialising an
# 8192x8192 score matrix, not the kernel under test, so sizing it off kernel
# cost alone would guess far too low. The cost of being wrong is asymmetric:
# too high and a wedge idles a worker for the excess (a handful of tests a
# pass); too low and a slow-but-passing test is recorded as a crash, a
# restart is burned, and the scheduler goes through the teardown path again.
# ``GPU_LEASE_BUDGET_S`` overrides it, mainly so a test of the watchdog
# itself can use a fuse of seconds rather than waiting out the real one.
_DEFAULT_BUDGET_S = 600


def _budget_ns() -> int:
    """Per-test heartbeat budget in nanoseconds. Read lazily, never at import."""
    raw = os.getenv('GPU_LEASE_BUDGET_S', default=None)
    seconds = _DEFAULT_BUDGET_S if raw is None else float(raw)
    return int(seconds * 1_000_000_000)


# Set by the leased branch of `gpu_id` once a lease is actually held, and
# cleared again at its teardown; `None` the rest of the time. The
# `pytest_runtest_call` hookwrapper below needs to know whether there is
# a page to heartbeat, but it is a plugin-level hook, not a fixture, so it
# cannot request `gpu_id` or read its generator frame -- this module global
# is the connecting state. A plain global is safe here because `gpu_id` is
# session-scoped and, under xdist, "session" means "per worker process": at
# most one lease is ever active in a given interpreter.
_active_lease: tuple[int, int] | None = None  # (fd, page_base)


# Stand-in for a process that is not an xdist worker: the controller, a plain
# `pytest` run, or xdist not loaded at all. All three are the same thing here, so
# one set of defaults covers them and callers never test for absence.
#
# 'master' is the label xdist's own worker_id fixture reports off-worker. A count
# of 0 is an unambiguous sentinel for "not distributed": a real worker always sees
# at least 1, and `-n 0` creates no workers and hence no workerinput.
_NO_XDIST = MappingProxyType({'workerid': 'master', 'workercount': 0})


def _workerinput(config):
    """xdist's per-worker payload, with off-worker defaults filled in.

    Always returns a mapping. Two keys matter, and both are deliberately sourced
    from here rather than from the more obvious places:

    ``workercount`` -- the size of the GPU pool. NOT
    ``PYTEST_XDIST_WORKER_COUNT``: xdist sets that inside the worker process, but
    this module is a ``pytest11`` entry-point plugin imported during
    ``Config._preparse``, long before. Read at module scope it saw 0 and silently
    put every worker on GPU 0. The value here is fixed at the original ``-n`` for
    the whole run -- a crashed worker is replaced from its own spec, so the pool
    never resizes and the page it freed stays in range for its replacement.

    ``workerid`` -- the label used in announcements. NOT xdist's ``worker_id``
    fixture: depending on it made ``gpu_id`` unresolvable whenever xdist was not
    loaded (``-p no:xdist``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD``), including on the
    pinned and no-xdist paths, which need nothing from xdist at all.
    """
    workerinput = getattr(config, 'workerinput', None)
    return _NO_XDIST if workerinput is None else workerinput


def _env_pin() -> int | None:
    """``GPU_LEASE_PIN`` as an int, or None. Read lazily, never at import."""
    raw = os.getenv('GPU_LEASE_PIN', default=None)
    return None if raw is None else int(raw)


def _announce(config, message: str) -> None:
    """Write `message` to stderr *now*, bypassing pytest's output capture.

    The lease is decided during fixture setup, and pytest captures setup output at
    the fd level, replaying it only in the report section -- and only for failing
    tests, unless ``-rA`` is given. On a green run the GPU assignment would never
    be shown until the run was over, which is the whole point of announcing it.

    ``capsys.disabled()`` is the documented way to suspend capture, but every
    capture fixture is function-scoped while ``gpu_id`` is session-scoped, so
    requesting one here would raise ``ScopeMismatch``. We therefore call the
    capture manager that ``capsys.disabled()`` itself delegates to.

    That is ``_pytest.capture`` internals, not public API. If a future pytest
    reorganises it, the lookup below degrades to a plain print rather than
    breaking the run -- the symptom is the line reappearing only in the replayed
    "Captured stderr setup" section, which is the cue to pin pytest and revisit.
    """
    capman = config.pluginmanager.getplugin('capturemanager')
    disabled = getattr(capman, 'global_and_fixture_disabled', None)
    if disabled is None:  # -p no:capture, or the internals moved
        print(message, file=sys.stderr, flush=True)
        return
    with disabled():
        print(message, file=sys.stderr, flush=True)


@pytest.fixture(scope='session')
def _gpu_lease_lockfile(tmp_path_factory):
    """Path to the run-wide lock file, created if absent.

    NOT autouse: this plugin auto-loads into every pytest run in the environment,
    including GPU-less suites (python/test, modules/flash/tests/test_gpu_targets.py),
    which must not touch the filesystem.

    The file is never sized or truncated. POSIX record locks may be placed beyond EOF,
    so pre-sizing buys nothing -- and the old open(..., 'wb') let a late-starting worker
    truncate a file its peers were already locking.

    ``GPU_LEASE_LOCKFILE``, when set, wins over the derived path. The watchdog is a
    separate process started before pytest exists, so it cannot predict what
    ``getbasetemp()`` will resolve to for this run -- run-test.sh instead mints a
    path up front and both sides read it from the environment. The derived path
    stays as the fallback so a plain ``pytest`` invocation, with no watchdog and
    no env var, is unchanged. Side benefit: the lease file becomes greppable by a
    fixed name during a hang instead of living inside a per-run ``pytest-NNN/`` dir.
    """
    override = os.getenv('GPU_LEASE_LOCKFILE', default=None)
    if override is not None:
        lockfile = Path(override)
    else:
        # getbasetemp().parent is shared by all workers of the run; getbasetemp() is per-worker.
        lockfile = tmp_path_factory.getbasetemp().parent / 'gpulock'
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
    os.close(fd)
    return lockfile


@pytest.fixture(scope='session')  # under xdist, "session" scope is per-worker process
def gpu_id(request):
    """Index of the GPU this worker owns for the duration of its session.

    Every mode announces its choice, not just the leasing one: without it there is
    no way to confirm that GPU_LEASE_PIN actually took effect either.

    Depends on no xdist fixture, so it resolves even under ``-p no:xdist``.
    """
    workerinput = _workerinput(request.config)
    worker_id = workerinput['workerid']

    # GPU_LEASE_PIN wins over everything, distributed or not: "put all work on
    # GPU n" is a debugging override and should not depend on how pytest is run.
    pinned = _env_pin()
    if pinned is not None:
        _announce(request.config, f'{worker_id} uses GPU {pinned} (GPU_LEASE_PIN, no lease)')
        yield pinned
        return

    nworkers = int(workerinput['workercount'])
    if nworkers == 0:
        _announce(request.config, f'{worker_id} uses GPU 0 (no xdist, no lease)')
        yield 0
        return

    # Resolved lazily, NOT as a fixture parameter: pytest instantiates declared
    # params before the body runs, so naming _gpu_lease_lockfile in the signature
    # would create the file in the no-xdist and pinned modes too -- the very
    # side effect dropping `autouse` was meant to prevent.
    lockfile = request.getfixturevalue('_gpu_lease_lockfile')
    global _active_lease
    with open(lockfile, 'r+b') as f:
        for gpu in itertools.cycle(range(nworkers)):
            page_base = PAGE_SIZE * gpu
            claim = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET,
                                page_base, PAGE_SIZE, 0)
            try:
                fcntl.fcntl(f, fcntl.F_SETLK, claim)
            except BlockingIOError:
                # Every page is taken for the moment. Sleep instead of spinning --
                # the original loop pegged a core while waiting.
                if gpu == nworkers - 1:
                    time.sleep(_RETRY_INTERVAL)
                continue
            _announce(request.config,
                      f'{worker_id} uses GPU {gpu} filelock = {lockfile}')
            # A Python-level SIGTERM handler would fail the same way pytest-timeout's
            # SIGALRM handler does: CPython only runs a Python signal handler when the
            # main thread reaches a bytecode boundary, and a thread parked in a HIP
            # call never reaches one. faulthandler's handler is C-level and needs no
            # GIL, so it still fires -- and dumps every thread's stack -- while the
            # main thread is stuck. Verified: a process blocked in a ctypes.PyDLL
            # call, sent SIGTERM, printed a full stack dump naming the frame.
            faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True)
            _active_lease = (f.fileno(), page_base)
            try:
                yield gpu
            finally:
                _active_lease = None
                release = struct.pack(STRUCT_FLOCK, fcntl.F_UNLCK, os.SEEK_SET,
                                      page_base, PAGE_SIZE, 0)
                fcntl.fcntl(f, fcntl.F_SETLK, release)
            return


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call():
    """Heartbeat this worker's page with the current test's deadline.

    Inert unless `gpu_id` actually holds a lease: `_active_lease` is only set by
    the leased branch above, so pinned mode, no-xdist mode, and GPU-less suites
    (which never request `gpu_id`, so that fixture body never runs at all) hit
    the `is None` check and do no I/O -- this hook fires for every test in every
    pytest process in the environment, so being a no-op absent a lease is not
    optional.

    Wraps `pytest_runtest_call`, NOT `pytest_runtest_protocol`, and that choice
    is load-bearing, not cosmetic. `runtestprotocol()` runs setup, call and
    teardown as three separate hook calls in sequence -- `pytest_runtest_setup`,
    then this one, then `pytest_runtest_teardown` -- and `gpu_id`'s lock-acquire
    (which sets `_active_lease`) happens during fixture setup, i.e. inside the
    *first* of those. A hookwrapper on `pytest_runtest_protocol` instead wraps
    all three phases as one call, so its pre-yield code runs *before* setup has
    resolved any fixtures at all: on a worker's very first test, `_active_lease`
    is still `None` at that point regardless of whether a lease is about to be
    taken, and the first test's heartbeat -- often the one most likely to be hit
    by an early wedge -- silently never gets written. (Caught by an end-to-end
    test that wedged the first parametrized case on a worker: the watchdog never
    fired, because the deadline it was polling for had never been written.)
    Wrapping `pytest_runtest_call` instead means setup has already completed
    -- and `_active_lease` is already current -- every time this hook's
    pre-yield code runs, first test included. The same reasoning rules out
    `pytest_runtest_teardown` for the zeroing half below: it is a still later,
    separate phase, so zeroing there would race the *next* test's setup instead.

    Runs on whatever thread pytest calls hooks on for this test, which under
    xdist is the worker's MainThread -- deliberately: the heartbeat must stop
    the instant that thread stops reaching this hook, which is exactly what
    happens when it wedges in a HIP call. A background heartbeat thread would
    keep ticking regardless and defeat the whole scheme.

    The write is a single aligned 8-byte pwrite at the page base -- the
    platform's atomic write granularity for a page-aligned offset -- so the
    watchdog's concurrent pread never observes a torn value; no seqlock or
    other synchronisation is needed on either side. Zeroing the deadline after
    the test, rather than leaving the last one in place, is what keeps an idle
    worker between tests from ever looking like a candidate to the watchdog.
    """
    lease = _active_lease
    if lease is None:
        yield
        return
    fd, page_base = lease
    deadline = time.monotonic_ns() + _budget_ns()
    os.pwrite(fd, struct.pack('<Q', deadline), page_base)
    try:
        yield
    finally:
        # `gpu_id`'s teardown (closing `fd`) happens in the later, separate
        # `pytest_runtest_teardown` phase, so `fd` is still the live lease fd
        # here even on a worker's last test -- unlike the now-abandoned
        # `pytest_runtest_protocol` wrapping, where that teardown ran inside
        # this same wrapped call. The `except OSError` stays anyway: it costs
        # nothing to tolerate an fd that turns out to be closed, and pinning
        # correctness here on the exact phase boundaries of a pytest internal
        # we do not control would be one refactor away from a silent regression.
        try:
            os.pwrite(fd, struct.pack('<Q', 0), page_base)
        except OSError:
            pass


@pytest.fixture(scope='session')
def gpu_device_class() -> str:
    """Accelerator class used to build ``gpu_device``. Defaults to ``'cuda'``.

    The lease mechanism itself is device-agnostic -- it hands out an ordinal and
    never imports torch -- so retargeting a suite at another backend is purely a
    matter of how that ordinal is spelled. Override this fixture in a conftest.py
    (see "Extending to other device classes" in the plan) or set
    ``GPU_LEASE_DEVICE_CLASS`` in the environment.
    """
    return os.getenv('GPU_LEASE_DEVICE_CLASS', default='cuda')


@pytest.fixture(scope='session')
def gpu_device(gpu_id, gpu_device_class) -> str:
    """``gpu_id`` as a torch device string, e.g. ``'cuda:3'`` or ``'xpu:3'``."""
    return f'{gpu_device_class}:{gpu_id}'


@pytest.fixture(scope='session')
def torch_gpu(gpu_id) -> int:
    """Back-compat alias for :func:`gpu_id`."""
    return gpu_id
