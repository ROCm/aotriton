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
# `pytest_runtest_protocol` hookwrapper below needs to know whether there is
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
            # Arm a deadline right now, not just from the first heartbeat below.
            # `pytest_runtest_protocol`'s hookwrapper (below) wraps setup, call
            # and teardown as a single hook call, and its pre-yield code -- where
            # a fresh deadline is normally written -- runs *before* that call
            # even starts, i.e. before this fixture body has run at all. So for
            # this worker's first test, that pre-yield code sees `_active_lease`
            # still `None` and, correctly, touches nothing; without this write,
            # the page would carry no deadline through the whole of that first
            # test's setup, call, and teardown, and a wedge anywhere in it would
            # be invisible to the watchdog. This write closes exactly that gap
            # and nothing else: every later test gets a fresh deadline from the
            # hookwrapper itself, since by then `_active_lease` is already set
            # when that wrapper's pre-yield code runs.
            os.pwrite(f.fileno(), struct.pack('<Q', time.monotonic_ns() + _budget_ns()), page_base)
            try:
                yield gpu
            finally:
                _active_lease = None
                release = struct.pack(STRUCT_FLOCK, fcntl.F_UNLCK, os.SEEK_SET,
                                      page_base, PAGE_SIZE, 0)
                fcntl.fcntl(f, fcntl.F_SETLK, release)
            return


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol():
    """Heartbeat this worker's page with the current test's deadline.

    Inert unless `gpu_id` actually holds a lease: `_active_lease` is only set by
    the leased branch above, so pinned mode, no-xdist mode, and GPU-less suites
    (which never request `gpu_id`, so that fixture body never runs at all) hit
    the `is None` check and do no I/O -- this hook fires for every test in every
    pytest process in the environment, so being a no-op absent a lease is not
    optional.

    Wraps the whole protocol -- setup, call, and teardown -- deliberately, so a
    worker that wedges anywhere in any of the three (a GPU allocation during
    setup, the test body itself, a cache-empty or sync during teardown) is
    covered by one and the same deadline. The alternative, wrapping only
    `pytest_runtest_call`, was tried first and rejected: it left setup and
    teardown outside the heartbeat entirely, so a fixture-level wedge would
    never be caught at all.

    That whole-protocol wrapping has one gap of its own, and `gpu_id` closes it
    rather than this hook: `runtestprotocol()` still runs setup, call and
    teardown as separate hook calls internally, and `gpu_id`'s lock-acquire
    (which sets `_active_lease`) happens inside the first of those, i.e. *after*
    this wrapper's pre-yield code has already run for a worker's first test.
    That pre-yield code reads `_active_lease` once, before the wrapped call
    starts, so on the first test it correctly sees `None` and writes nothing --
    but with nothing else in place, that leaves the entire first test
    heartbeat-less. (Caught by an end-to-end test that wedged the very first
    parametrized case on a worker: the watchdog never fired, because no
    deadline had ever been written for that page.) `gpu_id` covers this by
    writing an initial deadline itself, at the moment the lease is taken --
    see the comment there. From the second test onward, `_active_lease` is
    already set by the time this wrapper's pre-yield code runs, so it takes
    over the refreshing normally. The two writers never race: at most one of
    them is ever the one enabled for a given test, controlled by the same
    single check of `_active_lease` here.

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
    (The first test is the one exception: this wrapper takes the `is None`
    branch for it and never zeros afterwards, leaving `gpu_id`'s initial write
    in place until the second test's pre-yield code overwrites it. That stale
    value cannot cause a false idle read -- it is a real, if slightly dated,
    future deadline, not a zero -- and cannot cause a false SIGTERM either
    unless the gap between the first test finishing and the second one
    starting itself exceeds the budget, which would mean the worker really is
    stuck somewhere between tests.)
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
        # On this worker's last test (`nextitem is None`), the very call being
        # wrapped also tears down the session-scoped `gpu_id` fixture -- which
        # closes `fd` and releases the lock -- before control returns here, so
        # `fd` may already be a stale, closed number. That is fine to ignore:
        # zeroing a deadline is only ever meaningful while the page is still
        # locked, and the watchdog never reads a page's deadline without first
        # finding it locked, so a released page's stale value is never seen.
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


def _tolerate_closed_worker_channel() -> None:
    """Stop a second dying worker from turning the first one's crash into an
    INTERNALERROR that kills the whole session.

    The failure this guards against: the controller notices worker A died and
    calls ``DSession.worker_errordown`` -> ``LoadScheduling.remove_node(A)``,
    which reassigns A's pending items and then, still inside that same call,
    tops up every *other* node's queue via ``check_schedule`` ->
    ``_send_tests`` -> ``WorkerController.send_runtest_some`` ->
    ``sendcommand`` -> ``execnet``'s ``channel.send``. If worker B is also
    dying at that exact moment -- its own crash not yet reported, since that
    report is itself an asynchronous execnet callback racing this one -- that
    send raises ``OSError: cannot send (already closed?)``, uncaught, and
    pytest reports it as an INTERNALERROR that ends the run, workers that were
    fine included. This is not specific to any one test suite or wedge
    scenario; it is a bug in xdist's own error path with no synthetic
    reproduction needed here -- it was hit organically while exercising the
    watchdog above, with no wedge involved at all, just two ordinary crashes
    close enough together in time.

    Every scheduling mode (``--dist=load/loadscope/loadgroup/worksteal/each``)
    has its own scheduler class and its own call site for sending tests, but
    every one of them ends up going through this same
    ``WorkerController.sendcommand`` -- so patching here, instead of one
    scheduler class's ``_send_tests``, is the version of this fix that
    actually covers every ``--dist`` mode rather than just the default one.
    It is also not really foreign monkeypatching so much as finishing a job
    xdist already started: ``WorkerController.shutdown`` right next to this
    method already wraps its own ``sendcommand`` call in ``try/except
    OSError: pass`` for exactly this reason -- every other caller (the
    schedulers) was simply missed.

    Swallowing the error here and doing nothing further is deliberate, not
    lazy: bookkeeping a phantom "sent" test against a dead node is harmless,
    because that node's own crash -- already in flight, asynchronously -- will
    shortly call ``remove_node`` on it too, which pops its *entire* pending
    list (phantom entry included) and requeues it for a live node. Retrying
    the send, or reaching into the scheduler to remove the node from inside
    this call, would instead re-enter node removal from inside the very send
    that node removal itself triggered -- recursion worth avoiding on
    principle, not just because nothing here needs it: the safer, simpler
    fix is to leave that removal to the async crash-detection path that is
    already in flight for this node.

    A no-op on any run that never sees a closed channel: the wrapped function
    still calls straight through, so a healthy session pays only the cost of
    one extra Python frame per command sent.
    """
    try:
        from xdist.workermanage import WorkerController
    except ImportError:
        return  # xdist not installed in this environment, or this process never imports it

    original = WorkerController.sendcommand
    if getattr(original, '_gpu_lease_tolerates_closed_channel', False):
        return  # pytest_configure can run more than once in the same interpreter

    def sendcommand(self, name, **kwargs):
        try:
            original(self, name, **kwargs)
        except OSError as exc:
            print(f'pytest_gpu_lease: {self.gateway.id} channel already closed, '
                  f'dropping {name}({kwargs}) instead of crashing the session ({exc})',
                  file=sys.stderr, flush=True)

    sendcommand._gpu_lease_tolerates_closed_channel = True
    WorkerController.sendcommand = sendcommand


@pytest.hookimpl
def pytest_configure(config):
    """Apply the scheduler guard above once, in every process this plugin loads
    into. Harmless where it does not apply: workers never call
    ``WorkerController.sendcommand`` (only the controller schedules), and a
    plain non-distributed run never imports ``xdist.workermanage`` at all, so
    the early return in ``_tolerate_closed_worker_channel`` makes this a no-op
    there.
    """
    _tolerate_closed_worker_channel()
