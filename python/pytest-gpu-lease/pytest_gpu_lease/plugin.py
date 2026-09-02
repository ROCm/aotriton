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
import itertools
import os
import struct
import sys
import time
from types import MappingProxyType

import pytest

STRUCT_FLOCK = 'hhqqi'
PAGE_SIZE = 4096
_RETRY_INTERVAL = 0.05


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
    """
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
    with open(lockfile, 'r+b') as f:
        for gpu in itertools.cycle(range(nworkers)):
            claim = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET,
                                PAGE_SIZE * gpu, PAGE_SIZE, 0)
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
            try:
                yield gpu
            finally:
                release = struct.pack(STRUCT_FLOCK, fcntl.F_UNLCK, os.SEEK_SET,
                                      PAGE_SIZE * gpu, PAGE_SIZE, 0)
                fcntl.fcntl(f, fcntl.F_SETLK, release)
            return


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
