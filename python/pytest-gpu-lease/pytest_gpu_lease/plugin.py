# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lease exactly one GPU to each pytest-xdist worker.

Workers of a single pytest run coordinate through POSIX record locks on one shared
file: GPU *n* is represented by the byte range [n*PAGE_SIZE, (n+1)*PAGE_SIZE). A
worker walks the range round-robin and takes the first page it can write-lock; it
holds that lock for its whole session and releases it at teardown.

Three modes, selected by environment:

* ``PYTEST_XDIST_WORKER_COUNT == 0`` (no xdist) -- GPU 0, no locking.
* ``GPU_LEASE_PIN=<n>`` -- every worker pinned to GPU n, no locking. For single-GPU
  reruns and for bisecting a failure onto a known-good device.
* otherwise -- lease as described above.

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

import pytest

GPU_LEASE_PIN = os.getenv('GPU_LEASE_PIN', default=None)
PYTEST_XDIST_WORKER_COUNT = int(os.getenv('PYTEST_XDIST_WORKER_COUNT', default='0'))

STRUCT_FLOCK = 'hhllh'
PAGE_SIZE = 4096
_RETRY_INTERVAL = 0.05


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
def gpu_id(request, worker_id):
    """Index of the GPU this worker owns for the duration of its session."""
    if PYTEST_XDIST_WORKER_COUNT == 0:
        yield 0
        return
    if GPU_LEASE_PIN is not None:
        yield int(GPU_LEASE_PIN)
        return

    # Resolved lazily, NOT as a fixture parameter: pytest instantiates declared
    # params before the body runs, so naming _gpu_lease_lockfile in the signature
    # would create the file in the no-xdist and pinned modes too -- the very
    # side effect dropping `autouse` was meant to prevent.
    lockfile = request.getfixturevalue('_gpu_lease_lockfile')
    with open(lockfile, 'r+b') as f:
        for gpu in itertools.cycle(range(PYTEST_XDIST_WORKER_COUNT)):
            claim = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET,
                                PAGE_SIZE * gpu, PAGE_SIZE, 0)
            try:
                fcntl.fcntl(f, fcntl.F_SETLK, claim)
            except BlockingIOError:
                # Every page is taken for the moment. Sleep instead of spinning --
                # the original loop pegged a core while waiting.
                if gpu == PYTEST_XDIST_WORKER_COUNT - 1:
                    time.sleep(_RETRY_INTERVAL)
                continue
            print(f'{worker_id} uses GPU {gpu} filelock = {lockfile}',
                  file=sys.stderr, flush=True)
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
