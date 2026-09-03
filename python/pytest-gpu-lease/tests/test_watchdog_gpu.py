# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end watchdog test against a genuinely wedged GPU.

Everything else in this package wedges a process with `os.read` on an empty
pipe, which proves the signalling but not the thing the feature exists for: a
Triton kernel that never retires, a device sync that therefore never returns,
and a worker that no in-process timeout can interrupt. That is what this file
reproduces.

Opt-in via ``GPU_LEASE_GPU_TESTS=1``. It occupies two GPUs for about a minute
and deliberately hangs kernels on them, so it must not run by accident
alongside a real pass -- the nested session leases from its own lock file and
so cannot see, or be seen by, a pass already using those devices.

Measured on gfx1201 before this was written, since the whole design rests on
it: the wedged process sits in `R` (spinning in userspace on the sync, not
uninterruptible), SIGKILL reaps it in 0.11s, and the GPU is immediately
reusable afterwards -- two subsequent workloads on the same device passed.
"""

import os
import subprocess
import sys

import pytest

pytest_plugins = ['pytester']

_OPT_IN = 'GPU_LEASE_GPU_TESTS'


def _skip_reason() -> str | None:
    if not os.getenv(_OPT_IN):
        return f'set {_OPT_IN}=1 to run (hangs two GPUs for ~1 minute)'
    try:
        import torch
        import triton  # noqa: F401
    except ImportError as exc:
        return f'needs torch and triton ({exc})'
    if not torch.cuda.is_available():
        return 'needs a GPU'
    if torch.cuda.device_count() < 2:
        return 'needs 2 GPUs: the nested session runs -n 2'
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or '')


# The nested suite. `good` and `bad` differ only in the spin count handed to
# one kernel -- same launch, same comparison afterwards -- so the wedge is
# reached the way a real test reaches it: by comparing a GPU result, which
# syncs, rather than by calling synchronize() explicitly.
_NESTED_SUITE = '''
import torch
import triton
import triton.language as tl

import pytest


@triton.jit
def vecadd_spin(x_ptr, y_ptr, out_ptr, spin_ptr, n, BLOCK: tl.constexpr):
    """out = x + y, after spinning `*spin_ptr` times.

    The trip count is read from device memory so it cannot be constant-folded,
    and the body is a floating-point recurrence feeding the store, so it can
    be neither strength-reduced (fp add/mul are not associative, so the closed
    form is not a legal rewrite) nor deleted as dead. At spin=0 the loop does
    not execute and the kernel is an ordinary vector add.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(x_ptr + offs, mask=mask, other=0.0) + tl.load(y_ptr + offs, mask=mask, other=0.0)
    limit = tl.load(spin_ptr)
    i = tl.zeros((), dtype=tl.int64)
    while i < limit:
        acc = acc * 0.9999999 + 1.0
        i += 1
    tl.store(out_ptr + offs, acc, mask=mask)


def _vecadd(device, spin):
    n = 4096
    x = torch.rand(n, device=device)
    y = torch.rand(n, device=device)
    out = torch.empty_like(x)
    spin_t = torch.tensor([spin], dtype=torch.int64, device=device)
    vecadd_spin[(triton.cdiv(n, 1024),)](x, y, out, spin_t, n, BLOCK=1024)
    # The sync that hangs. Comparing the result is what a normal test does, and
    # it is what forces the device sync -- calling synchronize() by hand would
    # work too but would not resemble the workload this protects.
    torch.testing.assert_close(out, x + y)


@pytest.fixture(scope='session')
def warm_device(gpu_device):
    """Bind this worker to its leased GPU, then compile the kernel once.

    set_device is not optional. Tensors carry their device, but Triton launches
    on the *current* one, so leasing cuda:1 and never setting it launches the
    kernel on device 0 against device-1 pointers -- which faults the GPU
    ("Memory access fault ... kernel: vecadd_spin") rather than failing
    cleanly, and looks nothing like the wedge under test.

    Compiling here keeps JIT out of the timed window. Measured cold at 1.1s,
    far under the threshold, but a slower machine should not be able to fail
    this test for a reason unrelated to what it checks.
    """
    torch.cuda.set_device(gpu_device)
    _vecadd(gpu_device, 0)
    return gpu_device


@pytest.mark.parametrize('kind', ['good', 'bad', 'bad', 'good'])
def test_vecadd(kind, warm_device):
    _vecadd(warm_device, 0 if kind == 'good' else 2 ** 50)
'''


def test_wedged_gpu_worker_is_killed_and_session_survives(pytester, monkeypatch, tmp_path):
    """Two of four tests hang the GPU; the run must still finish, correctly.

    Checks, in order of what would hurt most to lose:

    * no INTERNALERROR -- the failure mode that ended a real pass at 80%, and
      the reason `_tolerate_closed_worker_channel` exists. Two workers dying
      close together is exactly how that race is reached.
    * both `good` tests pass, including any that had to be requeued off a
      killed worker.
    * the watchdog escalated: SIGTERM first (faulthandler turns it into a
      stack dump rather than a death, so the wedged worker survives it), then
      SIGKILL once the grace period expires.
    """
    lockfile = tmp_path / 'gpulock'
    lockfile.touch()
    monkeypatch.delenv('GPU_LEASE_PIN', raising=False)
    monkeypatch.setenv('GPU_LEASE_LOCKFILE', str(lockfile))
    pytester.makepyfile(_NESTED_SUITE)

    # idle_polls is deliberately generous. A replacement worker re-imports
    # torch before it reaches its first test and takes a lease, so with both
    # workers killed at once the lock file can legitimately show no locks for
    # several seconds; a short idle window would let the watchdog self-exit
    # mid-run.
    watchdog = subprocess.Popen(
        [sys.executable, '-m', 'pytest_gpu_lease.watchdog',
         '--lockfile', str(lockfile), '--workers', '2',
         '--threshold', '10', '--grace', '5',
         '--poll_interval', '1', '--idle_polls', '30'],
        stderr=subprocess.PIPE, text=True)
    try:
        result = pytester.runpytest_subprocess(
            '-n', '2', '--max-worker-restart', '9999', '-p', 'xdist', timeout=300)
        _, watchdog_err = watchdog.communicate(timeout=120)
    finally:
        if watchdog.poll() is None:
            watchdog.kill()
            watchdog.communicate()

    assert result.ret is not None, 'nested session never terminated'

    transcript = '\n'.join(result.outlines + result.errlines)
    assert 'INTERNALERROR' not in transcript, transcript[-4000:]

    outcomes = result.parseoutcomes()
    assert outcomes.get('passed', 0) == 2, (outcomes, transcript[-4000:])

    assert watchdog_err.count('SIGTERM') >= 2, watchdog_err
    assert watchdog_err.count('SIGKILL') >= 2, watchdog_err
