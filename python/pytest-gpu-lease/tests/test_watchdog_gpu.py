# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end watchdog tests against a genuinely wedged GPU.

Everything else in this package wedges a process with `os.read` on an empty
pipe, which proves the signalling but not the thing the feature exists for: a
Triton kernel that never retires, a device sync that therefore never returns,
and a worker that no in-process timeout can interrupt. That is what this file
reproduces.

Nothing here runs by default. ``GPU_LEASE_TEST_LEVEL`` selects what is defined
at all, in the spirit of the flash suite's ``FOR_RELEASE``:

* ``0`` (default) -- this module defines no tests.
* ``1`` -- the watchdog test. Hangs two GPUs for about a minute.
* ``2`` -- also the pytest-timeout control, which additionally needs
  pytest-timeout installed (this repo no longer depends on it) and hangs one
  GPU for as long as it takes to prove a timeout did not fire.

They are off by default because they deliberately hang kernels on shared
devices, and because the nested sessions lease from their own lock file and so
cannot see -- or be seen by -- a real pass already using those GPUs.

Measured on gfx1201 before any of this was written, since the whole design
rests on it: the wedged process sits in `R` (spinning in userspace on the
sync, not uninterruptible), SIGKILL reaps it in 0.11s, and the GPU is
immediately reusable afterwards.
"""

import os
import re
import subprocess
import sys

import pytest

pytest_plugins = ['pytester']

_TEST_LEVEL = int(os.getenv('GPU_LEASE_TEST_LEVEL', '0'))


def _hardware_skip_reason() -> str | None:
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


# Level decides what exists; hardware decides whether it can run. Kept apart so
# that asking for these tests on a machine without GPUs reports why, instead of
# silently collecting nothing. Not evaluated at level 0, where importing torch
# would be pure cost for a module that defines nothing.
_HW_SKIP = _hardware_skip_reason() if _TEST_LEVEL >= 1 else None
pytestmark = pytest.mark.skipif(_HW_SKIP is not None, reason=_HW_SKIP or '')


# Shared by both nested suites below: the kernel, and the fixture that binds a
# worker to its leased GPU. `good` and `bad` differ only in the spin count
# handed to this one kernel -- same launch, same comparison afterwards -- so the
# wedge is reached the way a real test reaches it, by comparing a GPU result.
_KERNEL = '''
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


WEDGE = 2 ** 50   # ~1e15 iterations: no wall clock will outlast it


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
    far under any threshold used here, but a slower machine should not be able
    to fail these tests for a reason unrelated to what they check.
    """
    torch.cuda.set_device(gpu_device)
    _vecadd(gpu_device, 0)
    return gpu_device
'''


_WATCHDOG_SUITE = _KERNEL + '''

@pytest.mark.parametrize('kind', ['good', 'bad', 'bad', 'good'])
def test_vecadd(kind, warm_device):
    _vecadd(warm_device, 0 if kind == 'good' else WEDGE)
'''


# Runs in definition order, and the order is the point: a passing test, then a
# Python-level hang that pytest-timeout must catch, then a GPU hang it cannot.
_TIMEOUT_CONTROL_SUITE = _KERNEL + '''
import time


def test_vecadd_good(warm_device):
    _vecadd(warm_device, 0)


def test_python_level_hang(warm_device):
    time.sleep(3600)


def test_vecadd_wedged(warm_device):
    _vecadd(warm_device, WEDGE)
'''


if _TEST_LEVEL >= 1:
    def test_wedged_gpu_worker_is_killed_and_session_survives(pytester, monkeypatch, tmp_path):
        """Two of four tests hang the GPU; the run must still finish, correctly.

        Checks, in order of what would hurt most to lose:

        * no INTERNALERROR -- the failure mode that ended a real pass at 80%,
          and the reason `_tolerate_closed_worker_channel` exists. Two workers
          dying close together is exactly how that race is reached.
        * both `good` tests pass, including any requeued off a killed worker.
        * the watchdog acted: a SIGTERM per wedged worker, and a stack dump
          relayed for each. SIGTERM is expected to end them -- faulthandler
          runs with chain=True -- so SIGKILL is the last resort and normally
          does not appear at all.
        """
        lockfile = tmp_path / 'gpulock'
        lockfile.touch()
        monkeypatch.delenv('GPU_LEASE_PIN', raising=False)
        monkeypatch.setenv('GPU_LEASE_LOCKFILE', str(lockfile))
        pytester.makepyfile(_WATCHDOG_SUITE)

        watchdog = subprocess.Popen(
            [sys.executable, '-m', 'pytest_gpu_lease.watchdog',
             '--lockfile', str(lockfile), '--workers', '2',
             '--threshold', '10', '--grace', '5',
             '--poll_interval', '1'],
            stderr=subprocess.PIPE, text=True)
        try:
            result = pytester.runpytest_subprocess(
                '-n', '2', '--max-worker-restart', '9999', '-p', 'xdist', timeout=300)
            # Stopped from outside, like run-test.sh does: the watchdog never
            # decides on its own that a pass is over. This run is the case that
            # would have fooled a rule based on an idle lock file -- both
            # workers are killed at once, and neither replacement holds a lease
            # until it has re-imported torch.
            watchdog.terminate()
            _, watchdog_err = watchdog.communicate(timeout=60)
        finally:
            if watchdog.poll() is None:
                watchdog.kill()
                watchdog.communicate()

        assert result.ret is not None, 'nested session never terminated'

        transcript = '\\n'.join(result.outlines + result.errlines)
        assert 'INTERNALERROR' not in transcript, transcript[-4000:]

        outcomes = result.parseoutcomes()
        assert outcomes.get('passed', 0) == 2, (outcomes, transcript[-4000:])

        assert watchdog_err.count('sent SIGTERM') >= 2, watchdog_err
        # The dump is the point of the SIGTERM; assert it arrived rather than
        # asserting on SIGKILL, which a worker that obeys SIGTERM never needs.
        assert watchdog_err.count('stack of pid') >= 2, watchdog_err


if _TEST_LEVEL >= 2:
    # Long enough that a slow machine cannot make it fire late by accident,
    # short enough that the wall clock below is many times it.
    _CONTROL_TIMEOUT_S = 5
    _CONTROL_WALL_S = 40

    def _verdict(transcript: str, name: str) -> str | None:
        """The PASSED/FAILED/ERROR pytest recorded for `name`, or None.

        Scoped to the span between this test's `-v` line and the next one. The
        lease announcement goes to stderr, unbuffered, and lands between a test
        name and its verdict, so a search that just scans forward from the name
        would run past the end of the test and pick up its successor's verdict
        -- which for the wedged test is the difference between "no verdict, as
        predicted" and a false pass.
        """
        segment = re.search(rf'::{re.escape(name)}\b(.*?)(?=^\S+\.py::|\Z)',
                            transcript, re.S | re.M)
        if segment is None:
            return None
        found = re.search(r'\b(PASSED|FAILED|ERROR)\b', segment.group(1))
        return found.group(1) if found else None

    def test_pytest_timeout_cannot_interrupt_a_wedged_kernel(pytester, tmp_path):
        """Control for the watchdog: show pytest-timeout does not cover this.

        This is the claim the whole feature rests on -- a Level-3 pass ran
        265,282 tests in 15h46m under `--timeout=300` and reported zero
        timeouts -- so it is worth demonstrating rather than asserting.
        pytest-timeout's signal method arms `setitimer(ITIMER_REAL)` and raises
        from a SIGALRM handler, and CPython runs a Python-level signal handler
        only when the main thread reaches a bytecode boundary. A thread inside
        an unreturning device sync reaches none.

        Both halves matter. A run where the timeout simply never fires proves
        nothing -- the plugin might not be loaded, or the option misspelled --
        so the same session first hangs in `time.sleep`, where the timeout
        *must* fire. What separates the two cases is only where the main
        thread is parked.

        Deliberately without xdist: nothing here should be attributable to the
        combination pytest#2223 warns about.
        """
        pytest.importorskip(
            'pytest_timeout',
            reason='the control needs the very plugin the repo dropped; '
                   'pip install pytest-timeout to run it')
        import torch

        nested = pytester.makepyfile(_TIMEOUT_CONTROL_SUITE)
        transcript_path = tmp_path / 'nested.out'

        env = dict(os.environ)
        env.pop('GPU_LEASE_LOCKFILE', None)
        # Pin rather than lease: with no xdist there is nothing to coordinate,
        # and pinning keeps this off the lock file entirely. Highest ordinal on
        # the theory that a pass, if one is running, started at 0.
        env['GPU_LEASE_PIN'] = str(torch.cuda.device_count() - 1)
        # Without this the transcript is block-buffered and lost when we kill
        # the process -- which we always do, since it never finishes.
        env['PYTHONUNBUFFERED'] = '1'

        with open(transcript_path, 'w') as sink:
            proc = subprocess.Popen(
                [sys.executable, '-m', 'pytest', str(nested), '-v',
                 '-p', 'no:xdist', '-p', 'no:cacheprovider',
                 '--timeout', str(_CONTROL_TIMEOUT_S), '--timeout-method', 'signal'],
                stdout=sink, stderr=subprocess.STDOUT, cwd=str(pytester.path), env=env)
            try:
                proc.wait(timeout=_CONTROL_WALL_S)
                finished = True
            except subprocess.TimeoutExpired:
                finished = False
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=60)

        transcript = transcript_path.read_text()

        # pytest-timeout is loaded and armed the way this test intends. Checked
        # from the header rather than from a "Timeout >5.0s" message, because
        # that message lives in the FAILURES section, which a session killed
        # mid-test never prints.
        assert f'timeout: {float(_CONTROL_TIMEOUT_S)}s' in transcript, transcript[-3000:]
        assert 'timeout method: signal' in transcript, transcript[-3000:]

        # The GPU is fine and the kernel is right: the ordinary case passes.
        assert _verdict(transcript, 'test_vecadd_good') == 'PASSED', transcript[-3000:]

        # ...and the timeout does fire, when the main thread is in Python.
        assert _verdict(transcript, 'test_python_level_hang') == 'FAILED', \
            f'pytest-timeout did not fire on time.sleep\n{transcript[-3000:]}'

        # Same timeout, same session, same vecadd kernel -- only the spin count
        # differs from the passing case above. No verdict at all, and the
        # session had to be killed from outside.
        assert 'test_vecadd_wedged' in transcript, transcript[-3000:]
        assert _verdict(transcript, 'test_vecadd_wedged') is None, \
            f'expected no verdict for the wedged test, got one:\n{transcript[-3000:]}'
        assert not finished, (
            f'the wedged test was expected to outlast a {_CONTROL_TIMEOUT_S}s timeout '
            f'for the full {_CONTROL_WALL_S}s, but the session exited on its own\n'
            f'{transcript[-3000:]}')
