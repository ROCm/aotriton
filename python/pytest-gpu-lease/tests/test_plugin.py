# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-test suite for pytest_gpu_lease.

Needs no GPU: every mode either short-circuits to a plain int or exercises
``fcntl`` against a temp file. Each case drives a nested pytest session via the
``pytester`` fixture, with the environment fully controlled through
``monkeypatch`` so the outer (this) session's own env vars never leak in.

See ".claude/docs/pytest_gpu_lease_plan.md", "The plugin's own test suite", for
the rationale behind each case, in particular the worker-crash-and-restart case.
"""

import pytest

pytest_plugins = ['pytester']


def _clear_lease_env(monkeypatch):
    monkeypatch.delenv('GPU_LEASE_PIN', raising=False)
    monkeypatch.delenv('GPU_LEASE_DEVICE_CLASS', raising=False)


def test_no_xdist_gpu_id_is_zero_and_no_lockfile(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '0')

    pytester.makepyfile("""
        def test_it(gpu_id):
            assert gpu_id == 0
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    assert not list(pytester.path.rglob('gpulock')), \
        "no-xdist mode must not create a lockfile"


def test_pinned_gpu_id_matches_pin_and_no_lockfile(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '2')
    monkeypatch.setenv('GPU_LEASE_PIN', '3')

    pytester.makepyfile("""
        def test_it(gpu_id):
            assert gpu_id == 3
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    assert not list(pytester.path.rglob('gpulock')), \
        "pinned mode must not create a lockfile"


def test_leased_mode_assigns_distinct_gpus_and_creates_lockfile(pytester, monkeypatch, tmp_path):
    _clear_lease_env(monkeypatch)
    ledger = tmp_path / 'leases.txt'
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '2')
    monkeypatch.setenv('GPU_LEASE_LEDGER', str(ledger))

    pytester.makeconftest("""
        import os, pytest

        @pytest.fixture(autouse=True)
        def _record(worker_id, gpu_id):
            with open(os.environ['GPU_LEASE_LEDGER'], 'a') as f:
                f.write(f'{worker_id} {gpu_id}\\n')
    """)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.parametrize('i', range(4))
        def test_it(i):
            pass
    """)
    result = pytester.runpytest_subprocess('-n', '2', '-p', 'xdist')
    result.assert_outcomes(passed=4)

    entries = [line.split() for line in ledger.read_text().splitlines()]
    gpus = {int(gpu) for _, gpu in entries}
    # Exactly two distinct GPUs, one per worker -- 1:1, no oversubscription.
    assert gpus == {0, 1}
    assert list(pytester.path.rglob('gpulock')), \
        "leased mode must create the shared lockfile"


def test_gpu_device_default_is_cuda(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '0')

    pytester.makepyfile("""
        def test_it(gpu_device):
            assert gpu_device == 'cuda:0'
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_gpu_device_class_via_env(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '0')
    monkeypatch.setenv('GPU_LEASE_DEVICE_CLASS', 'xpu')

    pytester.makepyfile("""
        def test_it(gpu_device):
            assert gpu_device == 'xpu:0'
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_gpu_device_class_conftest_override_wins_over_env(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '0')
    monkeypatch.setenv('GPU_LEASE_DEVICE_CLASS', 'xpu')

    pytester.makeconftest("""
        import pytest

        @pytest.fixture(scope='session')
        def gpu_device_class():
            return 'rocm'
    """)
    pytester.makepyfile("""
        def test_it(gpu_device):
            assert gpu_device == 'rocm:0'
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_no_autouse_lockfile_absent_when_gpu_id_not_requested(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '2')

    pytester.makepyfile("""
        def test_it():
            assert True
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    assert not list(pytester.path.rglob('gpulock')), \
        "the lockfile fixture must stay inert unless gpu_id is actually requested"


@pytest.mark.timeout(60)
def test_replacement_worker_releases_and_reacquires(pytester, monkeypatch, tmp_path):
    ledger = tmp_path / 'leases.txt'
    monkeypatch.setenv('PYTEST_XDIST_WORKER_COUNT', '2')
    monkeypatch.setenv('GPU_LEASE_LEDGER', str(ledger))   # test-only, read by the conftest below

    pytester.makeconftest("""
        import os, pytest

        @pytest.fixture(autouse=True)
        def _record(worker_id, gpu_id):
            with open(os.environ['GPU_LEASE_LEDGER'], 'a') as f:
                f.write(f'{worker_id} {gpu_id}\\n')
    """)
    pytester.makepyfile("""
        import os, pytest

        @pytest.mark.parametrize('i', range(8))
        def test_maybe_crash(i):
            if i == 3:
                os._exit(139)      # emulate exit_pytest(); teardown never runs
    """)

    result = pytester.runpytest_subprocess('-n', '2', '--max-worker-restart', '4', '-p', 'xdist')

    entries = [l.split() for l in ledger.read_text().splitlines()]
    gpus = [int(g) for _, g in entries]

    # 1. The run terminated at all -- a leaked lease would hang until the outer timeout.
    assert result.ret is not None
    # 2. Every lease is in range for a 2-GPU pool.
    assert set(gpus) <= {0, 1}
    # 3. Both pages were in play, and the pool was re-entered after the crash:
    #    more lease events than workers means at least one re-lease happened.
    assert len(entries) > 2
    assert set(gpus) == {0, 1}
