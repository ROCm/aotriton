# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-test suite for pytest_gpu_lease.

Needs no GPU: every mode either short-circuits to a plain int or exercises
``fcntl`` against a temp file. Each case drives a nested pytest session via the
``pytester`` fixture, with the environment fully controlled through
``monkeypatch`` so the outer (this) session's own env vars never leak in.

Each case carries its own rationale in its docstring; the crash-and-restart and
no-xdist cases are the two that guard against regressions already seen in the wild.
"""

import pytest

pytest_plugins = ['pytester']


def _clear_lease_env(monkeypatch):
    monkeypatch.delenv('GPU_LEASE_PIN', raising=False)
    monkeypatch.delenv('GPU_LEASE_DEVICE_CLASS', raising=False)
    # Must also go: a stale value from the outer run would leak into the
    # nested one. The plugin ignores it now, but leaving it set would hide
    # a regression back to reading it.
    monkeypatch.delenv('PYTEST_XDIST_WORKER_COUNT', raising=False)
    # Same leak risk as above, but for real this time: run-test.sh (and a
    # developer's own shell, while poking at the watchdog) sets these, and a
    # value inherited here would silently redirect a test that asserts on the
    # *derived* lockfile path to whatever the outer run happened to be using.
    monkeypatch.delenv('GPU_LEASE_LOCKFILE', raising=False)
    monkeypatch.delenv('GPU_LEASE_BUDGET_S', raising=False)


def test_no_xdist_gpu_id_is_zero_and_no_lockfile(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)

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
    """Distinct GPU per worker, driven purely by `-n` with no env hints.

    This is the regression guard for the collapse-to-GPU-0 bug: the plugin used
    to decide the mode from PYTEST_XDIST_WORKER_COUNT read at module import,
    which as a pytest11 entry-point plugin runs before xdist populates the
    worker environment. It read 0, took the no-xdist branch, and put all eight
    CI workers on GPU 0. `_clear_lease_env` deletes that variable precisely so
    this test fails if the plugin ever reaches for it again.
    """
    _clear_lease_env(monkeypatch)
    ledger = tmp_path / 'leases.txt'
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


def test_gpu_lease_lockfile_env_overrides_derived_path(pytester, monkeypatch, tmp_path):
    """``GPU_LEASE_LOCKFILE`` wins over the ``tmp_path_factory``-derived path.

    The watchdog is a separate process started before this pytest session
    exists, so it cannot predict where ``getbasetemp()`` will land; run-test.sh
    instead mints the path itself and both sides read it from the environment.
    This is the regression guard for that override actually taking effect --
    without it, the watchdog and the workers would each be watching a
    different file and the whole mechanism would silently do nothing.
    """
    _clear_lease_env(monkeypatch)
    lockfile = tmp_path / 'chosen_by_env_gpulock'
    monkeypatch.setenv('GPU_LEASE_LOCKFILE', str(lockfile))

    pytester.makepyfile("""
        def test_it(gpu_id):
            assert gpu_id == 0
    """)
    result = pytester.runpytest_subprocess('-n', '1', '-p', 'xdist')
    result.assert_outcomes(passed=1)

    assert lockfile.exists(), 'the env-chosen path must be the one actually used'
    assert not list(pytester.path.rglob('gpulock')), \
        'the derived path must not also be created once the env var wins'


def test_gpu_device_default_is_cuda(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)

    pytester.makepyfile("""
        def test_it(gpu_device):
            assert gpu_device == 'cuda:0'
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_gpu_device_class_via_env(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('GPU_LEASE_DEVICE_CLASS', 'xpu')

    pytester.makepyfile("""
        def test_it(gpu_device):
            assert gpu_device == 'xpu:0'
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_gpu_device_class_conftest_override_wins_over_env(pytester, monkeypatch):
    _clear_lease_env(monkeypatch)
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


def test_announcement_is_live_not_replayed(pytester, monkeypatch):
    """The GPU choice must reach stderr as it happens, not only in the report.

    Regression test for the original complaint: the announcement was printed
    from fixture setup, which pytest captures at the fd level and replays in a
    "Captured stderr setup" block -- shown only for failing tests, so a green
    run revealed the assignment after it was already over.

    ``_announce`` suspends capture, so the line lands on the real stderr. Two
    assertions pin that down: it IS in the subprocess's stderr stream, and it is
    NOT sitting inside a replayed capture section on stdout.
    """
    _clear_lease_env(monkeypatch)

    pytester.makepyfile("""
        def test_it(gpu_id):
            assert gpu_id == 0
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)

    assert any('uses GPU 0' in line for line in result.errlines), \
        "lease announcement did not reach the real stderr; capture was not suspended"
    assert not any('Captured stderr' in line for line in result.outlines), \
        "announcement was captured and replayed instead of printed live"


def test_gpu_id_resolves_without_the_xdist_plugin(pytester, monkeypatch):
    """`gpu_id` must resolve when xdist is not loaded at all.

    Regression guard for review r3798750956: the fixture took xdist's `worker_id`
    as a parameter, so `-p no:xdist` (or PYTEST_DISABLE_PLUGIN_AUTOLOAD) made
    `gpu_id` unresolvable with a "fixture not found" error -- including on the
    no-xdist and pinned paths, which need nothing from xdist. The worker label is
    now derived from config.workerinput, which is simply absent here.
    """
    _clear_lease_env(monkeypatch)

    pytester.makepyfile("""
        def test_it(gpu_id, gpu_device):
            assert gpu_id == 0
            assert gpu_device == 'cuda:0'
    """)
    result = pytester.runpytest_subprocess('-p', 'no:xdist')
    result.assert_outcomes(passed=1)
    assert any('master uses GPU 0' in line for line in result.errlines), \
        "off-worker runs should announce under the 'master' label"


def test_pinned_resolves_without_the_xdist_plugin(pytester, monkeypatch):
    """GPU_LEASE_PIN is a debugging override and must not require xdist either."""
    _clear_lease_env(monkeypatch)
    monkeypatch.setenv('GPU_LEASE_PIN', '3')

    pytester.makepyfile("""
        def test_it(gpu_id):
            assert gpu_id == 3
    """)
    result = pytester.runpytest_subprocess('-p', 'no:xdist')
    result.assert_outcomes(passed=1)
    assert not list(pytester.path.rglob('gpulock')), \
        "pinned mode must not create a lockfile"


def test_sendcommand_guard_swallows_closed_channel_instead_of_raising(capsys):
    """Unit test for the controller-side scheduler guard, exercised directly
    rather than by racing two real worker crashes against each other.

    The actual failure -- two workers dying close enough together that the
    second's channel is already closed by the time the first's crash handling
    tries to top up its queue -- is exactly the kind of race that only
    sometimes reproduces under `-n N`. Driving it deterministically here,
    against the real `xdist.workermanage.WorkerController` class rather than a
    substitute, tests the actual guard installed in the real environment.
    """
    xdist_workermanage = pytest.importorskip('xdist.workermanage')
    WorkerController = xdist_workermanage.WorkerController

    original = WorkerController.sendcommand
    try:
        def _always_closed(self, name, **kwargs):
            raise OSError('cannot send (already closed?)')

        WorkerController.sendcommand = _always_closed
        from pytest_gpu_lease.plugin import _tolerate_closed_worker_channel
        _tolerate_closed_worker_channel()

        class _FakeGateway:
            id = 'gw-fake'

        class _FakeNode:
            gateway = _FakeGateway()

        # Must not raise: this is the exact call site (LoadScheduling._send_tests
        # -> WorkerController.send_runtest_some -> sendcommand) that used to
        # propagate OSError all the way out to an INTERNALERROR.
        WorkerController.sendcommand(_FakeNode(), 'runtests', indices=[1, 2])

        err = capsys.readouterr().err
        assert 'channel already closed' in err
        assert 'gw-fake' in err
    finally:
        WorkerController.sendcommand = original


def test_sendcommand_guard_is_idempotent(capsys):
    """Calling the guard installer twice must not stack a second wrapper --
    `pytest_configure` is not guaranteed to run exactly once per interpreter
    in every embedding, and a double-wrap would still work but would print
    the "channel already closed" line twice per failure for no reason.
    """
    xdist_workermanage = pytest.importorskip('xdist.workermanage')
    WorkerController = xdist_workermanage.WorkerController

    original = WorkerController.sendcommand
    try:
        def _always_closed(self, name, **kwargs):
            raise OSError('cannot send (already closed?)')

        WorkerController.sendcommand = _always_closed
        from pytest_gpu_lease.plugin import _tolerate_closed_worker_channel
        _tolerate_closed_worker_channel()
        _tolerate_closed_worker_channel()

        class _FakeGateway:
            id = 'gw-fake'

        class _FakeNode:
            gateway = _FakeGateway()

        WorkerController.sendcommand(_FakeNode(), 'runtests', indices=[1])
        assert capsys.readouterr().err.count('channel already closed') == 1
    finally:
        WorkerController.sendcommand = original
