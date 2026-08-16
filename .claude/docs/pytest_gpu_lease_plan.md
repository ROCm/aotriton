# `pytest-gpu-lease`: Extract the GPU-Leasing Fixture into a Standalone Plugin

Implementation plan. Everything needed to execute is inline — file contents, exact
edits with line anchors, and verification commands.

## Motivation

`modules/flash/tests/_core_test_backward.py:62-103` defines a `torch_gpu` fixture that
hands each `pytest-xdist` worker its own GPU, coordinating via `fcntl` byte-range locks
(one 4 KiB page per GPU) on a lockfile shared through `tmp_path_factory.getbasetemp().parent`.

It is buried in a flash-specific test module. The only way other test files reach it is by
re-importing the fixture names (`test_backward.py:26-27`, `triton_tester.py:19-20`) — a
pattern that does not scale past the flash module. As the repo modularizes into
`modules/<family>/tests/`, every future module needs the same worker→GPU assignment.

Extract it into a standalone, separately installable pytest plugin so any suite gets it for
free via the `pytest11` entry point.

## Decisions

| Topic | Decision |
|---|---|
| Name | `pytest-gpu-lease`. A *lease* is an exclusive, bounded, released claim — exactly the `fcntl` acquire/yield/release lifecycle. |
| Location | `python/pytest-gpu-lease/`, its own distribution with its own `pyproject.toml`. |
| Activation | `pytest11` entry point; installed through `requirements-dev.txt`. |
| Public fixtures | `gpu_id` (int), `gpu_device` (`'<class>:N'`), `gpu_device_class` (`'cuda'`, overridable), `torch_gpu` (back-compat alias). |
| Configuration | Environment variables only: `GPU_LEASE_PIN`, `GPU_LEASE_DEVICE_CLASS`. xdist state comes from `config.workerinput`, never the environment. |
| Oversubscription | **Not supported.** Strict 1:1 worker↔GPU — see below. |
| Scope | Relocation + behaviour-preserving cleanups. No semantic change. |

### Why strictly 1:1, and no `--gpu_lease_count` knob

A pytest CLI option to decouple GPU count from worker count was considered and
**deliberately rejected for now**. Running M tests concurrently on one GPU invites memory
pressure and races in the runtime / driver / firmware / VBIOS. 1:1 is the correct default
and should stay the only mode. Revisit only when async Triton/FlyDSL compilation makes
oversubscription genuinely useful.

Do not add `pytest_addoption` in this change.

### Why a dash-named directory under `python/` is safe

`python/` maps to the importable `aotriton` package via `setup.py`'s
`package_dir`/`find_packages`. A second project living inside it does *not* leak into the
`aotriton` wheel:

- `setup.py:39` calls `find_packages(where=str(_PYDIR))`. setuptools'
  `PackageFinder._find_iter` skips any directory lacking `__init__.py` **and does not
  descend into it**. `python/pytest-gpu-lease/` is a project root (holds `pyproject.toml`,
  no `__init__.py`), so the walk stops there.
- `build_py` for `packages=['aotriton']` with `package_dir={'aotriton': 'python'}` globs
  only `python/*.py`, non-recursively.

> **Constraint to preserve:** `setup.py` must keep using `find_packages`, never
> `find_namespace_packages` — the latter *would* sweep the plugin into the `aotriton` wheel.
> Record this as a comment at the `find_packages` call.

This is verified empirically in the Verification section; do not take it on faith.

---

## Step 1 — Create `python/pytest-gpu-lease/`

```
python/pytest-gpu-lease/
  pyproject.toml
  README.md
  pytest_gpu_lease/
    __init__.py
    plugin.py
  tests/
    test_plugin.py        # GPU-free self-test; see Verification
```

`tests/` has no `__init__.py`, so it is invisible to both `[tool.setuptools] packages`
and the repo-root `find_packages`.

### `pyproject.toml`

```toml
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# A standalone distribution living inside python/, which otherwise maps to the
# `aotriton` package. Safe because the repo-root setup.py uses find_packages(), which
# does not descend into directories lacking __init__.py -- and this directory has none.
# See .claude/docs/pytest_gpu_lease_plan.md.

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "pytest-gpu-lease"
version = "0.1.0"
description = "Lease one GPU per pytest-xdist worker via fcntl byte-range locks"
requires-python = ">=3.10"
dependencies = [
    "pytest",
    "pytest-xdist>=1.15",   # the worker_id fixture identifies the leaseholder
]

[project.entry-points.pytest11]
gpu_lease = "pytest_gpu_lease.plugin"

[tool.setuptools]
packages = ["pytest_gpu_lease"]
```

`pytest-xdist` becomes a hard dependency because the plugin consumes its `worker_id`
fixture unconditionally. Today xdist appears only in `requirements-dev.txt`.

### `pytest_gpu_lease/__init__.py`

Docstring plus `__version__ = '0.1.0'`. Do **not** re-export fixtures here — pytest
discovers them through the entry point's `plugin` module.

### `pytest_gpu_lease/plugin.py`

Port of `_core_test_backward.py:62-103`, restructured. Behaviour by mode:

| Mode | Condition | Yields |
|---|---|---|
| Pinned | `GPU_LEASE_PIN` is set (checked first, xdist or not) | `int(GPU_LEASE_PIN)` |
| No xdist | no `config.workerinput` | `0` |
| Leased | otherwise | round-robin `fcntl` write-lock on page N, held for the session |

> **Do not read `PYTEST_XDIST_WORKER_COUNT`.** See "Detecting xdist" below — doing so
> at module scope is what put all eight CI workers on GPU 0.

### Environment variables

Everything this plugin owns is prefixed `GPU_LEASE_`, matching the distribution name.

| Variable | Default | Effect |
|---|---|---|
| `GPU_LEASE_PIN` | unset | Bypass leasing; pin every worker to this GPU index. |
| `GPU_LEASE_DEVICE_CLASS` | `cuda` | Accelerator class `gpu_device` formats with. |
| `PYTEST_XDIST_WORKER_COUNT` | — | **Not used.** xdist sets it, but too late for an entry-point plugin — see "Detecting xdist". |

`GPU_LEASE_PIN` renames the old unprefixed `ON_GPU` (`_core_test_backward.py:38,73,79`).
"Pin" states the semantics: it does not merely select a device, it disables the lease
protocol entirely. `GPU_LEASE_ON_GPU` stutters; `GPU_LEASE_DEVICE_ID` reads as a plain
selector and hides the bypass.

**No deprecation shim.** `grep -rn ON_GPU` over the repo finds only those three lines in
the fixture being deleted — nothing in `.ci/`, `.tune/`, `dockerfile/`, or `docs/`. It is a
developer-interactive knob only, so a clean rename is safe. It will, however, break shell
history and any personal scripts, so call it out in the PR description.

```python
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

import pytest

STRUCT_FLOCK = 'hhllh'
PAGE_SIZE = 4096
_RETRY_INTERVAL = 0.05


def _worker_count(config) -> int:
    """Number of xdist workers in this run; 0 when not running distributed.

    Deliberately NOT read from ``PYTEST_XDIST_WORKER_COUNT``. xdist sets that
    variable inside the worker process, and this module -- being a ``pytest11``
    entry-point plugin -- is imported during ``Config._preparse``, far earlier
    than the collection-time import the fixture used to live in. Reading it at
    module scope saw 0 and silently put every worker on GPU 0.

    ``config.workerinput`` is xdist's own interface for this (absent in the
    controller and in non-distributed runs) and is populated well before any
    fixture runs, so there is no ordering to get wrong.
    """
    workerinput = getattr(config, 'workerinput', None)
    if workerinput is None:
        return 0  # controller process, or plain `pytest` with no -n
    return int(workerinput['workercount'])


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
def gpu_id(request, worker_id):
    """Index of the GPU this worker owns for the duration of its session.

    Every mode announces its choice, not just the leasing one: without it there is
    no way to confirm that GPU_LEASE_PIN actually took effect either.
    """
    # GPU_LEASE_PIN wins over everything, distributed or not: "put all work on
    # GPU n" is a debugging override and should not depend on how pytest is run.
    pinned = _env_pin()
    if pinned is not None:
        _announce(request.config, f'{worker_id} uses GPU {pinned} (GPU_LEASE_PIN, no lease)')
        yield pinned
        return

    nworkers = _worker_count(request.config)
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
```

### Extending to other device classes

`gpu_device_class` is the seam. Three ways to set it, cheapest first:

```bash
GPU_LEASE_DEVICE_CLASS=xpu pytest -n 8 modules/whatever/tests
```

```python
# modules/<family>/tests/conftest.py -- retarget one suite
import pytest

@pytest.fixture(scope='session')
def gpu_device_class():
    return 'xpu'
```

...or a downstream plugin overriding it globally. Standard pytest fixture overriding: a
conftest-level definition shadows the plugin's for everything below it.

> **Alternative considered.** A callable factory — `gpu_device(device_class='cuda')`,
> invoked as `gpu_device('xpu')` — matches the literal "option with a default" shape but
> makes `gpu_device` a function rather than a string, so `f'{gpu_device}'` silently renders
> `<function _fmt at 0x...>` instead of failing. If the factory signature is preferred
> anyway, it is a one-line swap; the rest of the plan is unaffected either way, since no
> current call site consumes `gpu_device`.

Deltas from the original worth calling out to a reviewer:

- **`autouse=True` dropped** from the lockfile fixture (was `_core_test_backward.py:83`).
  Mandatory — an entry-point plugin loads into GPU-less suites too.
- **No pre-sizing / no truncation.** Drops the `seek`+`write` at `:87-89` and switches
  `'wb'` → `'r+b'` at `:94`, eliminating the truncation race.
- **`time.sleep`** on a full sweep of `BlockingIOError`, replacing the busy spin at `:95-104`.
- **`try/finally`** around the `yield`, so the lease is released even when the session
  errors out. The original leaked the lock on any exception.
- **`testrun_uid` dropped** — it was an unused parameter in both original fixtures.
- Three separate module-level fixture definitions collapse into one with internal branching.
- **`gpu_device` / `gpu_device_class` are new** — no equivalent existed. Nothing consumes
  them yet (`core_test_op_bwd` needs the int), so they carry no migration risk.

### Detecting xdist

**This bit an actual CI run — `bash .ci/run-test.sh 0 1 split` put all eight workers on
GPU 0.** Worth recording why, because the mistake is invisible in review.

The original fixture read the worker count at module scope:

```python
PYTEST_XDIST_WORKER_COUNT = int(os.getenv('PYTEST_XDIST_WORKER_COUNT', default='0'))
```

That was correct *where it used to live*. `_core_test_backward.py` is imported during
**collection**, deep into a worker's life, by which point xdist has long since populated
the worker environment. Moving the same line into a `pytest11` entry-point plugin moved
*when it runs*: entry-point modules are imported during `Config._preparse`, at the very
start of worker startup. The variable read as unset, the fixture took the no-xdist branch,
and every worker returned GPU 0 — no error, no warning, just eight processes on one device.

The fix is not to chase the ordering but to stop depending on it. xdist publishes the same
information on the config object, and it is populated before any fixture runs:

```python
def _worker_count(config) -> int:
    workerinput = getattr(config, 'workerinput', None)
    if workerinput is None:
        return 0  # controller process, or plain `pytest` with no -n
    return int(workerinput['workercount'])
```

`hasattr(config, 'workerinput')` is xdist's documented "am I a worker" test. Two
consequences:

- **Nothing in the plugin may be read from the environment at import time.**
  `GPU_LEASE_PIN` and `GPU_LEASE_DEVICE_CLASS` are read lazily inside their functions for
  the same reason, and because it makes them monkeypatchable in the self-tests.
- **`GPU_LEASE_PIN` is now checked before the xdist test.** "Put everything on GPU n" is a
  debugging override; it should not depend on whether `-n` was passed.

`test_leased_mode_assigns_distinct_gpus_and_creates_lockfile` is the regression guard, and
`_clear_lease_env` deletes `PYTEST_XDIST_WORKER_COUNT` specifically so the test fails if
the plugin ever reaches for it again.

### Live announcement of the lease

The lease is decided during **fixture setup**, and pytest captures setup output at
the fd level, replaying it only in the report section — and only for failing tests
unless `-rA` is passed. On a green run the GPU assignment was therefore invisible
until the run had already finished, which defeats the point of printing it.

`capsys.disabled()` is the documented way to suspend capture
(`how-to/capture-stdout-stderr.html`, "Accessing captured output from a test
function"). It is **not usable here**: every capture fixture — `capsys`,
`capfd`, `capteesys`, and the binary variants — is function-scoped, while `gpu_id`
is session-scoped, so requesting one raises `ScopeMismatch`. pytest ships no
session-scoped capture fixture.

`_announce` therefore calls the capture manager that `capsys.disabled()` itself
delegates to. Three consequences to keep in mind:

- **It is `_pytest.capture` internals, not public API.** Accepted deliberately; the
  remedy if it breaks is an upper pin on pytest, noted in `pyproject.toml`'s
  `dependencies`. The `getattr` guard degrades to a plain print rather than failing
  the run, so the symptom of a future break is the line reappearing only inside a
  replayed `Captured stderr setup` block — not a crash.
- **All three modes announce**, not just the leasing one. Without that there is no
  way to confirm `GPU_LEASE_PIN` took effect either. All three keep the
  `<worker_id> uses GPU <n>` prefix, so `awk '{print $4}'` still yields the index.
- **Alternative rejected:** `--capture=tee-sys` (pytest 5.4+) live-prints without
  any plugin code, but applies globally — `_core_test_backward.py` prints `tri_out=`
  and `adiff=` per test, so CI logs would balloon.

> **Unverified:** whether a worker's announcement reaches the controller's terminal
> live under `-n`. The reasoning is that execnet's popen gateway uses pipes for
> stdin/stdout (the channel) but leaves stderr inherited. This was not confirmed
> against pytest-xdist's docs or a real run. If it turns out workers cannot emit
> live, add an append-mode ledger file (`GPU_LEASE_LOG=<path>` + `tail -f`), which
> is immune to both capture and relay.

### `README.md`

Short: what it does, the three modes, the 1:1 rationale, install
(`pip install ./python/pytest-gpu-lease` from repo root), and the fixture table.

---

## Step 2 — Strip the origin

**`modules/flash/tests/_core_test_backward.py`** — delete lines 62-103 in full (the
`PYTEST_XDIST_WORKER_COUNT` / `STRUCT_FLOCK` / `PAGE_SIZE` constants and the whole
`if/elif/else` fixture block), plus these, which become dead:

| Line | Symbol | Why safe |
|---|---|---|
| 8 | `import sys` | its sole remaining use was `file=sys.stderr` inside the deleted fixture |
| 12 | `import fcntl` | used only in the deleted block |
| 13 | `import struct` | used only in the deleted block |
| 14 | `import itertools` | used only in the deleted block |
| 38 | `ON_GPU = os.getenv('ON_GPU', default=None)` | **outside the 62-103 range** — read only at `:73` and `:79`, both inside it |

Verify after editing; all must return nothing:

```bash
grep -n 'fcntl\|struct\.\|itertools\.\|ON_GPU\|\bsys\b' modules/flash/tests/_core_test_backward.py
grep -rn 'ON_GPU' modules/ .ci/ .tune/ docs/          # whole-repo sweep for stragglers
```

Do **not** delete `RECORD_ADIFFS_TO` / `USE_ADIFFS_TXT` at `:39-40` — unrelated, still used.

`import sys` (line 8) goes too — its only remaining reference was the
`file=sys.stderr` in the `torch_gpu` fixture being deleted.

**Do not touch `core_test_op_bwd`** (`:365-381`). Its `device: int | None` parameter must
stay an int — it feeds `torch.cuda.device(device)` at `:370` as well as
`f'cuda:{device}'` at `:371`. Call sites therefore consume `gpu_id`, not `gpu_device`.

---

## Step 3 — Update call sites

Two files, same mechanical pattern. Nothing else in the suite references these fixtures
(confirmed by grep across `modules/flash/tests/`).

1. Delete `gpufilelock,` and `torch_gpu,` from the `from _core_test_backward import (...)`
   list — the plugin supplies them now.
2. Rename the fixture parameter `torch_gpu` → `gpu_id` and update the
   `core_test_op_bwd(..., device=...)` argument.

**`modules/flash/tests/test_backward.py`** — imports at `:26-27`; six tests at
`:45`, `:62`, `:78`, `:99`, `:117`, `:138` with their `core_test_op_bwd` calls at
`:48`, `:65`, `:85`, `:102`, `:123`, `:142`:

```python
-def test_fast(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, ...):
+def test_fast(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, ...):
     bias_type = None
     args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, ...)
-    core_test_op_bwd(request, args, device=torch_gpu)
+    core_test_op_bwd(request, args, device=gpu_id)
```

**`modules/flash/tests/triton_tester.py`** — imports at `:19-20`; one test at `:41`,
call at `:44`.

**`modules/flash/tests/conftest.py` needs no change.** The `pytest11` entry point handles
discovery; its `collect_ignore` for `test_forward.py` is unrelated.

---

## Step 4 — Install wiring

### `requirements-dev.txt`

Append the path, **non-editable** so container builds can delete the source afterwards:

```
./python/pytest-gpu-lease
```

pip resolves relative local-path requirements against the **current working directory**,
not the requirements file's location. Confirm this during implementation
(`cd /tmp && pip install --dry-run -r <abs>/requirements-dev.txt` should fail to find the
path); if pip's behaviour differs from expectation, adjust Steps 4b/4c accordingly. Two
consequences follow:

### 4b. Docs must say "from the repo root"

- `docs/How To Run Tests.md:20` — the pre-requisite `pip install -r requirements-dev.txt`.
  This is a live trap: the TL;DR block at `:4-5` does `cd build-test` first.
- `.ci/README.md:180` — same line, same note.

### 4c. `.tune/lib/create_dockerfile.sh:56-60`

Copies only `requirements*.txt` into the image and pip-installs with cwd `/`, so a relative
path in `requirements-dev.txt` cannot resolve. Needs the plugin source and a matching cwd:

```dockerfile
 COPY aotriton.src/requirements*.txt /tmp/
+COPY aotriton.src/python/pytest-gpu-lease /tmp/python/pytest-gpu-lease
+WORKDIR /tmp
 RUN ${CELERY_WORKER_PYTHON} -m pip install -r /tmp/requirements-tuning.txt && \
     ${CELERY_WORKER_PYTHON} -m pip install -r /tmp/requirements-dev.txt && \
-    rm /tmp/requirements*.txt
+    rm -rf /tmp/requirements*.txt /tmp/python
```

Note this is a heredoc-generated Dockerfile — `\\` line continuations and `\$` escapes in
the surrounding script must be matched.

### 4d. `setup.py`

Add a comment at the `find_packages` call (`:39`) recording that switching to
`find_namespace_packages` would pull `python/pytest-gpu-lease/` into the `aotriton` wheel.

---

## Step 5 — Relocate `test_gpu_targets.py` (Minor Change)

Folded into this PR as an accidental discovery; goes under `## Minor Changes` in the PR
description per `PR.instructions.md:92-111`.

`modules/flash/tests/test_gpu_targets.py` tests `aotriton.gpu_targets`, a codegen module.
It touches no GPU and no flash source — it only `subprocess.run`s a CLI five times. It
landed there as collateral of the bulk `git mv test/ -> modules/flash/tests` in `e8499a0d`
(Modularization Step 5); its prior home was `test/test_gpu_targets.py` at the repo root.

```bash
git mv modules/flash/tests/test_gpu_targets.py python/test/test_gpu_targets.py
```

### Fix the stale `REPO_ROOT`

```python
-REPO_ROOT = Path(__file__).resolve().parent.parent
+REPO_ROOT = Path(__file__).resolve().parents[2]   # python/test -> python -> repo root
```

At the old `test/` location `parent.parent` genuinely was the repo root; the bulk move
silently repointed it at `modules/flash`. The value is functionally inert — it is only the
`cwd` handed to `subprocess.run`, and `python -m aotriton.gpu_targets` resolves from the
installed package regardless (neither the repo root nor `python/` contains an importable
`aotriton/`) — but leaving a two-level-wrong constant named `REPO_ROOT` is a trap.

### Update `python/test/conftest.py`

Its docstring advertises the suite as self-contained, exercising "fake, minimal kernels
... with NO dependency on the real flash sources under modules/". That stays true, but
`test_gpu_targets.py` shells out to the *real* `aotriton.gpu_targets` CLI rather than a
fake. Add a sentence so the charter matches the contents.

### No other references

`grep -rn test_gpu_targets .ci/ docs/ .tune/ .github/` returns nothing — no CI script or
doc names the file. It is picked up only by directory collection.

### CI coverage consequence

Today the file runs in CI only because `.ci/run-test.sh:93` collects the whole
`modules/flash/tests` directory. Nothing runs `python/test`, so the move would drop it
from CI — Step 6 closes that.

---

## Step 6 — Add a CPU-only unit-test CI pass

**Gate this step first.** `python/test`'s ~40 files have never run in CI. Before writing any
script, run them and triage:

```bash
python3 -m venv /tmp/ut && /tmp/ut/bin/python -m pip install -q -r requirements-dev.txt .
/tmp/ut/bin/python -m pytest python/test -q
```

If anything fails, those are **pre-existing** failures, not regressions from this PR. Report
the list before deciding — fix, `xfail` with a reason, or `--deselect`. Do **not** blanket-
deselect to make the pass green; a silently narrowed suite is worse than a known-red one.
This is the scope risk accepted when choosing to add the CI pass.

### `.ci/aotriton-self-test.sh` (new)

Named to stay clear of the existing `.ci/run-test.sh` (the GPU flash suite) — a
`run-unit-test.sh` sitting beside `run-test.sh` invites picking the wrong one.

CPU-only: no GPU, no ROCm, no built library. Deliberately does not source
`common-vars.sh`, which calls `rocm_agent_enumerator` and would require a GPU host.

```bash
#!/bin/bash
# CPU-only unit tests: the ATI code generator (python/test) and the GPU-lease
# pytest plugin's own suite. No GPU, no ROCm, no built library required.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${AOTRITON_UNITTEST_VENV:-${ROOT}/build-unittest-venv}"

# cd BEFORE pip: requirements-dev.txt carries a CWD-relative path
# (./python/pytest-gpu-lease, added in Step 4).
cd "${ROOT}"

[ -d "${VENV}" ] || python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -q -r requirements-dev.txt
"${VENV}/bin/python" -m pip install -q .   # the aotriton package; python/test imports aotriton.*

exec "${VENV}/bin/python" -m pytest python/test python/pytest-gpu-lease/tests -q "$@"
```

Why a dedicated venv rather than reusing the build venv: the build venv has `aotriton` but
no pytest, and `CMakeLists.txt:250-254` keeps it deliberately disposable and minimal. The
user's own env has pytest but no `aotriton`. Neither is a fit, and mutating either is worse
than a throwaway.

The `build-unittest-venv` name is deliberate: `.gitignore:4` is `/*build*/`, which already
covers any root-level directory containing "build". No `.gitignore` change needed.

**Useful side effect:** because `requirements-dev.txt` now installs the plugin, this pass
runs it against two GPU-less suites — which *is* the no-autouse regression test. The manual
check under Verification becomes redundant once this lands.

### `.ci/Makefile`

```make
+.PHONY: check-self-test
+
+# AOTriton's own CPU-only test suites. No GPU or ROCm needed, so this runs anywhere.
+check-self-test:
+	bash $(CI_DIR)aotriton-self-test.sh
+
-check-all: check-gpu check-runtime check-gpu-asan check-runtime-asan
+check-all: check-self-test check-gpu check-runtime check-gpu-asan check-runtime-asan
```

`check-self-test` goes first in `check-all`: it is the fastest and needs no hardware, so it
fails early instead of after a GPU image build. The target name maps 1:1 onto the script;
the existing targets are all `check-<thing>`, so it fits the convention.

### `.ci/README.md`

Add a `## AOTriton Self Test` section between `## Build for Testing` (`:52`) and
`## Run Tests` (`:65`), noting it needs no GPU, no ROCm and no built library, and is the
fastest pre-flight check. Say explicitly how it differs from `run-test.sh` directly below
it, since the two are adjacent in the README as well as in the directory.

### `pyproject.toml`

Extend `testpaths` (`:25`) so a bare `pytest` at the repo root covers both suites:

```toml
-testpaths = ["python/test"]
+testpaths = ["python/test", "python/pytest-gpu-lease/tests"]
```

The plugin's own `pyproject.toml` declares no `[tool.pytest.ini_options]`, so the repo-root
config stays the single inifile and there is no rootdir conflict.

---

## Verification

### Packaging isolation — the primary risk

```bash
python3 -m pip wheel . -w /tmp/wheel --no-deps
python3 -m zipfile -l /tmp/wheel/aotriton-*.whl | grep -i 'gpu.lease'   # must print NOTHING
python3 -m zipfile -l /tmp/wheel/aotriton-*.whl | grep -c 'aotriton/codegen'  # still > 0
```

### Build path unaffected

`aotriton.generate` runs at CMake *configure* time (`v3src/CMakeLists.txt:151`), so a
configure alone proves the `aotriton` package still installs and imports:

```bash
cmake -S . -B /tmp/bt -DAOTRITON_TARGET_ARCH=gfx942 -DAOTRITON_NOIMAGE_MODE=ON -G Ninja
/tmp/bt/venv/bin/python -c "import aotriton, aotriton.codegen; print('ok')"
/tmp/bt/venv/bin/python -c "import pytest_gpu_lease" 2>&1 | grep -q ModuleNotFound && echo "correctly absent from build venv"
```

### Plugin discovery

```bash
pip install -r requirements-dev.txt          # from repo root
pytest --fixtures modules/flash/tests 2>/dev/null | grep -E 'gpu_id|gpu_device|gpu_device_class|torch_gpu'
# all four listed, sourced from pytest_gpu_lease.plugin

# `--trace-config` is the documented way to confirm entry-point registration
# (writing_plugins.html, "Making your plugin installable by others").
pytest --trace-config 2>/dev/null | grep -i gpu_lease
```

### The announcement is live

The point of the whole exercise — the assignment must appear *while* the run is
going, not in the end-of-run report:

```bash
# Watch the line appear before the run finishes, not after.
pytest -n 4 modules/flash/tests/test_backward.py -k test_fast -v 2>&1 | ts | head -40
```

Failure signature to look for: the line showing up only under a
`Captured stderr setup` heading means capture was not suspended — see "Live
announcement of the lease".

### The plugin's own test suite

`python/pytest-gpu-lease/tests/test_plugin.py` — needs **no GPU**: every mode either
short-circuits to an int or exercises `fcntl` against a temp file. Use pytest's `pytester`
fixture (enable with `pytest_plugins = ['pytester']`) to run nested pytest sessions with
controlled environments.

Cases to cover:

| Case | Setup | Expect |
|---|---|---|
| No xdist | no `-n` flag, `PYTEST_XDIST_WORKER_COUNT` unset | `gpu_id == 0`, no lockfile created |
| Pinned | `GPU_LEASE_PIN=3` | `gpu_id == 3`, no lockfile created |
| Leased | `-n 2`, count 2 | two workers, two distinct ids, lockfile exists |
| Device class default | — | `gpu_device == 'cuda:0'` |
| Device class via env | `GPU_LEASE_DEVICE_CLASS=xpu` | `gpu_device == 'xpu:0'` |
| Device class via override | conftest redefines `gpu_device_class` | override wins over the env var |
| No autouse | a test requesting neither fixture | lockfile absent afterwards |
| Live announcement | any mode | line is in `result.errlines`, not in a replayed `Captured stderr` block |
| **Worker crash + restart** | `-n 2 --max-worker-restart 4`, one test calls `os._exit(139)` | replacement worker re-leases the freed GPU; run terminates |

```bash
pytest python/pytest-gpu-lease/tests -q
```

#### Worker crash + restart — the load-bearing case

This is not hypothetical. `.ci/run-test.sh:93` runs with `--max-worker-restart 9999`, and
`_core_test_backward.py:57-59` deliberately calls `os._exit(139)` on
`hipErrorIllegalAddress` and `torch.AcceleratorError`. Crash-and-restart is a routine CI
path, and it is the one path where the fixture's own teardown never runs — `os._exit`
bypasses `finally`, so **the OS reclaiming the `fcntl` record lock on process death is the
only thing that frees the page.**

The invariant under test: a replacement worker must acquire a GPU rather than spin forever.
If the lease mechanism were ever changed to something not auto-released on process death
(a PID file, a `filelock` sentinel, a lock in shared memory), the replacement would loop in
`itertools.cycle` — and with the new `time.sleep` it would hang *quietly* instead of
burning a core, which is worse. This test is the guard on that.

**Do not scrape stderr for the `uses GPU` line.** Under `-n`, worker output relay depends on
capture settings and is fragile. Have the nested tests append their own record to a shared
file whose path arrives by env var:

```python
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
```

Notes for the implementer:

- Guard the whole test with a generous `@pytest.mark.timeout(60)`, so a regression surfaces
  as a clear failure rather than a hung CI job. `pytest-timeout` is already a dev dep.
- The `autouse` recorder lives in the *nested* conftest, not the plugin — the plugin's own
  lockfile fixture must stay non-autouse.
- Which worker crashes is not controllable under xdist's default scheduling, and does not
  need to be: the assertions are structural.
- With N workers and N GPUs the freed page is the only one available, so the replacement is
  expected to retake the *same* index. Asserting the exact index would over-fit — the
  round-robin scan starts at 0 and takes the lowest free page, which happens to coincide
  here. Assert the pool invariant, not the identity.
- If `pytester`'s nested-xdist run proves awkward (plugin autoload interactions), fall back
  to driving a real `subprocess.run([sys.executable, '-m', 'pytest', ...])` against a
  generated temp directory. The ledger-file approach works identically either way.

### All three modes

Needs a built tree: `PYTHONPATH=<build>/install_dir/lib`.

```bash
# no xdist -> GPU 0
pytest modules/flash/tests/test_backward.py -k test_fast -x -q

# pinned -> every test on cuda:3
GPU_LEASE_PIN=3 pytest modules/flash/tests/test_backward.py -k test_fast -q

# leased -> N distinct "gwK uses GPU n" lines, no GPU repeated
pytest -n 8 modules/flash/tests/test_backward.py -k test_fast -q 2>&1 \
  | grep 'uses GPU' | awk '{print $4}' | sort | uniq -d   # must print NOTHING
```

### No-autouse regression

GPU-less suites must run clean and create no lockfile — the plugin auto-loads into them and
must stay inert until `gpu_id` is actually requested. Step 6's CI pass *is* this test, since
it installs the plugin and then runs two GPU-less suites:

```bash
bash .ci/aotriton-self-test.sh
find /tmp/pytest-of-$USER -name gpulock 2>/dev/null   # must print NOTHING
```

### Lease release on failure — on real hardware

The plugin's own suite covers this GPU-free (see "Worker crash + restart"). Confirm it once
against the actual flash suite, since that is where `exit_pytest()` fires for real:

```bash
# --max-worker-restart mirrors .ci/run-test.sh:93. Expect: restarts happen, run terminates,
# and no GPU index is ever held by two live workers at once.
pytest -n 4 --max-worker-restart 9999 modules/flash/tests/test_backward.py -k test_fast -q 2>&1 \
  | tee /tmp/lease.log | grep -E 'uses GPU|replacing crashed worker'
awk '/uses GPU/ {print $4}' /tmp/lease.log | sort | uniq -c   # a repeated index is fine ONLY after a restart
```

Two distinct failure modes to watch for, which the `try/finally` alone does not cover:

- **Clean teardown** (`try/finally`) — releases the page on an ordinary exception.
- **`os._exit(139)`** (`_core_test_backward.py:57-59`) — bypasses `finally` entirely; the
  kernel must reclaim the record lock. If a replacement worker ever hangs instead of
  picking up a GPU, this is the mechanism that broke.

---

## PR description mapping

Per `PR.instructions.md`. Steps 1-4 are the headline; Steps 5-6 are accidental discoveries
that ride along.

```markdown
## Major Changes

* [test] New `pytest-gpu-lease` plugin (`python/pytest-gpu-lease/`): leases one GPU per
  pytest-xdist worker, extracted from `modules/flash/tests/_core_test_backward.py` so every
  module's suite can share it. Fixtures: `gpu_id`, `gpu_device`, `gpu_device_class`,
  `torch_gpu`.
* [test] `ON_GPU` is renamed `GPU_LEASE_PIN`. **Breaking for shell history / personal
  scripts**; no in-repo callers.
* [ci] New `.ci/aotriton-self-test.sh` + `make -C .ci check-self-test`: CPU-only pass over
  `python/test` and the plugin's own suite. `python/test` was previously run by no CI job.

## Minor Changes

* [test] Move `test_gpu_targets.py` from `modules/flash/tests` to `python/test` — it tests
  `aotriton.gpu_targets`, needs no GPU, and was collateral of the bulk move in `e8499a0d`.
* [test] Fix `REPO_ROOT` in `test_gpu_targets.py`, stale since that move.
* [build] Comment in `setup.py` recording why `find_packages` must not become
  `find_namespace_packages`.
```

Add a `## Known Issues` entry for anything the Step 6 gate turns up red.

---

## Out of scope — flagged, do not fix here

- **`test_varlen.py:72` hardcodes `device='cuda'`** and never requests the fixture. Since
  `.ci/run-test.sh:93` runs `pytest -n $ngpus modules/flash/tests` over the whole directory,
  every xdist worker currently drives its varlen tests on GPU 0 while the leases sit idle.
  Pre-existing; the plugin makes it a two-line fix, but it changes runtime behaviour and
  belongs in its own PR. `test_forward.py:58` has the same shape, but `conftest.py:20`
  excludes it from directory collection, so it only bites standalone runs.
- **Pre-existing failures in `python/test`, if the Step 6 gate finds any.** That suite has
  never run in CI, so anything red there predates this PR. Triage and report rather than
  silently deselecting; fixing unrelated generator bugs is not this PR's job.
- **The flash suite already assumes `aotriton` is importable in the test env** (via
  `test_gpu_targets.py`, until Step 5 moves it), which `CMakeLists.txt:255` only guarantees
  for the disposable build venv. A second argument for `requirements-dev.txt` growing a
  `-e .` line later.
- **PyPI name availability for `pytest-gpu-lease` is unverified** — no network egress from
  the planning session. Worth checking if publication is ever on the table; irrelevant
  while the install is path-based.
- **No `pytest_addoption` / CLI options.** See the 1:1 rationale above.
