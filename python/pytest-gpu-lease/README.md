# pytest-gpu-lease

A pytest plugin that leases exactly one GPU to each `pytest-xdist` worker,
coordinating through `fcntl` byte-range locks on a lockfile shared by the run.
Activated automatically via the `pytest11` entry point once installed --
no `conftest.py` wiring required.

## Install

```bash
pip install ./python/pytest-gpu-lease   # from the repo root
```

## Modes

| Mode | Condition | Behaviour |
|---|---|---|
| Pinned | `GPU_LEASE_PIN` is set (checked first, xdist or not) | Every worker is pinned to `int(GPU_LEASE_PIN)`, no locking. |
| No xdist | no `config.workerinput` | `gpu_id` is `0`, no locking. |
| Leased | otherwise | Round-robin `fcntl` write-lock on one page per GPU, held for the worker's whole session. |

A lease is an exclusive, bounded, released claim -- exactly this acquire/yield/release
lifecycle. Oversubscription (running several tests on one GPU) is **not supported** and
there is no CLI knob for it: it invites memory pressure and races in the runtime /
driver / firmware / VBIOS. The mapping is strictly 1:1 worker-to-GPU.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GPU_LEASE_PIN` | unset | Bypass leasing; pin every worker to this GPU index. |
| `GPU_LEASE_DEVICE_CLASS` | `cuda` | Accelerator class `gpu_device` formats with. |
| `GPU_LEASE_LOCKFILE` | derived from `tmp_path_factory` | Path of the shared lock file. Set it to run the watchdog (below), which cannot predict the derived path. Also enables the per-worker stack dump. |

`PYTEST_XDIST_WORKER_COUNT` is **not** consulted. `pytest-xdist` sets it inside the
worker process, but this module is imported at `pytest11` entry-point time -- earlier
than that -- so reading it saw `0` and put every worker on GPU 0. The worker count comes
from `config.workerinput`, which is populated before any fixture runs. Nothing here is
read from the environment at import time.

## Watchdog

A test whose GPU kernel never retires wedges its worker for good: the device sync
never returns, so no in-process timeout can fire -- `pytest-timeout`'s signal method
needs the main thread to reach a bytecode boundary, and a thread parked in a driver
call never does.

`pytest_gpu_lease.watchdog` detects that from outside. Each worker stamps
`time.monotonic_ns()` into its own lease page before every test and zeroes it after;
the watchdog polls the file and, when a page has gone `--threshold` seconds without a
fresh stamp, sends SIGTERM (a stack dump, via `faulthandler`), waits `--grace`, then
SIGKILLs. Signals go through a `pidfd`, so a reissued pid cannot be hit.

```bash
export GPU_LEASE_LOCKFILE=/dev/shm/gpu_lease.$$        # both sides must agree
python -m pytest_gpu_lease.watchdog --lockfile "$GPU_LEASE_LOCKFILE" --workers 4 &
watchdog_pid=$!
trap 'kill -s TERM "$watchdog_pid"; wait "$watchdog_pid"' EXIT
pytest -n 4 ...
```

**It is a service and never stops on its own.** Whoever starts it stops it, with
SIGTERM; every rule for inferring "the pass is over" from the lock file is a guess,
and a wrong guess silently leaves the rest of a long run unprotected. On SIGTERM it
removes the lock file and any stack dumps beside it. SIGKILL is the one case it
cannot clean up after.

`--lockfile` is required rather than read from the environment, so `ps` shows which
file each watchdog is watching -- the only way to tell a stale one from the live one.
See `--help` for `--threshold`, `--grace` and `--poll_interval`.

## Fixtures

| Fixture | Type | Description |
|---|---|---|
| `gpu_id` | `int` | Index of the GPU this worker owns for the duration of its session. |
| `gpu_device` | `str` | `gpu_id` formatted as a device string, e.g. `'cuda:3'`. |
| `gpu_device_class` | `str` | Accelerator class used to build `gpu_device`. Defaults to `'cuda'`; override in a `conftest.py` to retarget a suite at another backend (e.g. `'xpu'`). |
| `torch_gpu` | `int` | Back-compat alias for `gpu_id`. |

All fixtures are session-scoped (per-worker process under `xdist`) and none is
`autouse` -- the plugin stays inert until a test actually requests one of them, so it
is safe to auto-load into GPU-less suites.

None of them depends on an `xdist` fixture either. Everything needed from `xdist`
comes from `config.workerinput`, which is simply absent when it is not loaded, so
`gpu_id` still resolves under `-p no:xdist` or `PYTEST_DISABLE_PLUGIN_AUTOLOAD`
(falling back to GPU 0, announced under the `master` label).

### This is infrastructure: do not prune "unused" fixtures

Every fixture in the table above is public API, and the table is the contract. A fixture
having no caller in this repository right now is **not** evidence that it is dead --
this package exists to serve suites that have not been written yet, in modules that do
not exist yet. Do not delete one because a static "unused symbol" sweep flags it, and do
not narrow the surface to whatever the flash suite happens to use today. Removing an
entry is a deliberate API break; adding one is cheap.
