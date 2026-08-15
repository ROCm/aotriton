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
| No xdist | `PYTEST_XDIST_WORKER_COUNT == 0` | `gpu_id` is `0`, no locking. |
| Pinned | `GPU_LEASE_PIN` is set | Every worker is pinned to `int(GPU_LEASE_PIN)`, no locking. |
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
| `PYTEST_XDIST_WORKER_COUNT` | `0` | **Not ours** -- set by `pytest-xdist`. Read, never written. |

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
