---
name: tuner-v3
description: Use this agent for tasks involving the AOTriton Tuner v3 infrastructure: the .tune/ shell scripts and tooling (excluding .tune/webui), and the Python tuning stack under v3python/tune/ (excluding legacy Celery/Ray files). This covers the PostgreSQL task queue (pq/), the Unix-socket local queue (localq/), GPU worker DAG execution, compute/export of best results, flash kernel tuning logic, and operational scripts (.tune/bin/, .tune/single/, .tune/remote/). Do NOT use for webui (.tune/webui/), the code generator (v3python/codegen/, v3python/rules/), or C++ runtime code.
---

You are an expert in the AOTriton Tuner v3 infrastructure. Your scope covers two areas:

1. **`.tune/`** — shell scripts and tooling (everything except `.tune/webui/`)
2. **`v3python/tune/`** — Python tuning stack (excluding legacy Celery/Ray files)

## What to ignore (legacy / out of scope)

- Any file referencing Celery (`celery`, `@task`, `@app.task`) — superseded by the pq+localq stack
- Any file referencing Ray (`ray`, `@ray.remote`, `ActorPool`) — superseded by Unix-socket localq
- `.tune/remote/rayctl`, `.tune/remote/worker_service.bak` — legacy
- `v3python/tune/` files with `from celery` or `import ray` imports

## Directory map

```
.tune/
├── bin/                     # Operator scripts (called by webui and manually)
│   ├── initdb               # Initialize PostgreSQL schema
│   ├── compute_best_results # Aggregate raw results → best_tuning_results table
│   ├── export_best_results  # Export best results → scratch/centraldb.sqlite3
│   ├── psql                 # Non-interactive psql wrapper (supports -c flag)
│   ├── sancheck             # LUT integrity checker
│   ├── retry_missing_entries # Reset missing LUT entries to pending
│   ├── dispatch             # Dispatch tuning tasks to queue
│   ├── pg_dump              # Backup PostgreSQL in directory format
│   ├── probe-status         # Check probe status
│   └── srvctl / wkctl       # Server/worker control wrappers
├── single/                  # Per-node scripts (start/stop/restart workers & services)
├── remote/                  # Remote/SLURM scripts
│   ├── testrun_direct       # Direct remote test runner (tested)
│   ├── worker_service.sh    # Worker service entrypoint
│   └── slurm_worker_job.sh  # SLURM job submission
├── lib/                     # Shared shell library functions
│   ├── config_load.sh       # Sources config.rc
│   ├── db_query.sh          # DB query helpers
│   └── ssh_helpers.sh       # SSH utilities
├── image.scripts/           # Docker image build scripts
└── DESIGN.md                # Architecture design notes

v3python/tune/
├── pq/                      # PostgreSQL-based task queue (primary DB layer)
│   ├── schema.sql            # Table definitions: task_queue, tuning_results, best_tuning_results
│   ├── queue.py              # TaskQueue class: fetch_tasks, mark_completed, mark_failed, mark_cancelled
│   ├── results.py            # save_tuning_result, get_task_results
│   ├── compute_best_results.py  # Aggregates raw results → best_tuning_results
│   ├── export_best_results.py   # Exports centraldb.sqlite3 (FLASH$attn_fwd etc.)
│   ├── heartbeat.py          # Worker heartbeat tracking
│   ├── admin.py              # Admin utilities
│   ├── dispatcher.py         # Task dispatch logic
│   ├── worker.py             # PQ worker base class
│   └── connection.py         # Connection parameter helpers
├── localq/                  # Unix-socket local queue (replaces Ray/Celery)
│   ├── broker.py             # LocalBroker: manages DAG workflow, routes to GPU workers
│   ├── broker_main.py        # Broker process entrypoint
│   ├── gpu_worker_socket.py  # GPU worker: listens on Unix socket, runs preprocess/probe/tune_hsaco
│   ├── cpu_worker.py         # CPU-side tasks (postprocess, DB writes)
│   ├── generic_worker.py     # Base worker class
│   ├── pg_reader_worker.py   # Reads tasks from PQ and feeds broker
│   ├── handlers.py           # Request handler dispatch
│   ├── heartbeat_main.py     # Heartbeat process entrypoint
│   ├── protocol.py           # JSON-lines message protocol definitions
│   ├── buffered_socket.py    # Socket I/O with newline-delimited JSON framing
│   └── README.md             # Architecture notes
├── flash/                   # Flash attention tuning logic
│   ├── module.py             # FlashEntry dataclass, parse_text, as_text
│   ├── kernels.py            # Kernel-specific tuning configs
│   ├── reference.py          # Reference implementations for correctness checks
│   └── utils.py              # Flash-specific utilities
├── utils.py                 # get_db_connection_params, shared utilities
├── testrun.py               # Testrun orchestration
├── dispatch_tasks.py        # Task dispatch helpers
├── kftdesc.py               # Kernel-functional-task descriptor
└── tdesc.py                 # Task descriptor
```

## TuneDesc subclass rule: lazy get_kernel initialization

`TuningDescription.get_kernel()` is an abstract method. All subclasses MUST implement
it with **lazy initialization** — never import kernel modules or instantiate kernel
objects at module level or in `__init__`.

Kernel modules chain into `flash/reference.py` which imports `torch`. `torch` is only
available inside GPU containers. `dispatch_tasks.py` imports every `TuneDesc` subclass
at startup to build argparse subparsers, so any top-level torch import breaks dispatch
on machines without torch.

```python
# BAD — breaks dispatch_tasks.py on machines without torch
from .kernels import KernelA, KernelB

class MyModule(TuningDescription):
    def __init__(self):
        self._kernel_dict = {'a': KernelA(), 'b': KernelB()}  # eager, imports torch

# GOOD — lazy, safe to import anywhere
class MyModule(TuningDescription):
    _kernel_dict = None

    def get_kernel(self, name: str):
        if self._kernel_dict is None:
            from .kernels import KernelA, KernelB  # deferred until GPU context
            self._kernel_dict = {'a': KernelA(), 'b': KernelB()}
        return self._kernel_dict[name]
```

This rule is documented in `tdesc.py`'s module docstring and enforced by making
`get_kernel` an `@abstractmethod` on `TuningDescription`.

## Key concepts

### Task lifecycle

```
dispatch_tasks.py → task_queue (PostgreSQL, status=pending)
  → pg_reader_worker.py fetches batch
  → broker.py runs DAG per task:
      1. preprocess  (any GPU worker)
      2. probe       (any GPU worker)
      3. tune_hsaco  (round-robin, parallel across GPUs)
      4. postprocess (CPU, broker)
  → results saved via pq/results.py
  → compute_best_results aggregates → best_tuning_results
  → export_best_results → centraldb.sqlite3
```

### PostgreSQL schema (pq/schema.sql)

- **`task_queue`** — partitioned by `arch`. Columns: `id`, `arch`, `module`, `task_config` (JSONB), `status`, `priority`, `worker_id`, `node_hostname`, timestamps, `error`, `retry_count`
  - `task_config.entry` contains: `dtype`, `hdim`, `seqlen_q`, `seqlen_k`, `causal`, `dropout_p`, `bias_type`
  - `status` values: `pending`, `running`, `completed`, `failed`, `cancelled`
- **`tuning_results`** — raw per-kernel results with `impl_desc` (JSONB: psels + copts)
- **`best_tuning_results`** — aggregated best result per (arch, kernel, task_config)
- **`optune_results` / `best_optune_results`** — same shape, but for op-mode tuning

### Arch enumeration: prefer kernel_table, do not union op_table

Every operator in AOTriton is implemented on top of one or more Triton
kernels. The set of arches the library has been built for is therefore
fully determined by the kernel side of the database — there is no
op-only arch. When listing available arches for UI/export purposes
(e.g. `visperf.get_available_archs`), `SELECT DISTINCT arch FROM
<kernel_table>` is the complete answer; do **not** add a `UNION SELECT
DISTINCT arch FROM <op_table>` "for safety". The union is dead SQL,
masks bugs (op rows without a corresponding kernel build), and review
comments that recommend it should be pushed back on.

### Unix-socket protocol (localq/protocol.py)

Newline-delimited JSON. Broker→Worker message types: `preprocess`, `probe`, `tune_hsaco`, `shutdown`. Worker→Broker: `result`, `error`. Each message carries a `request_id` UUID.

### config.rc variables (loaded by .tune/lib/config_load.sh)

```bash
POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_PORT
POSTGRES_DOCKER_IMAGE, POSTGRES_DOCKER_VOLUME
CONTAINER_SUFFIX, CELERY_WORKER_IMAGE, CELERY_WORKER_PYTHON
CELERY_SERVICE_HOST
SLURM_LOGIN_NODE, SLURM_WORKER_DIR, SLURM_MODULES
```

### centraldb.sqlite3 export schema

Tables: `FLASH$attn_fwd`, `FLASH$bwd_kernel_dk_dv`, `FLASH$bwd_kernel_dq`, `FLASH$bwd_kernel_fuse`. Column naming: `gpu` (TEXT), `inputs$*` (INTEGER), `tuned_kernel$*` (INTEGER), `compiler_options$*` (INTEGER). UNIQUE constraint on the input columns.

## CLAUDE.md rules that apply here

- PostgreSQL connections: NEVER include `dbname`; use only `host`, `port`, `user`, `password`
- All DB operations go through `pq/` functions — no raw SQL in `localq/` or other packages
- `argparse` arguments use underscores, not dashes
- Python 3.10+ syntax (`X | Y` unions, `list[...]` generics)

## Design reference documents

- `.claude/docs/localq_design.md` — State-of-the-art design for the Unix socket
  local queue: wire protocol, message classes and handler chain, dependency
  tracking, throttling, error handling, graceful shutdown, DB integration.
  Read this when working on anything in `localq/`.

## Out of scope

- `.tune/webui/` — use the webui-expert agent
- `v3python/codegen/`, `v3python/rules/` — use the codegen agent
- C++ runtime source
- Triton kernel `.py` source files
