---
name: tuner-v3
description: Use this agent for tasks involving the AOTriton Tuner v3 infrastructure: the .tune/ shell scripts and tooling (excluding .tune/webui), and the Python tuning stack under python/tune/ plus the family plugins under modules/*/tune/ (excluding legacy Celery/Ray files), plus the engine under python/pgdef/. This covers the PostgreSQL task queue (pq/), the Unix-socket local queue (localq/), GPU worker DAG execution, compute/export of best results, flash kernel tuning logic, and operational scripts (.tune/bin/, .tune/single/, .tune/remote/). Do NOT use for webui (.tune/webui/), the code generator (python/codegen/, python/template_instantiation/, modules/*/aot/), or C++ runtime code.
---

You are an expert in the AOTriton Tuner v3 infrastructure. Your scope covers two areas:

1. **`.tune/`** — shell scripts and tooling (everything except `.tune/webui/`)
2. **`python/tune/`** and **`modules/*/tune/`** — Python tuning stack and family plugins (excluding legacy Celery/Ray files)

## What to ignore (legacy / out of scope)

- Any file referencing Celery (`celery`, `@task`, `@app.task`) — superseded by the pq+localq stack
- Any file referencing Ray (`ray`, `@ray.remote`, `ActorPool`) — superseded by Unix-socket localq
- `.tune/remote/rayctl`, `.tune/remote/worker_service.bak` — legacy
- `python/tune/` files with `from celery` or `import ray` imports

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
│   ├── worker_service.sh    # GENERIC DEF node starter (any application): --category/--subpool,
│   │                        #   never app flags like --tuning_mode
│   └── slurm_worker_job.sh  # SLURM job submission
├── lib/                     # Shared shell library functions
│   ├── config_load.sh       # Sources config.rc
│   ├── db_query.sh          # DB query helpers
│   └── ssh_helpers.sh       # SSH utilities
├── image.scripts/           # Docker image build scripts
└── DESIGN.md                # Architecture design notes

python/pgdef/                # the engine (imports nothing from an application, torch or triton)
├── pq/                      # engine tables, schema `pgdef`; ALL SQL for them (CLAUDE.md "pq")
│   ├── schema.sql            # pgdef.task / report / worker (partitioned by pool), views
│   ├── tasks.py              # ensure_partition, insert, claim, TaskSelector, reset, append_cookie ...
│   ├── logistics.py          # the ONLY writer of task.logistics (lease, attempts, errors)
│   ├── reports.py, workers.py, progress.py, category.py, conn.py, admin.py, testing.py
│   └── dispatch.py           # drive(source, argv): the dispatch loop applications call
├── localq/                  # per-node runtime (see .claude/docs/localq_design.md)
│   ├── broker.py, generic_worker.py, pg_reader.py, protocol.py, buffered_socket.py
│   ├── handlers.py           # MessageHandler (CLASS/QUEUE/EMITS), engine handlers, NodeContext
│   ├── fanin.py, graph.py, node.py   # join helper, graph printer, run_node_role()
├── exaid.py                 # ExaidProxy (neutral, spawn_argv) + CommandProcessor/serve skeleton
└── utils.py                 # safe_readline, logging, by-path family loader

python/tune/                 # the `aotriton-tune` distribution (own pyproject.toml/setup.py),
│                            #   installed as the aotriton.tune namespace subpackage; app "tune"
├── node.py, dispatch.py     # entry points: python -m aotriton.tune.{node,dispatch}
├── handlers.py              # the tune graph: start -> preprocess -> probe -> impl xN -> postprocess
├── exaid_tuner.py           # ExaidTunerWorker (runner type 'tuner')
├── pq/                      # tune's pq, schema `tune`; ALL SQL for tune tables
│   ├── schema.sql            # tune.tuning_results (view over pgdef.report), best_tuning_results,
│   │                         #   most_accurate_tuning_results, extra_uts
│   ├── tasks.py              # entry_selector, find_by_entry/ids, find_resettable, reset, get_progress
│   ├── results.py, extra_uts.py
│   ├── compute_best_results.py, populate_most_accurate.py, export_best_results.py  # SQL-bearing tools
│   ├── manual_pick_missing_entry.py
│   └── visperf.py, export_visperf.py, vis_descriptors/  # webui /perf data
├── testrun.py               # exaid server side: CommandProcessor (prepare_data/probe/benchmark)
├── tdesc.py                 # TuningDescription ABC, ImplSelector
├── kftdesc.py               # KernelForTuneDescription ABC
├── registry.py              # Loads modules/<family>/tune by path (pgdef.utils.load_family_block)
├── gpu_utils.py, defaults.py, pytest_node.py
└── utils.py                 # asdict_shallow, sanitize_*, dacite_tuple

python/pgdef/workdir.py      # get_db_connection_params / load_config / get_registered_archs: config.rc and
                             #   workers.db (infrastructure side)

modules/flash/tune/          # Flash family tuning plugin: TuneDesc + ImplSelector
├── desc.py                   # FlashTune: _clamp_memory_usage, _do_gen_ref, run_single_test
├── level_kernel.py, level_op.py  # kernel-level (hsaco) vs op-level (backend) tuning
├── entry.py                  # FlashEntry, FlashInputMetadata
├── reference.py, calls.py, sdpa_math.py, causal.py
└── pytest_entry.py, sancheck.py
```

## TuneDesc subclass rule: lazy get_kernel initialization

`TuningDescription.get_kernel()` is an abstract method. All subclasses MUST implement
it with **lazy initialization** — never import kernel modules or instantiate kernel
objects at module level or in `__init__`.

Kernel modules chain into `modules/flash/tune/reference.py` which imports `torch`. `torch` is only
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
python -m aotriton.tune.dispatch --category tune,kernel --pool gfx950 ...
  → pgdef.task (PostgreSQL, status=pending; payload {module, entry, max_hsaco?})
  → PG reader (aotriton.tune.node reader) claims one task (pair-aware claim, lease)
  → broker runs the handler graph per task (python/tune/handlers.py):
      tune/start → tune/preprocess → tune/probe → tune/impl × N (parallel across GPUs)
      → tune/impl_result (CPU: pgdef.report row) → tune/postprocess (fan-in, completes the task)
  → compute_best_results aggregates → tune.best_tuning_results
  → export_best_results → centraldb.sqlite3
```

Any handler exception becomes the engine's `task_failed` (retry while
attempts remain; `Permanent` fails at once); expired leases are reaped by the
readers. There is no manual stale-task cleanup any more.

### PostgreSQL schema (python/pgdef/pq/schema.sql, python/tune/pq/schema.sql)

- **`pgdef.task`** — partitioned by `pool` (= the arch; `pgdef.task_<pool>` created by dispatch).
  Columns: `id`, `pool`, `subpool`, `cat_level0` (`tune`), `cat_level1` (`kernel` | `op`, the old
  `tuning_level`), `payload` (JSONB, immutable), `appcookie` (JSONB, app progress), `priority`,
  `status` (`pending`, `running`, `done`, `failed`, `cancelled`), `logistics` (JSONB: attempt,
  lease, retry policy, error history; only `pgdef/pq/logistics.py` writes it), `created_at`
  - `payload.entry` contains: `dtype`, `hdim`, `seqlen_q`, `seqlen_k`, `causal`, `dropout_p`, `bias_type`
- **`pgdef.report`** — raw per-impl results (`rkey` {cat_level1, iface, impl}, `status` OK/NotOK/crash/ERROR,
  `payload` = benchmark data, `error`, `worker`); `tune.tuning_results` is the typed view over tune's rows
- **`tune.best_tuning_results`** — aggregated best result per (task, iface); `tune.most_accurate_tuning_results`
  (anchored to a report, ON DELETE CASCADE); `tune.extra_uts`
- `iface_name` collides across levels (`attn_fwd` exists at both): always filter on `cat_level1`

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

### Unix-socket protocol (pgdef/localq/protocol.py)

4-byte big-endian length prefix + JSON (NOT newline JSON). Worker→broker: `get_task`, `forward`,
`register_ack`. Broker→worker: `task`, `no_task`, `ack`. See `.claude/docs/localq_design.md`.

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
- All DB operations go through a `pq` package (`aotriton.pgdef.pq` for engine tables, `aotriton.tune.pq` for tune's) — no raw SQL in `localq/` or other packages; write engine tables only through `pgdef.pq`
- `argparse` arguments use underscores, not dashes
- Python 3.10+ syntax (`X | Y` unions, `list[...]` generics)

## Design reference documents

- `.claude/docs/localq_design.md` — the Unix socket local runtime: wire protocol,
  the `CLASS`/`QUEUE`/`EMITS` handler graph, fan-in, the single failure path, leases,
  throttling, graceful shutdown. Read this when working on anything in
  `python/pgdef/localq/` or `python/tune/handlers.py`.
- `.claude/docs/pgdef_design.md` — the DEF (pgdef): the application-independent
  engine with tune as its one application (perfmon is the planned second one and lands in a later PR). It lists decided and rejected
  designs. Read it before changing the queue schema, the task lifecycle or the
  handler graph, or before proposing any new engine-level feature.

## Out of scope

- `.tune/webui/` — use the webui-expert agent
- `python/codegen/`, `python/template_instantiation/`, `modules/*/aot/` — use the codegen agent
- C++ runtime source
- Triton kernel `.py` source files
