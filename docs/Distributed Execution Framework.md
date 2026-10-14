# Distributed Execution Framework (pgdef)

> Status: implemented (phases P0-P5 of the plan) but not yet exercised against a live
> PostgreSQL server or a GPU. The CPU-only tests pass; run the `pg`-marked tests
> against a disposable server (`aotriton.pgdef.pq.testing`) before trusting the SQL,
> and canary one arch against the previous tuning results before a campaign.

pgdef runs many independent GPU tasks across a fleet of machines. It uses
PostgreSQL as the task queue and a small per-node runtime to execute each task.
It knows nothing about tuning or performance measurement; those are
**applications** built on top of it. pgdef is a library: applications call it,
and it never loads them.

| Application | Package | What one task does |
|---|---|---|
| tune | `python/tune` | Benchmark every kernel/op implementation for one entry, and record the timings |

Perfmon (measuring one entry on a released AOTriton build) is the planned second
application. It lands in a later PR; nothing of it is in this tree.

## Design principles

* **Keep it simple.** A deployment lives for one campaign, which is less than a
  month, and the design changes almost every release. We prefer an agile system
  that sometimes needs an operator to reset rows over a polished one that takes
  months to build.
* **No database migrations.** Each release starts on a fresh database. Results
  leave through the application's export (LUTs, perf reports).
* **Tasks are independent.** A task never waits for another task. All
  dependencies are inside one task, on one node.
* **Prefer jsonb to new columns.** Anything that is not needed for routing
  lives in jsonb, so new features need no schema change.

## Tasks

A task is one row in `pgdef.task`:

| Column | Meaning |
|---|---|
| `pool`, `subpool` | Where it may run. `pool` is the arch (`gfx950`). `subpool` is an optional group of machines inside it (`MI355`) |
| `cat_level0`, `cat_level1` | The **category**: which application runs it (`tune`), and optionally which type of worker within it (`kernel` or `op`) |
| `payload` | The application's input (jsonb). Never changes after dispatch |
| `appcookie` | The application's own progress state (jsonb, may be NULL). pgdef stores it and hands it back on every claim without reading it. Applications write it with `append_cookie` (add items to a list) and `replace_cookie` (rewrite it whole). Kept across retries |
| `logistics` | pgdef's own bookkeeping (jsonb): attempts, lease, error history |
| `priority`, `status` | Scheduling. Status is `pending`, `running`, `done`, `failed` or `cancelled` |

`id` identifies a task. Raw results go to `pgdef.report`, one row per tuned
impl or measured backend, and each row points back to its task. Both tables are
partitioned by `pool`, which is why their primary keys are `(id, pool)`.

**Pairs.** `(pool, subpool)` and the category `(cat_level0, cat_level1)` follow the same
rule. A task with an empty second field is generic, and any worker of that pool
or application may take it. A worker takes generic tasks plus the tasks whose
second field matches its own.

The first level of the category *is* the application's name: everything in
category `tune,…` belongs to the application `aotriton.tune`. So a tool that
takes `--category` never also needs `--apps`.

The category is one 2-D value everywhere: the `cat_level0`, `cat_level1` columns in
PostgreSQL, a `tuple[str, str | None]` in Python, and a comma-separated
`--category tune,kernel` on the command line. More levels can be added later as
`cat_level2`, and so on.

```
dispatch  gfx950          -> any gfx950 machine
dispatch  gfx950 / MI355  -> only MI355 machines
```

Identical tasks are allowed. To redo a task, reset its status instead of
dispatching it again.

## Life of a task

```
pending ──claim──> running ──> done
   ^                 │  └────> failed      (an error, or a lease that expired)
   └─ retry (opt-in) ┘
   └───── reset (operator) <── done / failed / cancelled
```

* **Claim.** A worker takes the highest-`priority` pending task that matches its
  pool/subpool and category. The claim uses `FOR UPDATE SKIP LOCKED`.
* **Lease.** The claiming worker renews a lease while it runs the task. If the
  worker or its node dies, the lease expires and other workers mark the task
  `failed`. There is no manual cleanup of stuck rows.
* **Retry: supported, but unused.** By default a task gets one attempt
  (`max_attempts` = 1), so any failure or expired lease ends in `failed`, and
  redoing it is an operator reset. pgdef can retry automatically if a task is
  dispatched with a larger budget (`--max_attempts N`, or the application's
  default), but tune does not do this today.
* **Reset.** An operator reset clears `appcookie` by default, so the task is
  redone from scratch. Reset with `keep_appcookie` lets the application skip
  work it already finished.

## On a node

Each node runs **one application**. It has one broker and three types of
worker, all connected through a Unix socket:

```
PG readers (one task each) ──> broker ──> GPU workers ──> subprocess runner per GPU (exaid)
                                     └──> CPU workers ──> write results to PostgreSQL
```

A task runs as a small graph of **handlers**. Each handler's return value is
the next message. There is no DAG language. Instead, every handler declares
what it may emit:

```python
class ProbeHandler(MessageHandler):
    CLASS = 'tune/probe'
    QUEUE = 'gpu'
    EMITS = ('tune/impl', 'tune/postprocess')
```

`python -m aotriton.pgdef.localq.graph tune` prints the whole graph. If any
handler raises, the task fails through one common path.

GPU work runs in a supervised child process (**exaid**) that speaks a line
protocol: send a command, read back `OK <json>` or an error. A crash or a
timeout restarts the child without killing the worker.

A node may run zero GPU workers or zero CPU workers, for an application that
needs only one type.

## Writing an application

An application is an ordinary program that uses pgdef as a library. It
provides three entry points, as modules of `aotriton.<app>`:

| Entry point | What it does |
|---|---|
| `aotriton.<app>.node` | Starts one node process (broker, GPU/CPU worker, or reader), passing the app's handlers and runners to `pgdef.localq.node.run_node_role` |
| `aotriton.<app>.dispatch` | Turns the app's CLI options into tasks, through `pgdef.pq.dispatch.drive` |
| `aotriton.<app>.manifest` | Prints settings for the `.tune` scripts: image, container environment, install directories |

```python
# python/tune/node.py
run_node_role(app='tune', handlers=HANDLERS, start=TuneStartHandler,
              runners={'tuner': ExaidTunerWorker}, enrich=enrich, argv=sys.argv[1:])
```

`.tune/remote/worker_service.sh --category tune,kernel` starts `python -m aotriton.tune.node <role>`.
Despite its location under `.tune/`, `worker_service.sh` is the generic node
starter for every application. It takes only engine options (`--category`,
whose first level is the application, and `--subpool`), never an
application's own flags such as tune's `--tuning_mode`.

The `.tune` shell scripts learn everything application-specific from the
application's `MANIFEST` dict (`python/<app>/manifest.py`), one key at a time:

```bash
python3 -m aotriton.tune.manifest deploy.role
python3 -m aotriton.tune.manifest install.subtree --arch gfx950 --cat_level1 op
```

(`.tune/lib/manifest.sh` wraps this as `app_manifest` and `require_app`, the one
validation each script keeps.) The manifest holds where the build is installed, what to fetch from the build
node, the host role that runs the application, and the category levels it accepts.
A machine's subpool (MI355) is host inventory: a column of `workers.db`, set on
the webui's Workers page, and passed to its readers as `--subpool`.

## Database

Each layer owns one `pq` package and one PostgreSQL schema, and no SQL exists
outside a `pq` package:

| Layer | Package | Tables |
|---|---|---|
| engine | `aotriton.pgdef.pq` | `pgdef.task`, `pgdef.report`, `pgdef.worker` |
| tune | `aotriton.tune.pq` | `tune.*`, e.g. `tune.best_tuning_results` |

An application's `pq` reads engine tables directly when it needs to (for
example, typed views over its rows of `pgdef.report`), and writes them only
through `pgdef.pq`.

**Naming.** Database columns, and keys stored in database jsonb, use the
category names `cat_level0`/`cat_level1` everywhere, including in an
application's own tables. Application code and its tools use the application's
own terms instead. Tune says `tuning_level` and `--tuning_mode kernel|op`, not
`cat_level1` or `--category tune,kernel`. Each application's `pq` translates
between the two and documents its mapping, e.g. "tune stores its tuning level
as `cat_level1`".

`.tune/bin/initdb`, which is also behind the webui's Init DB button, creates
every schema for every application. Each `pq` ships an idempotent
`schema.sql`, and pgdef's runs first. `--recreate` drops and recreates them all.

## Common operations

```bash
# Dispatch: each application has its own dispatch command (tune today)
python -m aotriton.tune.dispatch --category tune,kernel --pool gfx950 --workdir <workdir> --module flash <tune options>

# Start a node's processes (normally done by .tune/remote/worker_service.sh --category tune,kernel)
python -m aotriton.tune.node reader --pool gfx950 --subpool MI355 --category tune,kernel
```

Progress and worker health come from the `queue_progress` and `worker_health`
views, which the webui shows.

## Code layout

```
python/pgdef/pq/       engine schema and SQL (tasks, logistics, reports, workers, dispatch driver)
python/pgdef/localq/   broker, workers, handler base classes, fan-in helper, run_node_role
python/pgdef/exaid.py  supervised subprocess runner
python/tune/           tuning application (its own pq/, node, dispatch, manifest, handlers)
```
