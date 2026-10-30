# Distributed Execution Framework (pgdef)

pgdef runs many independent GPU tasks across a fleet of machines. It uses
PostgreSQL as the task queue and a small per-node runtime to execute each task.
It knows nothing about tuning or performance measurement; those are
**applications** built on top of it. pgdef is a library: applications call it,
and it never loads them.

| Application | Package | What one task does |
|---|---|---|
| tune | `python/tune` | Benchmark every kernel/op implementation for one entry, and record the timings |

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

`id` identifies a task. Raw results go to `pgdef.report`, one row per item a
task produces (for tune, per tuned impl), and each row points back to its task. Both tables are
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
  + Completing, failing or releasing a task names the lease owner (the
    message's `lease_owner`), so a graph whose lease expired and was taken
    over cannot settle the new claim. `owner=None` explicitly skips the check.
* **Retry is opt-in.** By default a task gets one attempt (`max_attempts` = 1),
  so any failure or expired lease ends in `failed`, and redoing it is an
  operator reset. pgdef retries automatically only if a task is dispatched with
  a larger budget (`--max_attempts N`, or the application's default).
* **Reset.** An operator reset clears `appcookie` by default, so the task is
  redone from scratch. Reset with `keep_appcookie` lets the application skip
  work it already finished.

## On a node

Each node runs **one application**. It has one broker and three types of
worker, all connected through a Unix socket:

```
                  ┌────────────┐
                  │   broker   │   queues: gpu, cpu
                  └────────────┘
                        ↕  Unix socket
    ├── PG readers   (claim a task each; get its ack from the broker)
    ├── GPU workers  (one per GPU; run the subprocess runners, exaid)
    └── CPU workers  (write results to PostgreSQL)
```

The broker is the hub, and every connection is two-way. A handler on any
worker can send a message to any queue, and the broker tells the reader when
its task is finished.

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

### Fan-in: semaphore messages

A step that waits for several others (a join) is not a waiting handler. It
is a **semaphore message**: an ordinary message that carries a `depends` key,
listing the message classes it waits for. Tune's probe, for example, emits N
`tune/impl` messages plus one `tune/postprocess` that depends on
`tune/impl_result`.

1. A handler returns the semaphore message like any other message.
2. The broker's `forward()` sees `depends`. Instead of queueing the message,
   it parks it under each class it depends on.
3. Every later message of one of those classes is offered to the parked
   message: the broker calls `resolve_dependency(blocked, incoming)` on the
   **parked message's handler class**. This runs in the broker. The incoming
   message is still delivered to its own queue as usual.
4. When `resolve_dependency` returns true, the broker removes `depends` and
   forwards the semaphore message to its queue. Its handler then runs like any
   other.

`resolve_dependency` is the extension point: a handler can override it to
release on any condition. The default is the shared helper
`aotriton.pgdef.localq.semaphore`, which covers the usual case, "wait until a
signal has arrived for every key":

```python
# in the fanning-out handler: wait(depends, keys, result_msg)
join = semaphore.wait('tune/impl_result', [('attn_fwd', 0), ('attn_fwd', 1)], postprocess_msg)
semaphore.signal(signal_msg, key=('attn_fwd', 0), value=report)   # in each result
reports = semaphore.get_aggregate_results(msg)    # in the semaphore message's handler
```

**Getting the results.** A fan-in usually needs the parallel steps' results,
not just their completion. The semaphore message collects them:

* Each parallel step returns a message marked with
  `semaphore.signal(msg, key, value)`, where `value` is that step's result.
* While the semaphore message is held in the broker, `resolve()` (the default
  `resolve_dependency`) accumulates every matching message into it. A message
  matches if its class is in `depends`, it belongs to the same task, and it
  carries a key. Its value is stored in the semaphore message's own state:
  `msg['semaphore']['received'][key] = value`, next to the expected keys in
  `msg['semaphore']['expected']`.
* Once released, the semaphore message brings everything it collected to its
  handler. The handler reads it from its own input message:
  `semaphore.get_aggregate_results(msg)` returns `{key: value}`.

Only the value travels, not the rest of the signalling message, so put
everything the join needs into it. Values must be JSON values, and they are
held in the broker until release, so keep them small.

If the broker shuts down while a semaphore message is still parked, it calls
the handler's `teardown_with_unmet_dependency()`. By default that gives the task
back to the queue.

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

(`.def/lib/manifest.sh` wraps this as `app_manifest` and `require_app`, the one
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

`.def/bin/initdb`, which is also behind the webui's Init DB button, creates
every schema for every application. Each `pq` ships an idempotent
`schema.sql`, and pgdef's runs first. `--recreate` drops and recreates them all.

Partitions need no setup. Dispatch creates a pool's partitions the first time
it inserts into that pool, by calling the engine's SQL function
`pgdef.ensure_pool(pool)` with the pool name as a query argument.

## Common operations

```bash
# Dispatch: each application has its own dispatch command
python -m aotriton.tune.dispatch --category tune,kernel --pool gfx950 --workdir <workdir> --module flash <tune options>

# Start a node's processes (normally done by .tune/remote/worker_service.sh --category tune,kernel)
python -m aotriton.tune.node reader --pool gfx950 --subpool MI355 --category tune,kernel
```

Progress and worker health come from the `queue_progress` and `worker_health`
views, which the webui shows.

## Code layout

```
python/pgdef/pq/       engine schema and SQL (tasks, logistics, reports, workers, dispatch driver)
python/pgdef/localq/   broker, workers, handler base classes, semaphore messages, run_node_role
python/pgdef/exaid.py  supervised subprocess runner
python/tune/           tuning application (its own pq/, node, dispatch, manifest, handlers)
```
