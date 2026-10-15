# pgdef (DEF): agent reference

Read this before touching `python/pgdef/`, `python/tune/{pq,localq,exaid}`,
or before proposing any change to the distributed execution
framework.

* Human-facing doc: `docs/Distributed Execution Framework.md`.
* Full plan with survey and phases: `gDEF-ver0.md` at the repo root. It is
  untracked and may have been moved or removed; if it is missing, ask the user.

Status as of 2026-10-12: **design approved; implementation in progress**,
following `gDEF-ver0.md` rev 5. Check `git log` to see
which phases have landed. Landed so far:

* P0: standalone fixes (webui `action_tracker`, `initdb` `ON_ERROR_STOP`, writable HOME).
* P1: `python/pgdef/{utils,exaid}.py`, the layering test.
* P2: the engine database layer (`python/pgdef/pq/`), and tune ported onto it
  (`python/tune/pq/`, schema `tune`, `cat_level1` instead of `tuning_level`).
* P3: the readable local runtime (`python/pgdef/localq/`: handlers with
  `CLASS`/`QUEUE`/`EMITS`, `semaphore`, `graph`, `node.run_node_role`, the single
  failure path, leases in the reader) and `pgdef.pq.dispatch.drive`; tune's
  `node`, `dispatch` and `handlers`.

* P4: `MANIFEST` per application (`python/tune/manifest.py`; shared lookup in
  `pgdef/manifest.py`, `.tune/lib/manifest.sh`); `worker_service.sh`, `wkctl`,
  `deploy`, `sync_workdir`, `fetchbuild` and `start_worker` read it; `workers.db` has a `subpool` column. NOT done: unifying
  tuning onto the perfmon image model and non-root container (the plan keeps tuning
  root, installing aotriton at container launch, in this refactor).
* P5 (perfmon on pgdef) is **not in this PR**. This PR has one application, tune.
  Perfmon is the planned second application and lands in a later PR. See
  "Perfmon: what comes next" below.

## Vocabulary

* **DEF / pgdef.** The Distributed Execution Framework on PostgreSQL. The
  package is `python/pgdef/`, imported as `aotriton.pgdef`, because `python/`
  maps to the `aotriton` package in `setup.py`.
* **Application.** A layer on top of pgdef. Today there is one, `tune`
  (`python/tune`); `perfmon` is planned as a second one. They are peers; perfmon
  is not a sub-feature of tune. A task's `cat_level0` column holds the application name.
* **Protocol: the application name IS `cat_level0`.** An application named
  `<app>` (package `aotriton.<app>`, manifest `python/<app>/manifest.py`)
  owns every category whose first level is `<app>`. So the category alone
  identifies the application:
  * A tool that takes `--category <app>[,<level1>]` must **not** also take
    `--apps`. It derives the application as the category's first level (in
    shell, `${CATEGORY%%,*}`).
  * No shared tool takes `--apps`. The image tools (`prepwkdir`, `imgbld`,
    `build_image.sh`) are tune's own, unchanged from 0.14, and take no
    application at all.
  * **Category helpers** in `.tune/lib/category.sh`, mechanism separate from
    policy:
    * `parse_category <category>` is pure syntax. It sets `CAT_LEVEL0` and
      `CAT_LEVEL1` (one field each, empty if absent), with no manifest lookups
      and no errors.
    * `validate_category <category>` is the policy, from the manifest. It
      rejects a missing category, an unregistered application, a level not in
      the manifest's list, a missing level when the manifest lists some (`tune`
      alone), a level when it lists none (`<app>,x`), and more than two levels.
  * **End-to-end forwarding.** Scripts forward the category string as it is
    (`"$CATEGORY"`), never rebuilt from parsed levels, so a future
    `cat_level2` passes through every hop untouched. Each hop parses only the
    levels it uses itself.
    * **Endpoints validate**, where a level gets its meaning:
      `worker_service.sh`, `sync_workdir.sh` (install tree), `fetchbuild`.
    * **User-facing entry scripts validate early** (`deploy`, `wkctl`), so a
      typo fails before any ssh. That check is for speed only.
    * **Pure forwarders don't validate**: `start_worker.sh`, `restart_worker.sh`,
      `run_worker.sh`.

    There is no default category and no default application. `fetchbuild`,
    `worker_service.sh`, `run_worker.sh`, `start_worker.sh`, `deploy`, `wkctl`
    and `sync_workdir.sh` take `--category` only.
  * **Image building is out of the gDEF PR.** `prepwkdir`, `imgbld` and
    `build_image.sh` are as in 0.14: tune only, no `--apps`, no manifest. The
    manifest has no `image` or `container` section: no script reads them (the
    worker image is `$CELERY_WORKER_IMAGE`, as in 0.14). When the perfmon PR makes these tools
    application-aware, they take `--category <app>` and use level 0. If hosts
    are ever to be picked by the application's `deploy.role`, the **caller** of
    `imgbld` does that and passes `--host`; `imgbld` stays a mechanism.
* **pgdef is a library, not a framework.** Applications call it; it never
  discovers, imports or launches them.
  * Each application owns three entry points: `aotriton.<app>.node` (calls
    `pgdef.localq.node.run_node_role(app=, handlers=, start=, runners=,
    enrich=, argv=)`), `aotriton.<app>.dispatch` (calls
    `pgdef.pq.dispatch.drive(source, argv)`) and `aotriton.<app>.manifest`.
  * `worker_service.sh --category <app>[,<level1>]` runs `python -m
    aotriton.<app>.node <role>`. One application per node.
  * **`.tune/remote/worker_service.sh` is the generic DEF node starter, not a
    tuning script**, despite living under `.tune/`.
    * It takes only engine vocabulary: `--category <app>[,<level1>]`
      (required for start and restart; its first level is the application)
      and `--subpool`.
    * It must never grow application-specific flags such as tune's
      `--tuning_mode`. Those belong to the application's own tools.
    * It exports the neutral `AOTRITON_BROKER_SOCKET`.
    * The SLURM wrapper `.tune/remote/run_worker.sh` forwards `--category`
      and `--subpool` to it.
  * The shared process layout is a known limitation; the ideal fix is a
    service script per application. Until then, zero GPU or zero CPU workers
    must be a valid configuration, and a message for a queue with no workers
    fails the task instead of hanging.
* **One `pq` per layer, one PostgreSQL schema per layer.**
  `aotriton.pgdef.pq` owns `pgdef.*`, and `aotriton.<app>.pq` owns `<app>.*`
  (for example `tune.best_tuning_results`). Nothing goes in `public`.
  * An application's `pq` may read engine tables directly. It writes them only
    through `pgdef.pq`.
  * `pgdef.pq` never mentions an application schema.
  * Raw per-item results stay in the generic `pgdef.report`. Applications add
    typed views and derived tables in their own schema. Tables derived from
    reports use `ON DELETE CASCADE`, so a generic reset needs no application
    code.
* **initdb** (`.tune/bin/initdb`, which is also behind the webui buttons) applies
  every layer's idempotent `schema.sql` with `psql`. pgdef's runs first, then a
  glob of `python/*/pq/schema.sql`. It creates all applications' schemas,
  imports no Python, and `--recreate` drops those schemas with `CASCADE`.
* **The pairs:**
  * `(pool, subpool)`: pool = arch (gfx950), subpool = machine group (MI355).
  * The **category** `(cat_level0, cat_level1)`: cat_level0 = application (`tune`,
    `perfmon`); cat_level1 = tune's `kernel` / `op`, NULL for an application without levels (perfmon, planned). It replaces
    today's `--workload`/`tuning_level`.
    * PostgreSQL: one column per level, `cat_level0 TEXT NOT NULL`, `cat_level1 TEXT`.
    * Python: `category: tuple[str, str | None]`, converted only inside `pq`.
    * Shell: `--category tune,kernel`, with missing levels NULL.
    * C++ (if ever needed): a struct.
    * N-D: add `cat_level2`, and so on. Only trailing levels may be NULL (enforce
      with a CHECK). `pq` generates one `(cat_levelN IS NULL OR cat_levelN = %s)` claim
      term per level.
  * **The rule:** a NULL second field on a task means generic. A reader claims
    tasks whose second field is NULL or equals its own.
* **Task columns:**
  * `payload` (jsonb): application input, immutable.
  * `appcookie` (jsonb, nullable): application-owned progress state, opaque
    to the engine like an HTTP cookie. Kept on retry, cleared on reset unless
    `keep_appcookie`. Formerly `exstatus`, rejected as ambiguous and too
    close to `status`.
    * Writes: `pq.tasks.append_cookie(conn, task_ref, key, values)` appends
      atomically in SQL, so concurrent CPU handlers can't lose entries.
      `pq.tasks.replace_cookie(conn, task_ref, cookie)` compacts or
      overwrites.
    * There is no merge (`||`): it replaces existing keys, which forces
      re-sending whole lists and races with other writers.
  * `logistics` (jsonb): engine-owned attempts, lease, retry policy and error
    history.
    * **Automatic retry is implemented but unused.** `DEFAULT_MAX_ATTEMPTS = 1`
      (`pgdef/pq/logistics.py`), and tune does not set
      `default_max_attempts` or passes `--max_attempts`. So every failure or
      expired lease ends in `failed`, and redo is an operator reset. Keep the
      mechanism and its tests, but do not turn it on by default.
    * Only `pq/logistics.py` writes `logistics`.
* **Naming rule: category names for storage, application names for code.**
  * Anything that is a **database column, or a key stored in database
    jsonb**, uses the category names: `cat_level0`, `cat_level1`. That applies
    in pgdef's tables (`pgdef.task`, the `rkey` of `pgdef.report`) and in an
    application's own tables alike (e.g.
    `tune.most_accurate_tuning_results.cat_level1`).
  * **Application code and CLIs use the application's own concept**, because
    concrete terms read better than abstract levels. This covers identifiers,
    parameters, comments, log text and flags in `python/tune`,
    `modules/*`, the tune-only `.tune` tools, and the webui's
    calls into them. For tune that is `tuning_level` ('kernel'|'op') and the
    flag `--tuning_mode`.
  * **pgdef and code shared by applications** (`worker_service.sh` and the
    other worker, deploy and sync scripts, plus the webui's generic category
    setting) use `category` and `--category tune,kernel`.
  * **The translation lives in the application's `pq`.** Its storage mapping is
    documented there from the application's side: "tune stores its tuning
    level as `cat_level1`". SQL keeps the column name, while placeholders and
    parameters use the application term, e.g.
    `WHERE t.cat_level1 = %(tuning_level)s`.

## Decided: do not re-propose these

The user rejected each of the following. Reasons are given so you can
recognise variants of the same idea.

| Rejected | Reason |
|---|---|
| Dedicated columns for lease, attempt, max_attempts, timestamps | Every new maintenance capability would need DDL on a partitioned table. Use the `logistics` jsonb |
| `UNIQUE(pool, cat_level0, payload)`, a `task_key` hash, or any dedup constraint | Identical tasks are legal. Redo = reset status. Hashing adds confusion. Dedup is only the opt-in dispatch flag `--skip_completed` |
| A `dag` column, or several DAGs per application | Each application has exactly one handler graph |
| Declarative DAGs (Dag/Node/FanOut/Join), or any DSL in Python or in PostgreSQL | Overkill. Keep the implicit handler graph and make it readable: `CLASS`/`QUEUE`/`EMITS` on each handler, runtime check of `EMITS`, `localq/graph.py` printer |
| Task-to-task dependencies | All dependencies live inside one task on one node (localq). Neither tune nor the planned perfmon needs nodes to collaborate. This property is why we built our own DEF |
| pgdef discovering applications (`PGDEF_APPS`, entry points, a module-attribute convention); generic `pgdef` dispatch or manifest CLIs | pgdef is a library and applications launch it. The multi-application generality was never used: tune and perfmon need different images |
| An `on_reset` application hook | Replaced by `ON DELETE CASCADE` from application tables to `pgdef.report`; best results are recomputed anyway |
| A two-phase `--` dispatch CLI (engine options, then application options after `--`) | Each application owns its dispatch CLI, and the engine adds its common options to the same parser |
| A single `pq` for all layers, or application SQL in `pgdef/pq` | It would make every application schema change an engine edit, including the `PerfDescription` redesign |
| Raw results in per-application tables | A generic reset could not purge them; each application would need its own reset tool |
| A `merge_appcookie` (jsonb `||`) writer | Use `append_cookie` + `replace_cookie`; merge forces read-modify-write of whole lists |
| Engine-level resume or skip of partially done tasks | Skipping is application-specific (tune must re-run `prepare_data` anyway). Applications implement it with `appcookie` |
| `requires`/caps jsonb claim predicate; affinity column; RunnerPool | Replaced by the pairs. Perfmon keeps dispatching grouped by preset with `id`-ordered claims |
| A separate `aotriton-pgdef` distribution, or a `python/pgdef/pgdef/` directory | Small gain, and the directory stutter is disliked. Layering is enforced by a test instead |
| Generating a Dockerfile on the server at build time (perfmon's `generate_on_build` model), to avoid stale images | A second image model, remote-uid lookups and file shipping, to defend against one silent error. Make staleness visible instead: the ROCm version goes into the artifact (image tag) |
| Data migrations between schema versions | Deployments last under a month. Fresh DB per release; `initdb --recreate` is the upgrade path |
| A specific-before-generic claim order | Order is `priority DESC, id`. Applications use `priority` |
| PK/FK carrying category columns (e.g. `(id, pool, cat_level0)`) | `id` is the single identity of tasks and reports; extra key columns only complicate the mental model |
| Partitioning on `cat_level0`/`subpool` (a second level) | Not needed now. Revisit only if claim performance becomes a problem |
| A shared entry-description base for tune and perfmon | The sharing only existed because perfmon borrowed the tuner's DAG. Perfmon works at op granularity (entry × iface × backend); `TuningDescription` is kernel/hsaco-centric |
| Renaming `pq` to `pg` | Both are legitimate names (libpq vs postgres); keep `pq` |
| `cat_level1` (or `--category tune,…`) as a variable, parameter or flag name in application code, or in tune-only tools | Abstract names hide concrete concepts and generalise nothing there. Use the application's term (`tuning_level`, `--tuning_mode`); keep `cat_levelN` for database columns |
| Naming the pair `(workload, flavor)`, `(category, subcategory)`, `subworkload`, `variant`, `mode`; columns `cat_0`/`cat_1` or `pool_0`/`pool_1` | Superseded by the 2-D category `cat_level0, cat_level1`. Numbered bare names read as "two categories/pools". `pool`/`subpool` keep their names. `level` is fine: `tuning_level` gets revised when tune is ported, so it is no conflict |
| Category as a PostgreSQL composite type or `text[]` | Composite: psycopg needs per-connection registration to load it, and NULL semantics are surprising. `text[]`: unnamed positional levels. Plain columns `cat_levelN` instead |

**Guiding principle: KISS.** A deployment lives for one campaign of under a
month, and the design is revised almost every release. Prefer a small, agile
mechanism that can break over a polished one. When a design problem appears,
first try "replace the single field with a pair".

**Out of scope for the pgdef refactor:** moving the tuning container to
non-root. That is a separate PR; tuning stays root.

## Rules that still apply (from CLAUDE.md)

* `pq` functions take `conn`, never connection parameters.
* No `dbname` in connection parameters.
* **Main goal: no raw SQL outside a `pq` package.** There is one `pq` per
  application or library (engine, tune; perfmon later).
* `config.rc` holds tuning-infrastructure settings only. A machine's subpool
  goes into `workers.db` (a new column on the workers table), and per-application
  shell knowledge goes into the application's `MANIFEST`.
* `aotriton.pgdef` must not import `aotriton.tune`, `aotriton.perfmon`, torch or
  triton.

## Facts about the current code

Verify these before relying on them.

* **localq framing** is a 4-byte big-endian length prefix followed by JSON. It
  is not newline JSON.
* **exaid** is a line protocol: `OK <json>` means success, `OVERHEATING:` is a
  keep-alive, anything else is an error. The client is `pgdef.exaid.ExaidProxy`
  (neutral: a `spawn_argv` callable, relaunch after a crash); the server side
  skeleton is `pgdef.exaid.CommandProcessor`/`serve`, used by
  `python/tune/testrun.py`.
* **Nothing tuning-specific is in the engine.** The broker is built with the
  handler classes, priorities and queue names; the reader's category is a
  parameter; the engine ack class is `pgdef/dag_ack`.
* **Leases exist.** `tasks.claim` stamps `logistics.lease`; the reader renews
  every `lease_s/3` while its broker connection lives and runs a rate-limited
  `reap_expired` before claiming. Staleness by log mtime and `reset_task` for
  stranded rows are gone (the operator's `reset_task --select stale` remains
  as an override for a live-but-wedged reader).
* **Settling is lease-guarded.** `complete`/`fail`/`release` take a required
  keyword `owner` (`handlers.owner_of(msg)`); a mismatch changes nothing.
  `owner=None` is the explicit "don't care".
* **No leaked reader slot:** any handler exception becomes `task_failed`
  (`GenericWorker._report_failure`); a message for a queue with no workers fails
  the task (`LocalBroker._undeliverable`).
* **Verified only without PostgreSQL.** The SQL in `python/pgdef/pq/`,
  `python/tune/pq/` and the `.tune` scripts has NOT been executed: the
  development sandbox had no PostgreSQL server. The `pg`-marked tests
  (`python/test/pgdef/test_pq_pg.py`, `python/tune/tests/test_tune_pq_pg.py`)
  are written; run them against a disposable server first
  (`aotriton.pgdef.pq.testkit`).
* **Stale docs:** `.tune/DESIGN.md` still describes Ray, `worker_main.py` and
  `tuning_tasks`. The codegen agent's own paths (`v3python/codegen`, `rules`,
  `base`) have not been updated yet.
* **Packaging:** `python/tune` is its own distribution, `aotriton-tune`, with
  its own `pyproject.toml`/`setup.py`. It ships `aotriton.tune` into the PEP 420
  `aotriton` namespace via `package_dir={'aotriton.tune': '.'}`, and the root
  `setup.py` excludes `tune`. `aotriton.pgdef` (and later `aotriton.perfmon`)
  ship in the root `aotriton` distribution: `find_packages()` picks them up, and
  `package_data` lists their `pq/*.sql`.
* **Config.rc is infrastructure.** The engine reads `PG*` variables
  (`pgdef.pq.conn.params_from_env`); sourcing `config.rc` is done by shell
  (`worker_service.sh` exports them) or by `aotriton.pgdef.workdir` for CLIs
  that take `--workdir`.

## Perfmon: what comes next

Perfmon is not in this PR's tree. A first version exists outside it; the notes
below describe that version and the PR that brings it in.

* **Native harness, kept as is:** `perfmon/core`, the per-tag
  adapters in `modules/flash/perfmon/runner/`, `launch_runner.sh`,
  `build_subject.sh`, `scripts/build-*.sh`, and the sha256-verified
  release-asset cache.
* **Python, ported, to be redesigned later (on the branch).** `PerfDescription`
  was ported largely as is, with dead code dropped. Perfmon's SQL lives in
  `aotriton.perfmon.pq`, with no `LIKE` routing on preset names, and its
  handlers use the `CLASS`/`QUEUE`/`EMITS` style.
* **Next PR:** redesign `PerfDescription`, with no shared base with
  `TuningDescription`. It is neutral to pgdef: everything it decides crosses
  into the engine only as jsonb, or as perfmon code passed to `run_node_role`.
  If that redesign seems to need a pgdef change, treat it as a design smell.
* **Next PR (tentative): reorganise `.tune/` for all pgdef applications**, as
  part of materialising perfmon. The layout is **not decided**. Candidates:
  * a nested `.defs/` that becomes the shared directory of all pgdef
    applications;
  * a flat layout, `.def/` (the shared scripts) next to `.tune/` and
    `.perfmon/` (each application's own tools), which is easier to reach.

  Either way, several scripts there are already application-independent:
  `worker_service.sh`, `run_worker.sh`, `start_worker.sh`, `sync_workdir.sh`,
  `deploy`, `wkctl`, `fetchbuild`, initdb, `lib/manifest.sh` and
  `lib/category.sh`. Tune-only tools (the LUT pipeline, retry and
  broken-entry tools) need a decision on where they go. Do **not** start the
  move in the gDEF PR. `webui-to-root.md` (untracked, repo root) has the
  blast-radius survey for moving the webui out of `.tune/`, which should be
  planned together with this move.
* **Next PR: building perfmon's image and subjects.** This PR leaves out
  `.tune/lib/create_perfmon_dockerfile.sh`, `.tune/single/build_perfmon.sh`
  (which builds subjects on the build node, using `perfmon/build_subject.sh`),
  the webui's perfmon image button, its (arch, tag) subject matrix, and its
  fetch-subjects button. `build_image.sh`, `imgbld` and `prepwkdir` are back to
  0.14.
  * **Decided direction (KISS):** perfmon uses tune's image model. The
    Dockerfile is generated by `prepwkdir`, synced by `deploy`, and built on the
    runner.
  * **Put the ROCm version into the artifact** (e.g. the image tag,
    `-perfmon_{arch}_rocm{version}`). Then a worker that expects ROCm X finds no
    image for X, and a stale image is an error, never a silent one.
  * **Do not restore the branch's server-side Dockerfile generation**
    (`generate_on_build`, the remote-uid ssh, shipping `requirements/` with
    tar). Its only justification was "a stale image is built silently". The
    fix for that is visibility, not another build mechanism.
* **VRAM model divergence.** On the perfmon branch, `modules/flash/perfmon/resolve.py` has diverged
  from tune's memory model. Reconcile them in the next PR, together with the
  `PerfDescription` redesign. Perfmon still imports `aotriton.tune` for
  `FlashEntry`; drop that then too.
* **PON.** `python/utils/pon.py` never emits spaces inside tuples. The runner's
  old `rejoin()` workaround for space-split tokens was removed from
  `perfmon/core/main.cc` on the perfmon branch.
* **Design IDs in comments.** Comments in the perfmon branch's code may cite design IDs such
  as R01–R07 and D01–D12, or `perfmon-rev2.md`. That document no longer
  exists; this reference and `gDEF-ver0.md` supersede it.

## Implementation gotchas

* **Keys on partitioned tables.** PostgreSQL requires the partition key
  (`pool`) in every PK/UNIQUE constraint of a partitioned table, and an FK must
  reference such a key. That is the *only* reason the PK is `(id, pool)` and
  `pgdef.report` references `(task_id, pool)`. Conceptually `id` alone is the
  identity. PG 12 or newer.
* **Indexing logistics times.** Store them as epoch float8. `text::timestamptz`
  is not IMMUTABLE, so it cannot be used in an expression index.
* **Partitions.** One level, by `pool` (`pgdef.task_<pool>`,
  `pgdef.report_<pool>`), created on demand: `dispatch.drive` calls
  `tasks.ensure_pool(conn, pool)` before inserting, which runs the SQL function
  `pgdef.ensure_pool(%s)` with the pool as a bound argument (DDL takes no
  parameters, so the function builds the leaf name, validated and quoted with
  `%I`). An existing leaf is skipped via `to_regclass` before any DDL, so the
  parent is not locked; a concurrent creation is tolerated. Nothing reads
  workers.db for pools. There is no DEFAULT partition, so `tasks.insert`
  without `ensure_pool` fails loudly.
  All other SQL targets the parent table with `pool = %s` (partition pruning);
  never build leaf names with f-strings.
* **NULL comparisons.** A reader with a NULL `cat_level1` or subpool compares
  `x = NULL`, which is never true. It therefore claims only generic tasks, which
  is the intended behaviour.
* **Python 3.10 target.** There is no `tomllib`, so the application manifests are
  Python dicts.
