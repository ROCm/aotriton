# aotriton.perfmon.pq -- perfmon's PostgreSQL layer

The engine (`aotriton.pgdef.pq`, schema `pgdef`) owns the task queue, the raw
results and the workers: see `python/pgdef/pq` and
`docs/Distributed Execution Framework.md`. This package is the perfmon
application's own layer, in PostgreSQL schema `perfmon`. All SQL for perfmon
lives here. Engine tables are written only through `aotriton.pgdef.pq`; this
package may read them directly.

Every function takes a psycopg connection (`conn`); the caller owns it. Never
put a database name in the connection parameters: the server default is used.
Tools that are handed `--workdir` load config.rc with
`aotriton.pgdef.workdir.load_config` and connect with
`aotriton.pgdef.pq.conn.params_from_env()`.


## Vocabulary

  arch        The GPU architecture. It is the task's `pool`.
  variant     A group of machines of one arch expected to perform the same.
              It is the task's `subpool`; a task without one runs on any
              machine of the arch, and its results are variant `any`.
  subject     One build under test, `aotriton-<tag>+rocm<version>`.
  iface       The operator measured (`op_attn_fwd`, `op_attn_bwd`).
  backend     A backend forced by name, or NULL for the unforced operator.
  payload     A perfmon task's input, `{"module": "flash", "subject": ...,
              "entry": {...}}`, plus `"ifaces": [...]` for an LLM case that is
              measured forward only. The arch is a column, not a payload field.

perfmon has no second category level: `cat_level1` is NULL for every task.


## Schema (schema.sql, applied by `.tune/bin/initdb` after pgdef's)

```
  perfmon.measurement         view over pgdef.report JOIN pgdef.task WHERE cat_level0='perfmon':
                              id, task_id, arch, variant, subject, module, iface, backend,
                              entry (jsonb, sizes included), status (OK/NotOK/crash/excluded),
                              median_us, mean_us, stddev_us, min_us, p05_us, p95_us,
                              timing_method, iterations, l2_flush, platform, thermal,
                              harness, error, worker, attempt, created_at
  perfmon.latest_measurement  the newest row per (arch, variant, module, subject, entry,
                              iface, backend); older rows are never averaged in
```

No perfmon table exists. A measurement is a `pgdef.report` row with
`rkey = {"iface", "backend"}`; its payload holds `stats` (µs), `timing_method`,
`n`, `l2_flush`, `seed`, `harness`, `backend_forced`, `platform`, `thermal`,
`subject`, `module`, `entry`, `worker` and `gpu`. The schema adds one partial
expression index on perfmon's rows.


## Functions (measurements.py)

  `save_measurement(conn, (task_id, arch), report, *, attempt)`
    One `pgdef.report` row per (iface, backend) of a task, through
    `pgdef.pq.reports.insert(..., replace=True)`, so a retry leaves no
    duplicate.

  `iter_latest(conn, *, arch, variant=None, subjects=None, module=None)`
    Rows of `perfmon.latest_measurement`, for the export.

  `count_superseded(conn, *, arch, variant=None, subjects=None, module=None)`
    How many older rows the latest ones supersede.

  `list_selections(conn, *, module=None, arch=None, variant=None, subjects=None)`
    The (module, arch, variant, subject) combinations with measurements,
    with counts per status and the newest date: `export -l`, and the webui.

  `subject_progress(conn, *, arch=None)`
    Task counts per (arch, variant, subject) and status, for the webui.

The result handler records finished (iface, backend) pairs in the task's
appcookie (`{"done": [[iface, backend], ...]}`) through
`pgdef.pq.tasks.append_cookie`; after a reset with `keep_appcookie` the probe
skips them.
