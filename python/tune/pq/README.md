# aotriton.tune.pq -- tune's PostgreSQL layer

The engine (`aotriton.pgdef.pq`, schema `pgdef`) owns the task queue, the raw
results and the workers: see `python/pgdef/pq` and
`docs/Distributed Execution Framework.md`. This package is the tune
application's own layer, in PostgreSQL schema `tune`. All SQL for tune's tables
lives here. Engine tables are written only through `aotriton.pgdef.pq`; this
package may read them directly.

Every function takes a psycopg connection (`conn`); the caller owns it. Never
put a database name in the connection parameters: the server default is used.
Tools that are handed `--workdir` get parameters from
`aotriton.pgdef.workdir.get_db_connection_params`, which sources config.rc.


## Vocabulary

  arch        The GPU architecture. It is the task's `pool` in `pgdef.task`.
  tuning_level
              `kernel` or `op`. tune stores it as the `cat_level1` column of
              `pgdef.task` and of tune's own tables (and as `cat_level1` in
              the report's rkey); `--category tune,kernel` on the CLI.
              `iface_name` collides across the two levels, so every query over
              tune's tables filters on the column.
  payload     A tune task's input, `{"module": "flash", "entry": {...},
              "max_hsaco": {...}?}`. The arch and the level are columns, not
              payload fields.


## Schema (schema.sql, applied by `.tune/bin/initdb` after pgdef's)

```
  tune.tuning_results                view over pgdef.report WHERE cat_level0='tune':
                                     id, task_id, arch, cat_level1, iface_name,
                                     impl_index, result (OK/NotOK/crash/ERROR),
                                     result_data, error, worker ('gpu-<n>'), attempt
  tune.best_tuning_results           fastest accurate impl per (task_id, iface_name);
                                     rebuilt wholesale by compute_best_results
  tune.most_accurate_tuning_results  per (task, iface, test case, tensor) accuracy
                                     targets; report_id anchors each row to one of the
                                     task's reports (ON DELETE CASCADE), so a generic
                                     reset clears it with no tune code
  tune.extra_uts                     extra unit tests per task, from re-queued broken
                                     entries; (task_id, pool) -> pgdef.task, cascade
```

The raw per-impl results are rows of `pgdef.report` with
`rkey = {"cat_level1", "iface", "impl"}`. The schema adds partial expression
indexes on them for tune's rows.


## Modules

### tasks.py -- finding, resetting and counting tune tasks

  `category(level)`, `level_of(text_or_category)`, `check_level(level)`
    `('tune', 'kernel')`; and the inverse for a `--category` value.

  `entry_selector(entry, *, arch=None, tuning_level=None, module=None, **kw)`
    A `pgdef.pq.tasks.TaskSelector` for one tuning entry. Scalar fields match
    by jsonb containment; tuple/list fields by exact ordered path equality.

  `find_by_entry(conn, entry, *, arch, tuning_level, module, limit)`,
  `find_by_ids(conn, ids)`
    Rows `{id, arch, module, tuning_level, status}`.

  `find_resettable(conn, 'stale'|'failed', timeout_seconds=None)`,
  `find_resettable_by_ids(conn, ids)`
    Rows with `id, arch, tuning_level, status, lease_owner, age_s`.
    Readers reap expired leases themselves; 'stale' finds claims that are
    still leased but older than the timeout (a wedged reader).

  `reset(conn, ids, tuning_level, *, delete_results)`
    Reset to pending. `delete_results` is required: with it the tasks' reports
    are purged and `most_accurate_tuning_results` cascades.

  `get_progress(conn, tuning_level)`, `completed_payloads(conn, module, tuning_level)`
    Per-arch progress from the engine views; done payloads for `--skip_completed`.

### results.py

  `save_tuning_result(conn, (task_id, arch), report, *, attempt)`
    Insert one `pgdef.report` row for a tuned impl.
  `get_task_results(conn, task_id, tuning_level=None)`,
  `get_task_debug_snapshot(conn, task_id)`
    Reads, through `tune.tuning_results` and the derived tables.

### extra_uts.py

  `insert_extra_uts(conn, (task_id, arch), im_texts)`, `get_extra_uts`,
  `delete_extra_uts`. Rows accumulate across passes; a reset never deletes them.

### Tools (run as `python -m aotriton.tune.pq.<tool>`, wrapped by `.tune/bin`)

  compute_best_results       reports -> tune.best_tuning_results
  populate_most_accurate     reports -> tune.most_accurate_tuning_results
  export_best_results        best results -> SQLite LUTs
  manual_pick_missing_entry  hand-pick an impl for an entry
  visperf, export_visperf    data and standalone export for the webui perf page

All take `--category tune,kernel|tune,op` (default `tune,kernel`).


## Dispatching

`python/tune/dispatch_tasks.py` builds `TaskSpec(payload, category=('tune', level))`
for each entry and arch and inserts them with `pgdef.pq.tasks.insert`, after
`ensure_partition(conn, arch)` creates the pool's partitions on demand.
`--skip_completed` matches done payloads from `completed_payloads`.
