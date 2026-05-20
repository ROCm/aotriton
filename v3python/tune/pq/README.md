# pq -- PostgreSQL Task Queue

PostgreSQL-based distributed task queue for AOTriton tuning, replacing
Celery + RabbitMQ.  All modules accept a psycopg connection object; the
caller owns the connection lifetime.


## Modules

### queue.py -- TaskQueue

Core queue operations.  Accepts a psycopg connection on construction and
reuses it for all operations.

    import psycopg
    from v3python.tune.pq.queue import TaskQueue

    conn = psycopg.connect(host=..., port=..., user=..., password=...,
                           autocommit=True)
    q = TaskQueue(conn)

Methods:

  `fetch_tasks(arch, batch_size=10, tuning_mode='kernel')`
    Atomically claim batch_size pending tasks for arch using SELECT FOR
    UPDATE SKIP LOCKED.  tuning_mode='op' fetches operator tasks (module
    LIKE '%_op'); default fetches kernel tasks.  Returns list[Task].

  `mark_completed(task_id, arch)`
    Transition task to 'completed'.

  `mark_failed(task_id, *, arch=None, error_message)`
    Transition task to 'failed' with error text.  arch is keyword-only
    and optional; omit when arch is unknown (updates parent table).

  `mark_pending(task_id, arch)`
    Reset a running task to 'pending' (used during graceful shutdown).

  `retry_task(task_id, arch, max_retries=3)`
    Reset to 'pending' and increment retry_count.  Returns False if
    max_retries is exceeded.

  `get_queue_stats(arch=None)`
    Returns dict with pending/running/completed/failed counts.
    arch=None queries the parent table across all partitions.

  `detect_stale_tasks(timeout_seconds=7200)`
    Return tasks in 'running' state longer than timeout_seconds.

  `reset_stale_tasks(timeout_seconds=7200)`
    Reset stale running tasks to 'pending'; returns count reset.

Task dataclass:

    @dataclass
    class Task:
        id:             int
        arch:           str
        module:         str
        task_config:    dict
        status:         str
        priority:       int            = 5
        worker_id:      str | None     = None
        node_hostname:  str | None     = None
        created_at:     datetime | None = None
        started_at:     datetime | None = None
        completed_at:   datetime | None = None
        error:          str | None     = None
        retry_count:    int            = 0


### dispatcher.py -- TaskDispatcher

Bulk task insertion.  Creates its own short-lived connections.

    from v3python.tune.pq.dispatcher import TaskDispatcher

    dispatcher = TaskDispatcher({'host': ..., 'port': ...,
                                 'user': ..., 'password': ...})
    tasks = [
        {'arch': 'gfx942', 'module': 'attn_fwd',
         'task_config': {'BATCH': 4, 'H': 32, 'N_CTX': 1024,
                         'D_HEAD': 64},
         'priority': 5},
    ]
    count = dispatcher.dispatch_bulk(tasks)

Methods:

  `dispatch_bulk(tasks, batch_size=1000)`
    Batch INSERT an iterable of task dicts.  Returns total count.

  `dispatch_single(arch, module, task_config, priority=5)`
    Insert one task.

  `ensure_partition(arch)`
    Create the task_queue_<arch> partition if it does not exist.


### results.py -- result storage functions

Free functions; each accepts a psycopg connection as last argument.

```python
    from v3python.tune.pq.results import (
        save_tuning_result, save_optune_result, get_task_results)
```

  `save_tuning_result(task_id, report, conn)`
    Insert one hsaco benchmark result into tuning_results.
    report keys: kernel_name, hsaco_index, result,
                 result_data (JSONB, optional), error (JSONB, optional),
                 complete_on_gpu.

  `save_optune_result(task_id, report, conn)`
    Insert one operator benchmark result into optune_results.
    report keys: op_name, backend_index, result,
                 result_data (JSONB, optional), error (JSONB, optional),
                 complete_on_gpu.

  `get_task_results(task_id, conn)`
    Return all tuning_results rows for a task as a list of dicts.


### heartbeat.py -- HeartbeatManager

Worker liveness tracking.  Creates its own short-lived connections.

```python
    from v3python.tune.pq.heartbeat import HeartbeatManager

    hb = HeartbeatManager({'host': ..., ...}, arch='gfx942')
    hb.update('active')       # call periodically from worker loop
    hb.increment_completed()
    hb.mark_dead()            # call on graceful shutdown
```

Methods:

  `update(status)`
    Upsert heartbeat row; status is 'active' or 'idle'.

  `increment_completed()`
    Increment tasks_completed counter.

  `increment_failed()`
    Increment tasks_failed counter.

  `mark_dead()`
    Set worker status to 'dead'.

  `cleanup_dead_workers(threshold_seconds=300)`
    Mark workers with stale heartbeats as 'dead'.


### admin.py -- QueueAdmin

Schema initialization and maintenance.  Creates its own connections.

  `init_schema()`
    Create all tables, indexes, and views from schema.sql.

  `create_partition(arch)`
    Create `task_queue_<arch>` partition.

  `create_partitions(arch_list)`
    Create multiple partitions at once.

  `reset_stale_tasks(timeout_seconds=7200)`
    Reset long-running tasks to 'pending'.

  `cleanup_dead_workers(threshold_seconds=300)`
    Mark stale workers as 'dead'.

  `purge_completed(older_than_hours=24)`
    Delete old 'completed' rows.

  `get_statistics()`
    Return overall queue and worker statistics dict.


## Database Schema

### task_queue (partitioned by arch)

```
  id              BIGSERIAL    Task ID
  arch            TEXT         GPU architecture (partition key)
  module          TEXT         Tuning module name
  task_config     JSONB        Task configuration
  status          TEXT         pending/running/completed/failed/cancelled
  priority        INT          Higher value = higher priority
  worker_id       TEXT         Worker that claimed this task
  node_hostname   TEXT         Node hostname
  created_at      TIMESTAMP    Task creation time
  started_at      TIMESTAMP    Task start time
  completed_at    TIMESTAMP    Task completion/failure time
  error           TEXT         Error message (failed) or cancellation
                               reason (cancelled)
  retry_count     INT          Number of retries
```

Status values:

  pending    Waiting to be claimed.
  running    Being processed by a worker.
  completed  Finished successfully.
  failed     Terminated with error; eligible for retry.
  cancelled  Deliberately skipped; never retried or reset by scripts.

### worker_heartbeat

```
  worker_id        TEXT         Unique worker ID (hostname-pid)
  node_hostname    TEXT         Node hostname
  arch             TEXT         GPU architecture
  last_heartbeat   TIMESTAMP    Last heartbeat time
  status           TEXT         active/idle/dead
  tasks_completed  INT          Total completed tasks
  tasks_failed     INT          Total failed tasks
```

### SQL Views (schema.sql)

```
  queue_progress    Task counts by architecture
  worker_health     Worker status and activity
  task_timing_stats Duration statistics
  stale_tasks       Tasks running longer than threshold
  completion_eta    Estimated completion time
```


## Connection Pattern

TaskQueue and result functions take a persistent connection; create one
per worker process and reuse it:

```python
    import psycopg
    from v3python.tune.pq.queue import TaskQueue
    from v3python.tune.pq.results import save_tuning_result

    conn_params = {'host': ..., 'port': ..., 'user': ..., 'password': ...}
    conn = psycopg.connect(**conn_params, autocommit=True)

    q = TaskQueue(conn)
    while running:
        tasks = q.fetch_tasks('gfx942', batch_size=10)
        for task in tasks:
            report = execute(task)
            save_tuning_result(task.id, report, conn)
            q.mark_completed(task.id, arch='gfx942')

    conn.close()
```

TaskDispatcher, HeartbeatManager, and QueueAdmin manage their own
short-lived connections and accept conn_params dicts instead.

Note: never include 'dbname' in connection parameters. Let psycopg use
the default database (same as the username).  Workdir isolation is
achieved at the container level -- each workdir runs its own PostgreSQL
container backed by its own Docker volume, so different workdirs never
share a server instance and database-name separation is unnecessary.
