# Local Queue: Unix Socket Design (State of the Art)

Derived from design rev3 + feedback fb3, validated against the actual implementation
in `v3python/tune/localq/`.

## Overview

Replaces the legacy Ray/Celery framework with Unix domain sockets and a
message-passing architecture. The DAG (preprocess → probe → tune_hsaco × N →
postprocess) is formed implicitly by handlers forwarding result messages to
named queues in the broker.

## Process Topology

```
┌──────────────────────────────────────────────────────────────┐
│                        LocalBroker                           │
│  (epoll event loop, non-blocking sockets, BufferedSocket)    │
│  Queues: gpu_queue, cpu_queue, dispatcher_queue              │
│  Dependency tracker: blocked_messages dict                   │
│  Ack tracker: pending_acks dict                              │
└──────────────────────────────────────────────────────────────┘
    ↑  Unix socket (blocking, select+wakeup_fd)
    ├── PGReaderWorker × M   (pg_reader_worker.py)
    ├── GenericWorker/GPU × N (gpu_worker_socket.py, gpu_queue)
    └── GenericWorker/CPU × K (cpu_worker.py, cpu_queue)
```

**Broker** (`broker.py`): single-process epoll event loop. Non-blocking sockets
on the broker side, with `BufferedSocket` (`buffered_socket.py`) managing
length-prefix framing and send buffering for EPOLLOUT.

**Workers** (`generic_worker.py`): blocking sockets with `select` + a wakeup
pipe for signal-safe shutdown. Workers use `send_message`/`recv_message` from
`protocol.py` (length-prefix framing, 4-byte big-endian + JSON payload).

## Wire Protocol

**Framing**: 4-byte big-endian length prefix + JSON payload (NOT newline-delimited).
Max message size: 100 MB.

**Worker → Broker message types:**

| `type`         | Sender        | Fields                              | Effect |
|---------------|---------------|-------------------------------------|--------|
| `get_task`    | GenericWorker | `queue_name`, `worker_id`           | Broker dequeues and returns `task` or `no_task` |
| `forward`     | Any worker    | `message` (inner message dict)      | Broker calls `forward()` on inner message |
| `register_ack`| PGReaderWorker| `task_id`, `worker_id`              | Broker registers PG reader for ack on completion |

**Broker → Worker message types:**

| `type`    | Fields                       | Meaning |
|-----------|------------------------------|---------|
| `task`    | `message`                    | Task message for worker to handle |
| `no_task` | —                            | Queue empty, worker should sleep 0.5s |
| `shutdown`| —                            | Worker should exit |
| `ack`     | `task_id`, `negative?`       | Task completed (or failed if `negative=True`) |

## Inner Message Classes (the DAG)

All inner messages travel inside `forward` envelopes. Fields common to all:
`class`, `target_queue`, `task_id`.

| `class`                     | Queue       | Handler                        | Produces |
|-----------------------------|-------------|--------------------------------|----------|
| `tune_kernel`               | gpu_queue   | `TuneKernelHandler`            | `preprocess` |
| `preprocess`                | gpu_queue   | `PreprocessHandler(gpu_id)`    | `probe` or `mark_task_failed` |
| `probe`                     | gpu_queue   | `ProbeHandler(gpu_id)`         | `tune_hsaco` × N + `postprocess` (blocked) or `mark_task_failed` |
| `tune_hsaco`                | gpu_queue   | `TuneHsacoHandler(gpu_id)`     | `hsaco_result` |
| `hsaco_result`              | cpu_queue   | `WriteHsacoResultHandler(conn)`| None (resolves `postprocess` dep) |
| `postprocess`               | cpu_queue   | `PostprocessHandler(conn)`     | `tune_kernel_ack` |
| `mark_task_failed`          | cpu_queue   | `MarkTaskFailedHandler(conn)`  | `tune_kernel_ack` (negative) |
| `graceful_cancel_running_task` | cpu_queue| `GracefulCancelRunningTaskHandler(conn)` | None |
| `tune_kernel_ack`           | —           | Broker special-cases           | Sends `ack` to PGReaderWorker |

## Dependency Tracking

`postprocess` is enqueued by `ProbeHandler` with `depends: ['hsaco_result']`.
The broker holds it in `blocked_messages['hsaco_result']` until all expected
hsacos are received.

**Tracking structure in `postprocess` message:**
```python
{
    'expected_hsacos': {kname: [hsaco_index, ...]},   # set by ProbeHandler
    'received_hsacos': {kname: {hsaco_index: report}} # accumulated by resolve_dependency()
}
```

`PostprocessHandler.resolve_dependency()` is called in the **broker context**
(with `db_conn=None`) — it only mutates message dicts. `handle()` runs in the
**CPU worker context** with a real `db_conn`.

The broker hardcodes `PostprocessHandler(db_conn=None)` for dependency
resolution (see `broker.py:_resolve_dependencies`). A TODO exists to split
into `BrokerPostprocessTracker` + `WorkerPostprocessHandler`.

## Throttling

Each `PGReaderWorker` fetches one task, sends `tune_kernel`, then **blocks**
waiting for `ack`. Run M PGReaderWorker processes to allow M tasks in-flight.

```
PGReader: fetch → register_ack → forward(tune_kernel) → block on recv()
    ...DAG runs...
PostprocessHandler.handle() → mark_completed() → return tune_kernel_ack
Broker: forward(tune_kernel_ack) → _handle_ack() → send ack to PGReader socket
PGReader: unblocks → fetch next task
```

Negative acks (`negative=True`) are sent when a task fails — PGReader unblocks
and fetches next task.

## Error Handling

**Preprocess/Probe failure (GPU worker):** GPU workers have no DB connection.
`PreprocessHandler` and `ProbeHandler` catch `OSError`/`ExaidSubprocessNotOK`
and return a `mark_task_failed` message routed to `cpu_queue`.
`MarkTaskFailedHandler` (CPU worker) calls `task_queue.mark_failed()` and
returns a negative `tune_kernel_ack`.

**tune_hsaco failure:** `TuneHsacoHandler` catches exceptions internally and
returns `hsaco_result` with `result='crash'` or `result='NotOK'`. Results are
always written to DB; the task itself completes normally.

**Unhandled GPU exception:** `GenericWorker._handle_task()` catches and logs;
GPU workers lack `db_conn` so the task stays `running` forever. GPU handlers
must catch their own exceptions (see GPU handler comment in `generic_worker.py`).

## Graceful Shutdown

1. SIGTERM → `GenericWorker.shutdown()` / `PGReaderWorker.shutdown()` sets
   `running=False`, wakeup_fd triggers `select()` exit.
2. SIGHUP → `broker.graceful_shutdown()` sets `graceful_shutdown_requested=True`
   (does NOT stop the event loop).
3. After the broker's run loop exits, `_teardown_blocked_messages()` calls
   `PostprocessHandler.teardown_with_unmet_dependency()` for each blocked
   `postprocess` message, which returns `graceful_cancel_running_task`.
4. `GracefulCancelRunningTaskHandler` calls `task_queue.mark_pending()` so
   tasks re-enter the queue on next run.

## Database Integration

All DB writes go through `pq/` functions — no raw SQL in `localq/`. JSONB
columns use psycopg's `Jsonb` wrapper, not `json.dumps()`.

- `WriteHsacoResultHandler` → `save_tuning_result(task_id, report, conn)`
  → writes to `tuning_results` (`result_data JSONB`, `error JSONB`)
- `PostprocessHandler.handle()` → `task_queue.mark_completed(task_id, arch)`
  → updates `task_queue.status = 'completed'`
- **`tmpdir` is never written to the database**

**Schema notes:**
- `tuning_results` has `task_id BIGINT NOT NULL` (logical FK to `task_queue.id`,
  no explicit FOREIGN KEY constraint due to partitioning)
- No `tuning_hsaco_results` table — the original design name; actual table is
  `tuning_results`

## Queue Priority

Messages in `gpu_queue` and `cpu_queue` are inserted in priority order:

| Priority | Class |
|----------|-------|
| 4 | `postprocess` (frees resources, unblocks PGReader) |
| 3 | `probe` (fans out tune_hsaco work) |
| 2 | `tune_hsaco` (GPU benchmarking) |
| 1 | `preprocess` (setup) |
| 0 | `hsaco_result` (CPU DB write) |

## Key Implementation Files

| File | Role |
|------|------|
| `broker.py` | Epoll event loop, queue routing, dependency + ack tracking |
| `buffered_socket.py` | Non-blocking socket state machine (broker side) |
| `protocol.py` | `send_message`/`recv_message` — blocking, length-prefix framing |
| `handlers.py` | All `MessageHandler` subclasses |
| `generic_worker.py` | Worker loop: get_task → handle → forward |
| `cpu_worker.py` | CPU worker entrypoint (WriteHsacoResult, Postprocess, etc.) |
| `gpu_worker_socket.py` | GPU worker entrypoint (Preprocess, Probe, TuneHsaco) |
| `pg_reader_worker.py` | PGReaderWorker: fetch → throttle → ack |
| `broker_main.py` | Broker process entrypoint |
| `heartbeat_main.py` | Worker heartbeat reporter |
