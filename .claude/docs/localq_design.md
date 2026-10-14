# Local Queue: Unix Socket Runtime (`aotriton.pgdef.localq`)

The per-node runtime of pgdef. It is application-independent: an application
(today tune; perfmon is planned) hands it handlers, runner types and an `enrich` callable
through `pgdef.localq.node.run_node_role`. See `.claude/docs/pgdef_design.md`
for the decisions and `docs/Distributed Execution Framework.md` for the human
overview. The code lives in `python/pgdef/localq/`; tune's graph is
`python/tune/handlers.py`.

## Process topology

```
PG readers (one task each) ──> broker ──> GPU workers ──> runner per GPU (exaid)
                                     └──> CPU workers ──> database writes
```

One broker, R readers, N GPU workers, M CPU workers, all connected by one Unix
socket, all processes of ONE application (`python -m aotriton.<app>.node
<role>`). N and M may be zero: `broker --queues cpu` for a zero-GPU node. A
message for a queue that has no workers fails its task (the broker sends the
reader a negative ack carrying the reason; the reader records the permanent
failure) instead of hanging.

* **Broker** (`broker.py`): single-process epoll loop; non-blocking sockets
  with `BufferedSocket` framing. Knows no application: it is built with the
  handler classes (class -> handler, for joins and teardown), their
  `PRIORITY`, and the queue names that have workers.
* **Workers** (`generic_worker.py`): blocking sockets, `select` + wakeup pipe.
  A GPU worker has `ctx.gpu_id` and `ctx.runner(name, *key)`; a CPU worker has
  `ctx.db_conn`.
* **PG reader** (`pg_reader.py`): claims a task, sends the start message, holds
  the task until acked, renewing its lease.

## Wire protocol

4-byte big-endian length prefix + JSON (`protocol.py`), NOT newline JSON.
Worker -> broker: `get_task{queue_name, worker_id}`, `forward{message}`,
`register_ack{task_id, worker_id}`. Broker -> worker: `task{message}`,
`no_task`, `ack{task_id, negative?, error?}`.

## The handler graph: implicit, but readable

A handler's return value is the next message. No DAG language. Each handler
declares, next to `handle()`:

```python
class ProbeHandler(MessageHandler):
    CLASS = 'tune/probe'            # message class handled
    QUEUE = 'gpu'                   # which workers run it
    EMITS = ('tune/impl', 'tune/postprocess')
    PRIORITY = 3                    # broker queue priority (higher = sooner)
    def handle(self, msg, ctx): ...
```

* `GenericWorker.prepare_results` checks every returned message against
  `EMITS` (`EmitsViolation` otherwise) and fills `target_queue` from the
  target handler's `QUEUE`; handlers never write a queue name.
* `HandlerSet` (built from the application's classes plus the engine's) checks
  at node start that every `EMITS` entry names a handler or `pgdef/dag_ack`.
* `python -m aotriton.pgdef.localq.graph <app>` prints the graph from the
  `EMITS` declarations (it imports `aotriton.<app>.handlers`: a developer tool,
  the only by-name import in pgdef).
* Messages carry a header (`task_id, pool, subpool, category, attempt,
  payload`); `child(msg, cls, **fields)` copies it. The start message adds
  `appcookie` and `enrich`.

Engine classes (`pgdef/`): `task_failed`, `graceful_cancel` (handlers) and
`dag_ack` (consumed by the broker to release the reader).

## Joins: fan-in

`pgdef.localq.semaphore`: the producer builds the join with
`semaphore.wait('app/result', keys, join_msg)`, i.e. wait(depends, keys,
result_msg) (no keys: not blocked);
each parallel result calls `semaphore.signal(msg, key, value)`; the broker
resolves with `MessageHandler.resolve_dependency` (default `semaphore.resolve`,
runs on the CLASS in the broker: no database, no runner); the released handler
reads `semaphore.signals(msg)`. Emit the join BEFORE the parallel messages so it is
blocked before any result can arrive.

## Failure: one path

`GenericWorker` catches any exception from `handle()`, on any queue, and sends
`task_failed{error, permanent, origin}` to the CPU queue. `Permanent` (bad
input) fails the task at once; anything else fails the attempt and the task
retries while `logistics.retry.max_attempts` allows. `TaskFailedHandler` calls
`pq.tasks.fail`, runs the start handler's `cleanup(origin, ctx)`, and sends a
negative `dag_ack` so the reader slot is never lost. If the failure path itself
fails (database down), the worker sends only the negative ack: the task keeps a
lease nobody renews, so it lapses and is reaped.

## Leases and the reader

* `tasks.claim` bumps `logistics.attempt` and stamps `lease.until` (database
  clock).
* While it holds a task the reader renews every `lease_s / 3`, but only while
  its broker connection is alive (broker gone -> reader exits -> lease lapses).
  A failed renewal (reaped/reset/cancelled) stops further renewals.
* Before each claim the reader runs a rate-limited `tasks.reap_expired(pool)`
  (about once a minute): expired `running` rows go back to `pending`, or to
  `failed` when out of attempts. No daemon, no manual reset.
* The reader upserts `pgdef.worker` on claim, renewal and idle poll.
* Negative ack with an `error` (set only by the broker for undeliverable
  messages) makes the reader call `tasks.fail(permanent=True)` itself.

## Throttling

R readers = R tasks in flight per node. A reader fetches one task, sends the
start message, blocks until `dag_ack`, then fetches the next.

## Graceful shutdown

SIGTERM stops a worker/reader. SIGHUP makes the broker tear down blocked joins
once its loop ends: `teardown_with_unmet_dependency` (default: a
`graceful_cancel` message, so `tasks.release` puts the task back to pending
WITHOUT consuming an attempt). Applications override it to clean scratch state
and call `super()`.

## Files

| File | Role |
|------|------|
| `broker.py` | epoll loop, queues, dependency + ack tracking, undeliverable handling |
| `buffered_socket.py`, `protocol.py` | framing |
| `handlers.py` | `MessageHandler`, `StartHandler`, `HandlerSet`, `NodeContext`, engine handlers |
| `semaphore.py` | join helper |
| `generic_worker.py` | worker loop, `EMITS` check, failure path |
| `pg_reader.py` | claim, throttle, lease renewal, reaping |
| `node.py` | `run_node_role(app, handlers, start, runners, enrich, argv)` |
| `graph.py` | prints a graph |

Tests: `python/test/pgdef/test_localq.py` runs the real broker, workers and
reader in threads against an in-memory fake of the pq calls
(`python/test/pgdef/localq_harness.py`); `python/tune/tests/test_tune_graph.py` does
the same with tune's handlers and a scripted fake runner.
