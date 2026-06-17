# Local Queue Architecture

Unix socket-based message broker for GPU kernel tuning workloads. Replaces Ray ActorPool with simpler Unix domain socket IPC.

## Overview

The local queue system implements a **message-passing architecture** where:
- A central **broker** routes messages between workers via named queues
- Workers are **generic** - they pull tasks, execute handlers, and forward results
- The **DAG is implicit** - formed by handlers forwarding result messages to appropriate queues
- **Dependencies** are tracked by the broker - postprocess waits for all hsaco_result messages
- **PG readers** throttle task fetching by blocking until tune_kernel completes (ACK mechanism)

```
┌─────────────────────────────────────────────────────────────┐
│                      LocalBroker (broker.py)                 │
│  - Routes messages between workers via Unix sockets          │
│  - Manages named queues: gpu_queue, cpu_queue               │
│  - Tracks dependencies (postprocess waits for hsaco_result)  │
│  - Handles PG reader ACK mechanism for throttling           │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
    Unix socket          Unix socket          Unix socket
         │                    │                    │
         ↓                    ↓                    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ PG Reader    │    │ GPU Worker   │    │ CPU Worker   │
│ (pulls from  │    │ (pulls from  │    │ (pulls from  │
│  PostgreSQL) │    │  gpu_queue)  │    │  cpu_queue)  │
└──────────────┘    └──────────────┘    └──────────────┘
```

## DAG Workflow

The tuning workflow is a 5-step DAG, implicitly formed by message handlers:

```
tune_kernel (PG reader) 
    ↓
preprocess (GPU worker) - prepare test data
    ↓
probe (GPU worker) - discover hsaco kernels
    ↓
tune_hsaco (GPU workers, parallel) - benchmark each hsaco
    ↓
hsaco_result (CPU worker) - write result to DB
    ↓
postprocess (CPU worker) - aggregate & cleanup
    ↓
tune_kernel_ack (to PG reader) - unblock next task
```

## Architecture Components

### broker.py - LocalBroker

Message router with dependency tracking and priority queuing.

**Queues**:
- `gpu_queue`: preprocess, probe, tune_hsaco (handled by GPU workers)
- `cpu_queue`: hsaco_result, postprocess (handled by CPU worker)

**Dependency Resolution**:
The broker tracks postprocess messages that depend on `hsaco_result`. When a hsaco_result message arrives:
1. Broker calls `PostprocessHandler.resolve_dependency()` to accumulate the result
2. Once all expected hsaco_result messages arrive, postprocess is unblocked
3. Postprocess message is removed from `blocked_messages` and forwarded to cpu_queue

**Priority Ordering**:
Messages are enqueued with priority (higher = more urgent):
- `postprocess`: 4 (highest - frees resources, sends ACK)
- `probe`: 3 (generates tune_hsaco tasks)
- `tune_hsaco`: 2 (actual GPU work)
- `preprocess`: 1 (just setup)
- `hsaco_result`: 0 (lowest - CPU write)

**ACK Mechanism**:
PG readers register for ACK when sending tune_kernel. When postprocess completes, it sends `tune_kernel_ack` to the broker, which notifies the waiting PG reader to fetch the next task.

### generic_worker.py - GenericWorker

Generic worker base class that:
1. Connects to broker via Unix socket
2. Requests tasks from a specific queue (`get_task` message)
3. Dispatches to appropriate handler based on message class
4. Forwards handler results back to broker (`forward` message)
5. Marks failed tasks in database if handler raises exception

Workers don't know about the DAG structure - they just handle messages and forward results.

**Error Handling**:
When a handler raises an exception for top-level task messages (`tune_kernel`, `preprocess`, `probe`), GenericWorker marks the task as failed in the database with the error message.

### pg_reader_worker.py - PGReaderWorker

Fetches tasks from PostgreSQL `task_queue` and sends to broker. Uses ACK mechanism to throttle:

1. Fetch task from PostgreSQL (atomic claim with `SELECT FOR UPDATE SKIP LOCKED`)
2. Register for ACK (`register_ack` message)
3. Send `tune_kernel` message to broker (`forward` message)
4. **Block** waiting for `ack` response
5. Once ACK received, fetch next task

**Throttling**: Run M PG reader workers to have M tasks in-flight simultaneously.

**Database Connection**: Reuses persistent connection with `autocommit=True` and `statement_timeout=1000ms` to prevent blocking queries.

### handlers.py - Message Handlers

Handler classes for each message type:

**TuneKernelHandler**: Entry point, creates preprocess message

**PreprocessHandler**: Prepares test data using `exaid.prepare_data()`, returns probe message with updated task_config

**ProbeHandler**: Discovers hsaco kernels using `exaid.probe()`, returns:
- Multiple `tune_hsaco` messages (one per hsaco variant)
- One `postprocess` message with dependencies on `hsaco_result`

**TuneHsacoHandler**: Benchmarks single hsaco using `exaid.benchmark()`, returns `hsaco_result` message

**WriteHsacoResultHandler**: Writes hsaco result to `tuning_results` table, returns None (triggers dependency resolution)

**PostprocessHandler**: Aggregates all hsaco results after dependencies resolved, updates `task_queue` status to completed, cleans up tmpdir, returns `tune_kernel_ack`

### buffered_socket.py - BufferedSocket

Non-blocking buffered socket wrapper for efficient message I/O:
- **Send buffer**: Queue messages and flush incrementally when socket writable
- **Receive buffer**: Accumulate partial payloads and yield complete messages
- **Protocol**: Length-prefixed JSON (4-byte big-endian length + JSON payload)

### protocol.py - Message Protocol

Helper functions for message serialization:
- `send_message(sock, msg)`: Serialize message as length-prefixed JSON
- `recv_message(sock)`: Deserialize message from socket (blocking)

## Message Format

All messages are JSON objects with:
- `class`: Message class name (e.g., "tune_kernel", "preprocess", "probe")
- `target_queue`: Where this message should be dispatched (or None if not queued)
- `depends`: Optional list of dependency class names (for postprocess)
- `task_id`: Task ID from PostgreSQL task_queue
- Additional fields specific to message class

Example:
```json
{
  "class": "tune_kernel",
  "target_queue": "gpu_queue",
  "task_id": 42,
  "task_config": {"module": "attn_fwd", "arch": "gfx942", ...}
}
```

## Worker Types

### GPU Workers (gpu_worker_main.py)

Handle GPU operations: preprocess, probe, tune_hsaco

Handlers:
- `TuneKernelHandler`
- `PreprocessHandler(gpu_id)`
- `ProbeHandler(gpu_id)`
- `TuneHsacoHandler(gpu_id)`

### CPU Worker (cpu_worker_main.py)

Handle database writes and postprocessing

Handlers:
- `WriteHsacoResultHandler(db_conn)`
- `PostprocessHandler(db_conn)`

## Database Integration

### Connection Management

All database-accessing classes accept a connection object (not connection parameters):

```python
# GOOD - reuses connection
task_queue = TaskQueue(db_conn)
handler = PostprocessHandler(db_conn)

# BAD - creates new connection per object
task_queue = TaskQueue(conn_params)  # Don't do this
```

Connection is created once per worker and reused for all operations.

### Error Tracking

When handlers fail, GenericWorker marks tasks as failed in `task_queue`:
```sql
UPDATE task_queue
SET status = 'failed',
    completed_at = NOW(),
    error = <error_message>
WHERE id = <task_id>
```

This applies to top-level handlers: `tune_kernel`, `preprocess`, `probe`.

### Schema

**task_queue**: Partitioned by arch (e.g., task_queue_gfx942)
- Columns: id, arch, module, task_config (JSONB), status, priority, worker_id, node_hostname, created_at, started_at, completed_at, error, retry_count

**tuning_results**: Stores individual hsaco benchmark results
- Columns: id, task_id (FK to task_queue), kernel_name, hsaco_index, result, result_data (JSONB), error (JSONB), gpu_id, created_at

## Signal Handling and Graceful Shutdown

All components support graceful shutdown via SIGTERM/SIGINT:

1. **Signal Handlers**: Registered for SIGTERM and SIGINT
2. **Wakeup FD**: `signal.set_wakeup_fd()` interrupts blocking I/O (socket recv, select)
3. **Running Flag**: Signal handler sets `self.running = False`
4. **Cleanup**: Workers close sockets and database connections before exiting

**No zombies**: Workers wait for child processes. Docker containers use `--init` flag for PID 1 zombie reaping.

## Design Issues and Caveats

### PostprocessHandler Dual-Context Usage

**Issue**: `PostprocessHandler` serves two different roles in two different contexts:

1. **Broker Context**: The broker instantiates `PostprocessHandler(db_conn=None)` to call `resolve_dependency()` for tracking which hsaco_result messages have arrived. This runs in the broker process.

2. **CPU Worker Context**: The CPU worker instantiates `PostprocessHandler(db_conn=<valid_conn>)` to call `handle()` for actual postprocessing work. This runs in the CPU worker process.

**Implication**: There are effectively two "copies" of the postprocess message state:
- One in the broker's `blocked_messages` dict, tracking `received_hsacos`
- One in the CPU worker's handler, executing the final aggregation

**Why This Works**: 
- `resolve_dependency()` only manipulates message dictionaries and doesn't need database access
- `handle()` runs after all dependencies are resolved and needs database access

**Caution**: 
- Do NOT access `self.db_conn` in `resolve_dependency()` - it will be `None` in broker context
- If modifying `resolve_dependency()`, remember it runs without database access

**Future Refactoring**: Consider splitting into:
- `BrokerPostprocessTracker` - handles dependency resolution
- `WorkerPostprocessHandler` - handles actual postprocessing work

See detailed comments in `handlers.py` for more information.

### Dependency Tracking Details

The postprocess message tracks expected hsacos as a dict:
```python
{
  'expected_hsacos': {
    'kernel_name_1': [0, 1, 2],  # hsaco indices
    'kernel_name_2': [0, 1]
  },
  'received_hsacos': {
    'kernel_name_1': {
      0: <report>,
      1: <report>,
      2: <report>
    },
    'kernel_name_2': {
      0: <report>,
      1: <report>
    }
  }
}
```

This allows the broker to know which specific hsacos are pending, not just a count.

## Logging and Debugging

### Log Flushing

All components use line-buffered logging with immediate flush:
```python
from v3python.tune.utils import configure_logging_with_flush
configure_logging_with_flush()
```

This ensures logs appear immediately, critical for debugging blocking issues.

### Log Levels

- **INFO**: Task lifecycle events (task received, handler completed, ACK sent)
- **DEBUG**: Message routing, queue operations (only enabled when debugging)
- **ERROR**: Handler failures, database errors, connection errors

### Reducing Noise

Workers only log when tasks are in progress. Idle polling (get_task → no_task) does not spam logs.

### Debugging Socket Issues

All message send/recv can be logged:
```python
logger.debug(f"→ SEND to broker: {json.dumps(msg)}")
logger.debug(f"← RECV from broker: {json.dumps(msg)}")
```

## Performance Characteristics

### Parallelism

- **Preprocessing/Probing**: Any available GPU worker (no bottleneck on GPU 0)
- **HSACO Tuning**: All GPUs in parallel (round-robin distribution)
- **Database Writes**: Single CPU worker (serial writes)
- **Postprocessing**: Single CPU worker (cleanup)

### Overhead

- **Worker Pool Creation**: ~2-10s per module (exaid initialization), amortized across tasks
- **Socket Message Dispatch**: <1ms per message
- **DAG Coordination**: Minimal (broker just routes messages)

### Scalability

- **GPUs**: Linear scaling (8 GPUs = 8× throughput for hsaco tuning)
- **Tasks**: Handles 100-1000 hsaco variants per tuning task
- **Throttling**: Control in-flight tasks by number of PG reader workers

## Advantages Over Ray

| Aspect | Ray (old) | Unix Sockets (new) |
|--------|-----------|-------------------|
| **Debugging** | Opaque Ray internals | Application-level logs only |
| **Complexity** | Ray cluster, ActorPool | Broker + workers |
| **Timeouts** | Hidden in ActorPool.get_next() | Explicit socket timeouts |
| **Error Propagation** | Ray exception handling | Direct exception logging |
| **Dependencies** | `ray` package (~500MB) | Python stdlib only |
| **Process Management** | Ray actors | Standard subprocesses |
| **Network Overhead** | Ray object store | Unix domain sockets (in-memory) |

## Running the Local Queue

### Start Broker

```bash
python -m v3python.tune.localq.broker_main --socket_path /tmp/aotriton-broker.sock
```

### Start PG Readers (M workers for M tasks in-flight)

```bash
for i in {0..3}; do
  python -m v3python.tune.localq.pg_reader_worker \
    --worker_id pg-reader-gfx942-$i \
    --arch gfx942 \
    --workdir /path/to/workdir \
    --broker_socket /tmp/aotriton-broker.sock &
done
```

### Start GPU Workers (1 per GPU)

```bash
for gpu_id in {0..7}; do
  python -m v3python.tune.localq.gpu_worker_main \
    --worker_id gpu-worker-$gpu_id \
    --gpu_id $gpu_id \
    --workdir /path/to/workdir \
    --broker_socket /tmp/aotriton-broker.sock &
done
```

### Start CPU Worker

```bash
python -m v3python.tune.localq.cpu_worker_main \
  --worker_id cpu-worker-0 \
  --workdir /path/to/workdir \
  --broker_socket /tmp/aotriton-broker.sock &
```

### Stop All Workers

Send SIGTERM for graceful shutdown:
```bash
pkill -TERM -f "v3python.tune.localq"
```

## Future Enhancements

1. **Task Prioritization**: Priority-based queue ordering (already implemented)
2. **Adaptive Batching**: Dynamically adjust PG reader count based on queue depth
3. **Fault Tolerance**: Detect stale tasks and reassign to other workers
4. **Multi-Node**: Distribute workers across multiple machines (TCP sockets)
5. **Profiling**: Per-task performance metrics (handler execution time)
