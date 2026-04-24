# Tuner v3.5 PostgreSQL Queue

PostgreSQL-based distributed task queue for AOTriton tuning, replacing Celery + RabbitMQ.

## Features

- **PostgreSQL-only**: No RabbitMQ dependency, simpler infrastructure
- **Architecture partitioning**: Efficient querying with per-architecture table partitions
- **Atomic task claiming**: `SELECT FOR UPDATE SKIP LOCKED` prevents race conditions
- **Exponential backoff**: Workers poll aggressively when busy, back off when idle
- **Worker heartbeats**: Track worker health and detect dead workers
- **Comprehensive monitoring**: SQL views for progress, ETA, worker health

## Architecture

```
┌─────────────────────────────────────────┐
│    PostgreSQL (Central DB)             │
│  Tables:                                │
│    - task_queue (partitioned by arch)  │
│    - worker_heartbeat                   │
└─────────────────────────────────────────┘
                 ↓
      ┌──────────────────┐
      │   Worker Node    │
      │  ┌────────────┐  │
      │  │ Fetchers   │  │  (SELECT FOR UPDATE SKIP LOCKED)
      │  └──────┬─────┘  │
      │         ↓        │
      │  ┌────────────┐  │
      │  │ Ray/Local  │  │  (task execution)
      │  └────────────┘  │
      └──────────────────┘
```

## Components

### 1. TaskQueue (`queue.py`)
Core queue operations:
- `fetch_tasks(arch, batch_size)` - Atomic task claiming
- `mark_completed(task_id, arch)` - Mark task as done
- `mark_failed(task_id, arch, error)` - Mark task as failed
- `retry_task(task_id, arch)` - Retry failed task
- `get_queue_stats(arch)` - Get queue statistics
- `detect_stale_tasks()` - Find long-running tasks
- `reset_stale_tasks()` - Reset stuck tasks

### 2. TaskDispatcher (`dispatcher.py`)
Bulk task insertion:
- `dispatch_bulk(tasks)` - Efficient batch INSERT
- `dispatch_single(arch, module, config)` - Single task dispatch
- `ensure_partition(arch)` - Create partition if needed

### 3. Worker (`worker.py`)
Worker loop with exponential backoff:
- Polls queue for tasks
- Executes tasks via provided executor function
- Updates heartbeat
- Backs off exponentially when idle
- Graceful shutdown on SIGINT/SIGTERM

### 4. HeartbeatManager (`heartbeat.py`)
Worker health tracking:
- `update(status)` - Update heartbeat timestamp
- `increment_completed()` - Track successful tasks
- `increment_failed()` - Track failed tasks
- `mark_dead()` - Graceful shutdown marker
- `cleanup_dead_workers()` - Mark stale workers as dead

### 5. QueueAdmin (`admin.py`)
Administrative operations:
- `init_schema()` - Initialize database schema
- `create_partition(arch)` - Create architecture partition
- `reset_stale_tasks()` - Reset stuck tasks
- `cleanup_dead_workers()` - Clean up dead workers
- `purge_completed()` - Delete old completed tasks
- `get_statistics()` - Overall queue statistics

## Usage

### Initialize Schema

```python
from v3python.tune.pq.admin import QueueAdmin

conn_params = {
    'host': 'localhost',
    'port': 5432,
    'user': 'aotriton',
    'password': 'password',
    'dbname': 'aotriton'
}

admin = QueueAdmin(conn_params)
admin.init_schema()
admin.create_partitions(['gfx942', 'gfx90a', 'gfx1100'])
```

### Dispatch Tasks

```python
from v3python.tune.pq import TaskDispatcher

dispatcher = TaskDispatcher(conn_params)

tasks = [
    {
        'arch': 'gfx942',
        'module': 'attn_fwd',
        'task_config': {'BATCH': 4, 'H': 32, 'N_CTX': 1024, 'D_HEAD': 64},
        'priority': 5
    },
    # ... more tasks
]

count = dispatcher.dispatch_bulk(tasks)
print(f"Dispatched {count} tasks")
```

### Run Worker

**Important**: Each Worker process executes tasks **sequentially**. For parallel execution, launch **multiple Worker processes** per node (similar to Celery worker design).

#### Single Worker (for testing)

```python
from v3python.tune.pq import Worker

def my_task_executor(task):
    """Execute tuning task"""
    # Your task execution logic here
    print(f"Executing {task.module} with config {task.task_config}")
    return result

worker = Worker(
    conn_params=conn_params,
    arch='gfx942',
    executor=my_task_executor,
    batch_size=10,
    poll_interval=1.0,
    max_poll_interval=30.0
)

worker.start()  # Runs until SIGINT/SIGTERM
```

#### Production: Multiple Workers

For production deployments with multiple GPUs, launch **4-8 workers per node**:

```bash
# Using the worker management script
.tune/remote/worker_service.sh start <workdir> <arch> <num_workers>

# Example: Start 8 workers for gfx942
.tune/remote/worker_service.sh start /path/to/workdir gfx942 8

# Stop workers
.tune/remote/worker_service.sh stop /path/to/workdir gfx942

# Restart workers
.tune/remote/worker_service.sh restart /path/to/workdir gfx942 8

# Check status
.tune/remote/worker_service.sh status /path/to/workdir gfx942

# Force stop (SIGKILL)
.tune/remote/worker_service.sh force-stop /path/to/workdir gfx942
```

**Why multiple workers?**
- Each Worker executes tasks sequentially (blocking)
- Multiple workers = parallel task execution
- Recommended: 1-2 workers per GPU (e.g., 8 GPUs = 8-16 workers)
- Each worker fetches batches independently from the queue

### Query Statistics

```python
from v3python.tune.pq import TaskQueue

queue = TaskQueue(conn_params)

# Overall stats
stats = queue.get_queue_stats()
print(f"Pending: {stats['pending']}, Running: {stats['running']}")

# Architecture-specific stats
gfx942_stats = queue.get_queue_stats('gfx942')
print(f"gfx942: {gfx942_stats}")
```

### Monitor Progress

```python
# Using SQL views
import psycopg

conn = psycopg.connect(**conn_params)
cur = conn.cursor()

# Progress by architecture
cur.execute("SELECT * FROM queue_progress")
for row in cur.fetchall():
    print(row)

# Worker health
cur.execute("SELECT * FROM worker_health")
for row in cur.fetchall():
    print(row)

# Task duration statistics
cur.execute("SELECT * FROM task_timing_stats")
for row in cur.fetchall():
    print(row)

# Estimated completion time
cur.execute("SELECT * FROM completion_eta")
for row in cur.fetchall():
    print(row)
```

## Example Script

See `example.py` for a complete working example:

```bash
# Initialize schema and partitions
python v3python/pq/example.py init

# Dispatch example tasks
python v3python/pq/example.py dispatch

# Run a worker
python v3python/pq/example.py worker

# Show statistics
python v3python/pq/example.py stats
```

## Database Schema

### task_queue (partitioned by arch)

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Task ID (unique within partition) |
| arch | TEXT | GPU architecture (partition key) |
| module | TEXT | Tuning module name |
| task_config | JSONB | Task configuration |
| status | TEXT | pending/running/completed/failed/cancelled — see below |
| priority | INT | Task priority (higher = more urgent) |
| worker_id | TEXT | Worker that claimed this task |
| node_hostname | TEXT | Node hostname |
| created_at | TIMESTAMP | Task creation time |
| started_at | TIMESTAMP | Task start time |
| completed_at | TIMESTAMP | Task completion time |
| error | TEXT | Error message (if failed); cancellation reason (if cancelled) |
| retry_count | INT | Number of retries |

**Task status values:**

| Status | Description |
|--------|-------------|
| `pending` | Waiting to be claimed by a worker |
| `running` | Currently being processed by a worker |
| `completed` | Finished successfully |
| `failed` | Terminated with an error; eligible for retry |
| `cancelled` | Deliberately skipped; kept for archive purposes only. Not an error — workers never claim cancelled tasks and reset scripts must not touch them |

**Indexes:**
- `(status, priority DESC, id ASC) WHERE status = 'pending'` - Fast task fetching
- `(worker_id, status)` - Worker task lookups
- `(created_at DESC)` - Time-based queries

### worker_heartbeat

| Column | Type | Description |
|--------|------|-------------|
| worker_id | TEXT | Unique worker ID (hostname-pid) |
| node_hostname | TEXT | Node hostname |
| arch | TEXT | GPU architecture |
| last_heartbeat | TIMESTAMP | Last heartbeat timestamp |
| status | TEXT | active/idle/dead |
| tasks_completed | INT | Total completed tasks |
| tasks_failed | INT | Total failed tasks |

**Index:**
- `(last_heartbeat DESC) WHERE status = 'active'` - Active worker queries

## Performance Characteristics

### Task Dispatch
- **Bulk INSERT**: 10,000 tasks in ~0.5-1 second
- **2-4x faster** than individual Celery dispatches
- Uses psycopg `executemany` for efficient batching

### Task Fetching
- **Partitioned queries**: O(1) regardless of total queue size
- **Atomic claiming**: No race conditions with `FOR UPDATE SKIP LOCKED`
- **Batch fetching**: Fetch 10+ tasks per query for efficiency

### Worker Polling
- **Active phase**: Poll every 1 second when tasks available
- **Idle phase**: Exponential backoff to 30 seconds
- **Network reduction**: 90%+ less traffic vs constant polling

### Lock Contention
- **Eliminated**: Per-architecture partitions = independent locks
- **SKIP LOCKED**: Non-blocking fetch even with concurrent workers

## Migration from Celery

1. **Initialize schema**: `admin.init_schema()`
2. **Create partitions**: `admin.create_partitions(['gfx942', ...])`
3. **Update dispatcher**: Replace `task.apply_async()` with `dispatcher.dispatch_bulk()`
4. **Update workers**: Replace Celery worker with `Worker` class
5. **Test thoroughly**: Verify task execution and monitoring
6. **Gradual rollout**: Switch workers incrementally
7. **Remove Celery**: Once stable, remove Celery + RabbitMQ

## Monitoring

### SQL Views
- `queue_progress` - Task counts by architecture
- `worker_health` - Worker status and activity
- `task_timing_stats` - Duration statistics
- `stale_tasks` - Long-running tasks
- `completion_eta` - Estimated completion time

### Grafana Integration
See `docs/PostgreSQL-Only Queue Evaluation.md` for Grafana dashboard setup.

### Web Dashboard
Integrate with `.tune/webui` - see `docs/Tuner v3.5 WebUI Integration Plan.md`.

## Maintenance

### Cleanup Dead Workers
```python
from v3python.tune.pq.admin import QueueAdmin

admin = QueueAdmin(conn_params)
count = admin.cleanup_dead_workers(threshold_seconds=300)
print(f"Marked {count} workers as dead")
```

### Reset Stale Tasks
```python
count = admin.reset_stale_tasks(timeout_seconds=7200)
print(f"Reset {count} stale tasks")
```

### Purge Old Completed Tasks
```python
count = admin.purge_completed(older_than_hours=24)
print(f"Purged {count} old tasks")
```

## Dependencies

- `psycopg` (v3) - PostgreSQL driver
- Python 3.9+

Install:
```bash
pip install psycopg[binary]
```

## License

Copyright © 2026 Advanced Micro Devices, Inc.  
SPDX-License-Identifier: MIT
