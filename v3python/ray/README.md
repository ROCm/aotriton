# Ray-Based Node-Local Task Execution

Ray-based framework for GPU-exclusive task execution with proper DAG orchestration.
Replaces local Celery queues (CPUQ, GPUQ) with Ray actors.

## Overview

This framework handles the full tuning workflow with a **shared Ray cluster**:

```
                    Ray Cluster (shared, 1:1 GPU mapping)
                    ┌─────────────────────────────────────┐
                    │  GPU Worker Pool (N workers)        │
                    │  ├─ GPU 0 Worker (exclusive)        │
                    │  ├─ GPU 1 Worker (exclusive)        │
                    │  └─ GPU N Worker (exclusive)        │
                    └─────────────────────────────────────┘
                              ↑       ↑       ↑
                              │       │       │
        ┌─────────────────────┴───────┴───────┴──────────────┐
        │  Multiple worker_main.py instances (share cluster) │
        │  ├─ Worker 0: fetch task → submit to Ray           │
        │  ├─ Worker 1: fetch task → submit to Ray           │
        │  └─ Worker N: fetch task → submit to Ray           │
        └─────────────────────────────────────────────────────┘
                              ↑
                              │
                    PostgreSQL Queue (central)
    
Each task execution:
    1. Preprocess (any GPU, Ray ActorPool auto-selects least-loaded, exclusive)
         ↓
    2. Probe (any GPU, ActorPool auto-selects, exclusive) → discovers N hsaco kernels
         ↓
    3. Distribute tune_hsaco tasks across all GPUs (round-robin)
         ├→ GPU 0: tune_hsaco[0] (exclusive)
         ├→ GPU 1: tune_hsaco[1] (exclusive)
         ├→ GPU 2: tune_hsaco[2] (exclusive)
         └→ GPU N: tune_hsaco[N] (exclusive)
              ↓
    4. CPU db_writer tasks (parallel I/O)
         ↓
    5. Postprocess (CPU, cleanup)
```

## Key Features

### GPU Exclusivity

**Problem**: All tuning operations (preprocess, probe, tune_hsaco) need exclusive GPU access for accurate benchmarking.

**Solution**: Ray actors with `num_gpus=1` reservation:
- Each `GPUWorker` actor owns one GPU exclusively
- Ray ensures actor methods run **serially** (one at a time)
- Different GPUs run in **parallel**

```python
@ray.remote(num_gpus=1)
class GPUWorker:
    def preprocess(self, config):
        # Runs exclusively on this GPU
        ...
    
    def probe(self, config):
        # Runs exclusively on this GPU (after preprocess)
        ...
    
    def tune_hsaco(self, config, kname, hsaco_idx):
        # Runs exclusively on this GPU
        ...
```

**Timeline Example (single task)**:
```
GPU 2: preprocess (0-10s) → probe (10-15s) → tune_hsaco[2] (15-20s) → tune_hsaco[10] (20-25s)
GPU 0: [idle]             → [idle]         → tune_hsaco[0] (15-20s) → tune_hsaco[8] (20-25s)
GPU 1: [idle]             → [idle]         → tune_hsaco[1] (15-20s) → tune_hsaco[9] (20-25s)
CPU:   [idle]             → [idle]         → [idle]                 → db_writers + postprocess
```

**Timeline Example (3 concurrent tasks)**:
```
Task A uses GPU 0 for preprocess/probe, Task B uses GPU 1, Task C uses GPU 2
GPU 0: Task A preprocess → Task A probe → tune_hsaco (A, B, C tasks interleaved)
GPU 1: Task B preprocess → Task B probe → tune_hsaco (A, B, C tasks interleaved)
GPU 2: Task C preprocess → Task C probe → tune_hsaco (A, B, C tasks interleaved)
```

This avoids bottlenecking all preprocess/probe operations on GPU 0.

### Offloaded Database Writes

**Problem**: GPU workers blocking on PostgreSQL writes wastes GPU time.

**Solution**: Separate CPU tasks for database writes:
```python
# GPU task returns report immediately (no blocking)
report = gpu_worker.tune_hsaco(config, kname, hsaco_idx)

# CPU task handles database write (parallel I/O)
db_writer_task.remote(task_id, report)
```

GPU workers can immediately start next benchmark while CPU tasks handle I/O.

### Persistent Worker Pool with 1:1 GPU Mapping

**Problem**: Creating exaid instances is expensive (seconds per GPU).

**Solution**: Single persistent worker pool with 1:1 GPU mapping:
```python
# Pool created once (1 worker per GPU)
gpu_workers = get_gpu_worker_pool(num_gpus=4)

# Workers reused across all tasks and modules
result1 = execute_tuning_dag(task1_id, task1_config)  # Uses existing pool
result2 = execute_tuning_dag(task2_id, task2_config)  # Reuses same pool
```

Workers can handle any module because `exaid_create()` is cached by (module, gpu_id) internally.

## Architecture

### Components

**`gpu_worker.py`** - GPUWorker Actor
- Owns one GPU exclusively (`num_gpus=1`, 1:1 mapping)
- Handles all GPU task types (preprocess, probe, tune_hsaco) for ANY module
- Calls `exaid_create(module, gpu_id)` per task (cached internally)
- Ray ensures methods run serially

**`cpu_tasks.py`** - CPU Tasks
- `db_writer_task` - Write one report to PostgreSQL
- `postprocess_task` - Aggregate results and cleanup tmpdir

**`worker_pool.py`** - Worker Pool Management
- Manages single persistent GPU worker pool (1:1 GPU mapping)
- Workers handle any module (exaid caching is per (module, gpu_id))
- Pool reused across all tasks and modules

**`orchestrator.py`** - DAG Orchestration
- `init_ray()` - Initialize Ray runtime
- `execute_tuning_dag()` - Execute full workflow with dependencies

### Data Flow

1. **Preprocess** (any GPU, Ray ActorPool auto-selects least-loaded):
   - Prepares test data in `/dev/shm/` (shared tmpfs accessible by all GPUs)
   - Returns updated config with tmpdir path

2. **Probe** (any GPU, ActorPool auto-selects, depends on preprocess):
   - Discovers hsaco kernel variants by reading from shared tmpdir
   - Returns list of (kernel_name, hsaco_index) tuples

3. **Tune HSACO** (distributed across GPUs):
   - Round-robin distribution
   - Each GPU task benchmarks one hsaco variant
   - Returns report (does NOT write to DB)

4. **Database Writers** (CPU tasks):
   - Parallel writes to PostgreSQL
   - Offloaded from GPU workers

5. **Postprocess** (CPU, depends on all GPU tasks):
   - Aggregates reports into summary
   - Cleans up tmpdir

## Usage

### Starting the Ray Cluster

Before starting workers, start the shared Ray cluster:

```bash
# Start Ray cluster (one per node, shared by all workers)
.tune/bin/rayctl <workdir> start

# Check status
.tune/bin/rayctl <workdir> status

# Stop cluster
.tune/bin/rayctl <workdir> stop
```

The Ray cluster uses NUM_GPUS from config.rc (default: 4) to create GPU worker pool.

### From Worker Process

Worker processes connect to the shared Ray cluster:

```python
# v3python/tune/worker_main.py

from v3python.ray import execute_tuning_dag, init_ray

init_ray()  # Connect to existing Ray cluster (address='auto')

# Execute task
result = execute_tuning_dag(task_id, task_config)
```

Multiple worker_main.py instances share the same GPU worker pool.

### Direct Usage

```python
from v3python.ray import init_ray, execute_tuning_dag

# Initialize Ray (once per process)
init_ray(num_gpus=8)

# Execute tuning DAG
task_config = {
    "module": "attn_fwd",
    "arch": "gfx942",
    "entry": {
        "BATCH": 4,
        "H": 32,
        "N_CTX": 1024,
        "D_HEAD": 64,
    },
    "max_hsaco": {"*": 10}  # Limit to 10 hsaco variants per kernel
}

result = execute_tuning_dag(task_id="test-001", task_config=task_config)

# Result contains:
# {
#     "task_config": {...},
#     "brief": {
#         "kernel_name_1": {0: "OK", 1: "OK", 2: "NotOK"},
#         "kernel_name_2": {0: "OK", 1: "crash"},
#     }
# }
```

### Worker Pool Management

```python
from v3python.ray.worker_pool import (
    get_gpu_worker_pool,
    shutdown_worker_pool,
    get_worker_pool_stats
)

# Get or create worker pool (1:1 GPU mapping, handles all modules)
workers = get_gpu_worker_pool(num_gpus=4)

# Get statistics
stats = get_worker_pool_stats()
# {'num_workers': 4, 'active': True}

# Shutdown pool
shutdown_worker_pool()
```

## GPU Exclusivity Guarantees

Ray provides **two levels of exclusivity**:

1. **Resource Reservation** (`num_gpus=1`):
   - Ray reserves GPU for this actor
   - No other actors can use this GPU
   - Enforced by Ray scheduler

2. **Serial Execution** (actor threading model):
   - Actor methods run **one at a time**
   - If GPU 0 is running `preprocess`, `probe` waits
   - If GPU 0 is running `tune_hsaco[0]`, next task waits

**Combined effect**: Each GPU runs exactly one task at a time, ensuring exclusive access for accurate benchmarking.

## Performance Characteristics

### Parallelism

- **Preprocessing/Probing**: Distributed across all GPUs (Ray ActorPool load balancing)
  - Multiple concurrent tasks can preprocess/probe in parallel on different GPUs
  - ActorPool automatically selects least-loaded GPU for each operation
  - No bottleneck on GPU 0
- **HSACO Tuning**: All GPUs in parallel (main workload)
- **Database Writes**: CPU tasks in parallel (don't block GPUs)
- **Postprocessing**: Single CPU task (cleanup)

### Overhead

- **Worker Pool Creation**: ~2-10s per module (exaid initialization)
  - But amortized across all tasks (pools are persistent)
- **Ray Task Dispatch**: <1ms per task
- **DAG Coordination**: <10ms total (ray.get() calls)

### Scalability

- **GPUs**: Linear scaling (8 GPUs = 8× throughput for hsaco tuning)
- **Tasks**: Handles 100-1000 hsaco variants per tuning task
- **Modules**: Multiple modules can run concurrently (separate pools)

## Error Handling

GPU tasks return structured error reports instead of raising exceptions:

```python
{
    "kernel_name": "fwd_kernel",
    "hsaco_index": 5,
    "result": "crash",  # or "NotOK" or "ERROR"
    "error": {
        "errno": 9,
        "stderr": "..."
    }
}
```

This allows:
- Partial task completion (some hsacos succeed, others fail)
- Database writes for all attempted hsacos
- Detailed error tracking per hsaco variant

## Dependencies

- `ray` - Ray framework
- `v3python.tune.exaid` - Tuning execution backend
- `v3python.database` - PostgreSQL result storage

## Advantages Over Celery

| Aspect | Celery (old) | Ray (new) |
|--------|-------------|-----------|
| **GPU Exclusivity** | Manual locking/queues | Guaranteed by Ray actors |
| **DAG Support** | Limited (chords/groups) | Native (ray.get dependencies) |
| **Worker Persistence** | Process per queue | Actor per GPU (reusable) |
| **DB Write Offload** | GPU workers block | CPU tasks (parallel I/O) |
| **Complexity** | Triple queue design | Single orchestrator |
| **Network Overhead** | Local RabbitMQ polling | In-memory (shared memory) |
| **Error Handling** | Celery retry logic | Structured error reports |

## Debugging

Enable Ray logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or adjust Ray verbosity
ray.init(logging_level='debug')
```

Monitor Ray dashboard (if enabled):
```bash
# Ray starts dashboard on http://localhost:8265
# Shows actor status, task timeline, resource usage
```

Check worker logs:
```bash
tail -f /path/to/workdir/logs/worker-gfx942-0.log
```

## Future Enhancements

1. **Task Prioritization**: High-priority tasks jump queue within GPU
2. **Adaptive Batching**: Dynamically adjust hsaco batch sizes
3. **Fault Tolerance**: Actor fault detection and restart
4. **Multi-Node Ray**: Distribute across multiple machines
5. **Profiling**: Per-task performance metrics
