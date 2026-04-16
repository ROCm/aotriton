# Tuner v3.5 Design Documentation

## Directory Structure

### Workdir Layout

```
<workdir>/
├── config.rc                 # Configuration file (sourced by scripts)
├── workers.db               # SQLite database of registered workers
├── .tune/                   # Tuning framework (copied from aotriton/.tune)
├── v3python/                # Python tuning code (copied from aotriton/v3python)
├── image.build/             # Docker image build context (generated)
├── installed/               # Architecture-specific installed libraries
│   ├── gfx90a/
│   ├── gfx942/
│   └── ...
├── build/                   # Build artifacts (EXCLUDED from sync)
├── run/                     # Runtime state (PID files, etc.) (EXCLUDED from sync)
├── scratch/                 # Temporary/local-only data (EXCLUDED from sync)
│   └── webui-commands/      # Web UI command logs
├── secrets/                 # Secrets (certificates, keys) (EXCLUDED from sync)
│   ├── ca-cert.pem
│   ├── server-cert.pem
│   └── server-key.pem
└── logs/                    # Logs (synced to workers)
    ├── worker-<arch>-<id>.log
    └── ...
```

### Sync Exclusions

**Important**: When syncing workdir to remote workers (via `sync_workdir.sh`), the following directories are **excluded**:

- `/build/` - Build artifacts, worker-local
- `/installed/` - Architecture-specific, synced separately per-worker arch
- `/run/` - Runtime state (PID files), worker-local
- `/scratch/` - Temporary/ephemeral data, not needed on workers
- `/secrets/` - Security-sensitive, deployed separately with restricted permissions

**Design Rule**: Any data that should NOT be synced to workers MUST be placed in one of these excluded directories.

### AOTriton Root Structure

```
aotriton/
├── .tune/                   # Tuning infrastructure
│   ├── bin/                 # Control scripts (run on server/dev machine)
│   │   ├── srvctl          # Server control (PostgreSQL)
│   │   ├── wkctl           # Worker control (bulk operations)
│   │   ├── deploy          # Deploy workdir to all workers
│   │   ├── prepwkdir       # Prepare workdir structure
│   │   ├── initdb          # Initialize database schema
│   │   ├── libbld          # Build libraries (dev machine only)
│   │   └── imgbld          # Build Docker images on all workers
│   ├── single/             # Single-worker operations
│   │   ├── start_worker.sh
│   │   ├── stop_worker.sh
│   │   ├── restart_worker.sh
│   │   ├── sync_workdir.sh # Sync workdir to one worker
│   │   ├── build_image.sh  # Build Docker image on one worker
│   │   └── ...
│   ├── remote/             # Scripts executed ON remote workers
│   │   ├── worker_service.sh  # SysV-style init for workers
│   │   └── rayctl          # Ray cluster control
│   ├── lib/                # Shared bash libraries
│   │   ├── config_load.sh
│   │   └── db_query.sh
│   └── webui/              # Web dashboard
│       ├── __init__.py
│       ├── routes.py
│       ├── tasks.py
│       ├── action_tracker.py
│       └── templates/
├── v3python/               # Python tuning framework
│   ├── tune/               # Tuning modules
│   │   ├── worker_main.py  # Worker process entry point
│   │   ├── exaid.py        # Execution backend
│   │   └── attn_fwd/       # Example tuning module
│   ├── pq/                 # PostgreSQL queue implementation
│   │   ├── schema.sql
│   │   ├── queue.py        # Queue operations
│   │   └── results.py      # Result storage
│   └── ray/                # Ray-based task execution
│       ├── orchestrator.py # TuningOrchestrator class
│       ├── gpu_worker.py   # GPUWorker actor
│       ├── worker_pool.py  # Worker pool management
│       └── cpu_tasks.py    # CPU-only tasks
└── third_party/
```

## Architecture Decisions

### 1. PostgreSQL-Only Queue (Tuner v3.5)

**Previous**: Celery + RabbitMQ + PostgreSQL  
**Current**: PostgreSQL only

- **Queue**: PostgreSQL `tuning_tasks` table with `status` column
- **Results**: PostgreSQL `tuning_results` table
- **Worker**: `worker_main.py` polls PostgreSQL, executes via Ray
- **Rationale**: Simpler deployment, fewer moving parts, one database for everything

### 2. Ray Framework for GPU-Exclusive Task Execution

**Architecture**: Shared Ray cluster + persistent worker pool

```
Worker Machines:
  Ray Cluster (shared, started by worker_service.sh)
    └─ GPU Worker Pool (1 worker per GPU, 1:1 mapping)
        ├─ GPU 0 Worker (exclusive)
        ├─ GPU 1 Worker (exclusive)
        └─ GPU N Worker (exclusive)

  Multiple worker_main.py instances (share same Ray cluster)
    ├─ Worker 0: fetch task → submit to Ray ActorPool
    ├─ Worker 1: fetch task → submit to Ray ActorPool
    └─ Worker N: fetch task → submit to Ray ActorPool
```

**Key Points**:
- **1:1 GPU mapping**: Each GPU has exactly ONE Ray worker (GPUWorker actor)
- **Shared cluster**: All worker_main.py instances connect to same Ray cluster (`address='auto'`)
- **ActorPool load balancing**: Preprocess/probe tasks dispatched to least-loaded GPU
- **Persistent workers**: Worker pool survives across tasks, no per-task recreation
- **exaid caching**: Each GPUWorker calls `exaid_create(module, gpu_id)` per task (cached internally by key)

### 3. Web UI Command Execution

**Pattern**: Non-blocking subprocess execution with real-time output streaming

```python
# tasks.py
run_command(cmd, cwd, workdir, description)
  → Creates ActionTracker with unique action_id
  → Spawns subprocess with stdin=subprocess.DEVNULL
  → Captures stdout/stderr in background threads
  → Logs to workdir/scratch/webui-commands/<action_id>.{stdout,stderr}
  → Returns action_id immediately

# routes.py
/api/actions/<action_id>/output
  → Streams new output lines since last poll
  → Frontend polls every 1s, appends to textarea

# Frontend (command_widget.html)
  → Each command gets collapsible panel with:
    - × (remove button)
    - 🛑 (kill button, visible only when running)
    - Real-time output textarea
    - Auto-collapses to 2 lines when exit code = 0
```

**stdin suppression**: All web-initiated commands use `stdin=subprocess.DEVNULL` to prevent blocking on interactive prompts.

### 4. Docker Image Builds

**Individual Worker**:
```bash
.tune/single/build_image.sh <workdir> <hostname> [--follow]
  → ssh <hostname> "jobid=$(tsp docker build ...) && tsp -c $jobid"
  → --follow flag: web UI gets real-time output via tsp -c
  → tsp -c blocks until completion and exits with job's exit code
```

**Bulk (All Workers)**:
```bash
.tune/bin/imgbld <workdir>
  → Iterates workers.db
  → Calls build_image.sh for each worker (without --follow)
```

**Design Note**: Build libraries (`libbld`) only runs on dev machine once, never on workers.

### 5. Security: mTLS for Web Dashboard

**Location**: `<workdir>/secrets/` (excluded from sync)

```
secrets/
├── ca-cert.pem          # Certificate Authority
├── server-cert.pem      # Server certificate
└── server-key.pem       # Server private key
```

**Flask Configuration**:
```python
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('secrets/server-cert.pem', 'secrets/server-key.pem')
ssl_context.load_verify_locations('secrets/ca-cert.pem')
ssl_context.verify_mode = ssl.CERT_REQUIRED
```

**Rationale**: Self-signed certificates for internal deployment, client cert required.

### 6. Worker Database Schema

**SQLite**: `<workdir>/workers.db`

```sql
CREATE TABLE workers (
    hostname TEXT PRIMARY KEY,
    arch TEXT NOT NULL,
    workdir_override TEXT  -- NULL means use DEFAULT_WORKDIR from config.rc
);
```

**Worker Workdir Resolution**:
```bash
WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"
```

Each worker can have custom workdir path, falls back to `DEFAULT_WORKDIR` from config.rc.

## Important Constraints

### 1. Command Arguments: Use Underscores

**Pattern**: `--worker_id` NOT `--worker-id`

**Rationale**: Consistency across Python argparse and bash scripts.

### 2. PostgreSQL Connection: No Database Name

**Correct**:
```python
conn = psycopg.connect(
    host=host,
    port=port,
    user=user,
    password=password
    # NO dbname parameter
)
```

**Rationale**: Use default database, manage different databases via Docker volumes.

### 3. Git Commits: No Destructive Hooks

**Never** use:
- `--no-verify` (skip hooks)
- `--no-gpg-sign` (skip signing)
- `git commit --amend` after hook failure (creates new commit instead)

**Rationale**: Hooks are there for a reason, bypassing them can lose work.

### 4. Docker Image Pulls: Explicit First

**Pattern**:
```bash
# Pull first
docker pull "$IMAGE"

# Then run
docker run "$IMAGE" ...
```

**Rationale**: Clearer progress feedback, fail-fast if image doesn't exist.

## Workflow Patterns

### Worker Lifecycle

```bash
# Start workers (on each worker machine)
.tune/remote/worker_service.sh <workdir> <arch> start [num_workers]
  → Starts Ray cluster (shared)
  → Spawns N worker_main.py processes
  → Each worker_main.py connects to Ray cluster and polls PostgreSQL

# Stop workers
.tune/remote/worker_service.sh <workdir> <arch> stop
  → Graceful shutdown (SIGTERM, 30s timeout)
  → Force kill if needed (SIGKILL)
  → Stops Ray cluster
```

### Task Execution Flow

```
1. User submits task → PostgreSQL (status='queued')
2. worker_main.py polls → finds task → status='running'
3. worker_main.py → TuningOrchestrator.execute_tuning_dag()
4. TuningOrchestrator:
   a. ActorPool.submit(preprocess) → least-loaded GPU
   b. ActorPool.submit(probe) → least-loaded GPU
   c. Round-robin distribute tune_hsaco tasks across all GPUs
   d. CPU tasks write results to PostgreSQL
   e. Postprocess aggregates and marks task complete
5. Task marked complete in PostgreSQL
```

### Deployment Workflow

```bash
# 1. Prepare workdir (on server/dev machine)
.tune/bin/prepwkdir <workdir>
  → Creates directory structure
  → Copies config.rc template
  → Initializes workers.db

# 2. Build libraries (on dev machine ONLY, manual SSH)
.tune/bin/libbld <workdir>
  → Builds AOTriton libraries for all architectures
  → Stores in <workdir>/installed/<arch>/

# 3. Deploy to workers
.tune/bin/deploy <workdir>
  → Syncs to all workers (excludes /build, /run, /scratch, /secrets, /installed)
  → Syncs architecture-specific /installed/<arch> per worker

# 4. Build Docker images on workers
.tune/bin/imgbld <workdir>
  → Builds worker Docker image on each worker (via tsp)

# 5. Start workers
.tune/bin/wkctl <workdir> start
  → Starts worker_service.sh on each worker via SSH
```

## Configuration Management

### config.rc Variables

```bash
# PostgreSQL
CELERY_SERVICE_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=aotriton
POSTGRES_PASSWORD=<secret>
POSTGRES_DOCKER_IMAGE=postgres:17.6
POSTGRES_DOCKER_VOLUME=<path>

# Workers
DEFAULT_WORKDIR=/opt/aotriton/workdir
NUM_GPUS=4
CONTAINER_SUFFIX=<unique-id>

# Docker Images
CELERY_WORKER_IMAGE=aotriton-worker:latest
```

**Loading Pattern**:
```bash
. "$TUNE_ROOT/lib/config_load.sh"
load_config "$WORKDIR"  # Sources <workdir>/config.rc
```

## Testing Patterns

### Manual Testing Checklist

1. **Server Start**: `srvctl <workdir> start` → PostgreSQL container running
2. **DB Init**: `initdb <workdir>` → Tables created
3. **Worker Deploy**: `deploy <workdir>` → Files synced to workers
4. **Image Build**: Individual worker build button → See real-time output
5. **Worker Start**: `wkctl <workdir> start` → Workers running, Ray cluster active
6. **Task Submit**: Submit via PostgreSQL → Task executes, results stored
7. **Web UI**: Commands tracked, kill button works, output collapses on success

### Debugging

**Check Ray cluster**:
```bash
ssh <worker> "cd <workdir> && .tune/remote/rayctl <workdir> status"
```

**Check worker status**:
```bash
ssh <worker> "cd <workdir> && .tune/remote/worker_service.sh <workdir> <arch> status"
```

**View logs**:
```bash
ssh <worker> "tail -f <workdir>/logs/worker-<arch>-<id>.log"
```

**Web UI command logs**:
```bash
cat <workdir>/scratch/webui-commands/<action_id>.stdout
```

---

**Last Updated**: 2026-04-16  
**Version**: Tuner v3.5
