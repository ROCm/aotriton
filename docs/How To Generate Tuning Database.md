# Overview

AOTriton Tuner v3.5 is a distributed tuning framework that generates optimal kernel configurations for different GPU architectures. The system consists of:

- **PostgreSQL database**: Stores task queue and tuning results
- **Local queue system**: Unix socket-based message broker for task distribution
- **GPU workers**: Execute tuning tasks on specific GPUs
- **WebUI**: Browser-based interface for monitoring and management

This replaces the previous Celery-based architecture (Tuner v3.0) with a simpler, more debuggable design.

# Prerequisites

## System Requirements

* **Dev Node**: A host that can access all GPU workers via SSH
  - Used for building, deploying, and managing the tuning infrastructure
  - Can be the same machine as the server node
* **Server Node**: A host accessible to all GPU workers
  - Runs PostgreSQL database
  - Can be the same machine as the dev node
* **GPU Workers**: One or more hosts with AMD GPUs
  - Must be accessible via SSH from the dev node
  - All workers should have the same base Docker image

**Network Requirements:**
- PostgreSQL port (default: 5432, configurable in config.rc)
- WebUI port (TBD)
- SSH access from dev node to all GPU workers

Linux is assumed for all nodes.

## Software Requirements

**All nodes:**
- `ssh` - for remote access
- `docker` - for containerized execution (podman may work but is untested)

**Dev node:**
- `sqlite3` - for worker database management
- `rsync` - for deploying workdir to GPU workers

**GPU worker nodes:**
- `task-spooler` (`tsp`) - for queuing Docker image builds (optional but recommended)

**Docker base image** (shared across all workers):
- Python >= 3.10
- git, bash
- A Python venv with PyTorch pre-installed, or torch wheel files available at `/`
  - If using wheels, both `/torch-*.whl` and `/triton-*.whl` must be available
  - **Note:** PEP-668 prohibits installing packages to system-managed Python installations
  - A venv is strongly recommended regardless of container usage

# Quick Start

```bash
# 1. Clone AOTriton
git clone --recursive https://github.com/ROCm/aotriton.git -b main -o upstream
cd aotriton

# 2. Create and configure working directory
.tune/bin/initproj ~/wkdir.tuning

# 3. Register GPU workers (interactive web-based UI)
# TODO: WebUI-based worker registration - see "Register GPU Workers" section

# 4. Build AOTriton for all architectures
.tune/bin/libbld ~/wkdir.tuning

# 5. Prepare workdir and create Docker image
.tune/bin/prepwkdir ~/wkdir.tuning

# 6. Deploy to GPU workers
.tune/bin/deploy ~/wkdir.tuning

# 7. Build Docker images on each worker
.tune/bin/imgbld ~/wkdir.tuning

# 8. Initialize database schema
.tune/bin/initdb ~/wkdir.tuning

# 9. Start PostgreSQL service
.tune/bin/srvctl ~/wkdir.tuning start

# 10. Start GPU workers
.tune/bin/wkctl ~/wkdir.tuning start

# 11. Dispatch tuning tasks via WebUI
# TODO: WebUI-based task dispatch - see "Dispatch Tuning Tasks" section

# 12. Monitor progress via WebUI
# TODO: WebUI URL and usage

# 13. Stop workers and services when done
.tune/bin/wkctl ~/wkdir.tuning stop
.tune/bin/srvctl ~/wkdir.tuning stop
```

# Detailed Steps

## Clone the AOTriton Repository

**IMPORTANT:** Scripts depend on specific git configuration.

```bash
git clone --recursive https://github.com/ROCm/aotriton.git -b main -o upstream
cd aotriton
```

**Requirements:**
- The remote pointing to `ROCm/aotriton` **must** be named `upstream`
- The `main` branch must be cloned
- If working on a branch, it must be forked from `upstream/main`

**All following commands assume the current working directory is the cloned `aotriton/` directory.**

## Create Working Directory

```bash
.tune/bin/initproj <working_directory>
```

This interactive script will:
1. Prompt for configuration values (database credentials, Docker images, etc.)
2. Create `<working_directory>/config.rc` with your settings
3. Create `<working_directory>/workers.db` (SQLite database for worker registry)

**Key configuration values:**
- `POSTGRES_USER`, `POSTGRES_PASSWORD` - Database credentials
- `POSTGRES_PORT` - Database port (default: 5432)
- `POSTGRES_DOCKER_IMAGE` - PostgreSQL image version (e.g., `postgres:17.6`)
- `POSTGRES_DOCKER_VOLUME` - Docker volume name for persistent data
- `CELERY_WORKER_IMAGE_BASE` - Base Docker image for workers
- `CELERY_WORKER_IMAGE` - Name for the built worker image
- `CELERY_WORKER_PYTHON` - Path to Python executable in worker image
- `CELERY_SERVICE_HOST` - Hostname where PostgreSQL runs

Example `config.rc`:
```bash
POSTGRES_USER=aotriton
POSTGRES_PASSWORD=securepassword123
POSTGRES_PORT=5432
POSTGRES_DOCKER_IMAGE=postgres:17.6
POSTGRES_DOCKER_VOLUME=aotriton_pgdata-tunerv3.5
CONTAINER_SUFFIX=tunerv35
CELERY_SERVICE_HOST=server.example.com
CELERY_WORKER_IMAGE_BASE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_2.9
CELERY_WORKER_IMAGE=aotriton-tunerv3.5
CELERY_WORKER_PYTHON=/venv/bin/python
```

## Register GPU Workers

**TODO: WebUI-based worker registration interface**

Currently, workers must be registered by directly editing `<working_directory>/workers.db`:

```bash
sqlite3 <working_directory>/workers.db
```

```sql
-- Set default workdir for all workers
INSERT OR REPLACE INTO config (key, value) VALUES ('default_workdir', '/path/on/workers');

-- Add GPU workers
INSERT INTO workers (hostname, arch) VALUES ('gpu-01.example.com', 'gfx942');
INSERT INTO workers (hostname, arch) VALUES ('gpu-02.example.com', 'gfx942');
INSERT INTO workers (hostname, arch) VALUES ('gpu-03.example.com', 'gfx90a');

-- Optional: Set custom workdir for specific worker
UPDATE workers SET workdir_override='/custom/path' WHERE hostname='gpu-03.example.com';

-- View registered workers
SELECT * FROM workers;
```

**Worker Database Schema:**
- `hostname` - SSH hostname of the GPU worker
- `arch` - GPU architecture (e.g., gfx942, gfx90a, gfx1100)
- `workdir_override` - Optional custom workdir path (overrides default_workdir)

## Build AOTriton for All Architectures

**This step must be run in an environment compatible with `CELERY_WORKER_IMAGE_BASE`**

```bash
.tune/bin/libbld <working_directory>
```

This script:
1. Builds a Triton wheel from `third_party/triton/` and caches it in `<working_directory>/scratch/triton/`
   - Cached wheel is reused if Triton and Python versions match
   - Automatically rebuilds if versions change
2. Queries `workers.db` and builds AOTriton for each registered architecture
   - Build artifacts: `<working_directory>/build/<arch>/` (not synced to workers)
   - Installed files: `<working_directory>/installed/<arch>/` (synced to workers)

The script is idempotent - safe to run multiple times.

## Prepare Working Directory

```bash
.tune/bin/prepwkdir <working_directory>
```

This script:
1. Clones or updates AOTriton source to `<working_directory>/aotriton.src`
   - Performs shallow clone from `upstream/main`
   - If already cloned, performs `git pull` to update
2. Copies `.tune/image.scripts/` to `<working_directory>/image.scripts/`
3. Creates `<working_directory>/image.build/Dockerfile` from `config.rc`

**Generated Dockerfile:**
- Starts from `${CELERY_WORKER_IMAGE_BASE}`
- Creates Python venv at `${CELERY_WORKER_PYTHON}` if it doesn't exist
- Installs `requirements-tuning.txt`
- Runs all scripts matching `[0-9][0-9]-*.sh` in `image.scripts/`

### Customizing the Worker Image

Add custom scripts to `<working_directory>/image.scripts/` with numeric prefixes:

Example: `<working_directory>/image.scripts/90-install_torch.sh`
```bash
#!/bin/bash
/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

Scripts are executed in lexicographic order during Docker build.

## Deploy Working Directory to GPU Workers

```bash
.tune/bin/deploy <working_directory>
```

This script uses rsync to deploy the workdir to each registered GPU worker:
- Syncs all files except: `build/`, `scratch/`, `run/`, `secrets/`, `installed/`, `aotriton.src/`
- Architecture-specific files (`installed/<arch>/` and `aotriton.src/`) are synced separately with `--delete` to ensure exact copies
- Uses `--exclude '*.pyc'` to avoid issues with root-owned bytecode files
- Respects each worker's configured workdir (default or custom override)

**Note:** The script minimizes rsync calls to reduce SSH authentication overhead.

## Build Worker Docker Images

```bash
.tune/bin/imgbld <working_directory>
```

This script builds the Docker image on each GPU worker in parallel:
- SSHs to each worker
- Queues `docker build` command via `tsp` (task-spooler)
- Builds from `<worker_workdir>/image.build/Dockerfile`
- Tags image as `${CELERY_WORKER_IMAGE}`

To follow build progress on a specific worker:
```bash
.tune/single/build_image.sh <working_directory> <hostname> --follow
```

## Initialize Database Schema

**Ensure PostgreSQL service is running first** (see next section)

```bash
.tune/bin/initdb <working_directory>
```

This script:
1. Connects to PostgreSQL using credentials from `config.rc`
2. Executes `v3python/tune/pq/schema.sql` to create tables
3. Creates partitioned tables for each architecture in `workers.db`
   - Example: `task_queue_gfx942`, `task_queue_gfx90a`

To recreate the schema (drops all existing data):
```bash
.tune/bin/initdb <working_directory> --recreate
```

**Tables created:**
- `task_queue` - Main task queue (partitioned by architecture)
- `task_queue_<arch>` - Per-architecture partitions
- `tuning_results` - Stores kernel tuning results
- Helper functions for partition management

## Start PostgreSQL Service

```bash
.tune/bin/srvctl <working_directory> start
```

This starts a PostgreSQL container on the server node:
- Uses `--network=host` (no port mapping needed)
- Listens on port configured in `config.rc` (`POSTGRES_PORT`)
- Data persisted in Docker volume `${POSTGRES_DOCKER_VOLUME}`
- Container name: `aotriton_pgsql.${CONTAINER_SUFFIX}`

**Other commands:**
```bash
.tune/bin/srvctl <working_directory> stop     # Stop PostgreSQL
.tune/bin/srvctl <working_directory> restart  # Restart PostgreSQL
.tune/bin/srvctl <working_directory> status   # Check status
```

**Connecting to PostgreSQL:**
```bash
.tune/bin/psql <working_directory>  # Interactive psql shell
```

## Start GPU Workers

```bash
.tune/bin/wkctl <working_directory> start
```

This script:
- SSHs to each registered GPU worker
- Starts a Docker container with the worker image
- Mounts `<worker_workdir>` to `/wkdir` inside container
- Starts local queue system (broker + GPU workers + PG readers + CPU workers)
- Records container ID in `<worker_workdir>/run/worker.containerid`

**Worker components started (per node):**
- 1 local broker (message router via Unix socket)
- N GPU workers (one per GPU, detected automatically)
- 4 PostgreSQL reader workers (fetch tasks from database)
- 4 CPU workers (postprocess and write results)

**Start specific workers only:**
```bash
.tune/bin/wkctl --host gpu-01.example.com --host gpu-02.example.com <working_directory> start
```

**Stop all workers:**
```bash
.tune/bin/wkctl <working_directory> stop
```

**Restart workers:**
```bash
.tune/bin/wkctl <working_directory> restart
```

**Check worker status:**
```bash
# SSH to a worker and check processes
ssh gpu-01.example.com "docker exec \$(cat <worker_workdir>/run/worker.containerid) bash .tune/remote/worker_service.sh status /wkdir gfx942"
```

## Dispatch Tuning Tasks

**TODO: WebUI-based task dispatch interface**

Currently, tasks can be dispatched programmatically:

```bash
.tune/bin/dispatch <working_directory> <module> [options]
```

**Available modules:**
- `flash` - Flash attention kernels

**Common options:**
- `--arch ARCH [ARCH ...]` - Target architecture(s)
- Module-specific parameter filters (see module help)

Example:
```bash
# Dispatch all flash tasks for gfx942
.tune/bin/dispatch ~/wkdir.tuning flash --arch gfx942

# Dispatch only float16 tasks with specific sequence lengths
.tune/bin/dispatch ~/wkdir.tuning flash --arch gfx942 \
  --dtype float16 --seqlen_q 128 256 --seqlen_k 128 256
```

**How it works:**
1. Script queries the module for all parameter combinations
2. Creates task entries in PostgreSQL `task_queue_<arch>` partitions
3. PostgreSQL reader workers fetch tasks and send to local broker
4. Broker distributes tasks to GPU workers
5. Results written back to `tuning_results` table

## Monitor Tuning Progress

**TODO: WebUI monitoring interface**

Currently, monitor via database queries:

```bash
# Connect to database
.tune/bin/psql ~/wkdir.tuning
```

```sql
-- Queue statistics for all architectures
SELECT arch, 
       COUNT(*) FILTER (WHERE status = 'pending') as pending,
       COUNT(*) FILTER (WHERE status = 'running') as running,
       COUNT(*) FILTER (WHERE status = 'completed') as completed,
       COUNT(*) FILTER (WHERE status = 'failed') as failed
FROM task_queue
GROUP BY arch;

-- Recent completed tasks
SELECT id, arch, module, completed_at, error
FROM task_queue
WHERE status = 'completed'
ORDER BY completed_at DESC
LIMIT 10;

-- Failed tasks
SELECT id, arch, module, error
FROM task_queue
WHERE status = 'failed'
ORDER BY id DESC
LIMIT 10;

-- Tuning results count
SELECT COUNT(*) FROM tuning_results;
```

**Check worker logs:**
```bash
# On GPU worker
tail -f <worker_workdir>/run/logs/gpu-worker-0.log
tail -f <worker_workdir>/run/logs/pg-reader-gfx942-0.log
tail -f <worker_workdir>/run/logs/cpu-worker-0.log
```

## Export Tuning Results

**TODO: Export functionality**

Results are stored in the `tuning_results` table and can be exported for integration into AOTriton.

# Troubleshooting

## Workers Not Fetching Tasks

**Check:**
1. PostgreSQL is running: `.tune/bin/srvctl <workdir> status`
2. Workers are running: `ssh <worker> docker ps`
3. Database connection works: `.tune/bin/psql <workdir>`
4. Tasks exist in queue: `SELECT COUNT(*) FROM task_queue WHERE status='pending';`

**Common issues:**
- Incorrect `POSTGRES_PORT` in config.rc
- Firewall blocking PostgreSQL port
- Wrong credentials in config.rc

## Tasks Stuck in "Running" State

**Possible causes:**
- Worker crashed without marking task as failed
- Exaid subprocess hung (check GPU worker logs)
- Database connection lost during task execution

**Solutions:**
- Reset stale tasks: `SELECT reset_stale_tasks(7200);` (2 hour timeout)
- Restart workers: `.tune/bin/wkctl <workdir> restart`

## Permission Errors During Rsync

**Error:** `rsync: failed to set permissions on "...": Operation not permitted`

**Cause:** Root-owned `.pyc` files created inside Docker containers

**Solution:** Already handled by `--exclude '*.pyc'` in sync_workdir.sh. If issue persists:
```bash
# On remote worker, delete root-owned files
ssh <worker> "docker run --rm -v <worker_workdir>:/wkdir -w /wkdir <worker_image> find aotriton.src -name '*.pyc' -delete"
```

## Docker Build Fails

**Check:**
1. Base image exists: `docker pull ${CELERY_WORKER_IMAGE_BASE}`
2. Dockerfile syntax: `cat <workdir>/image.build/Dockerfile`
3. Build logs: `ssh <worker> tsp -c <job_id>`

## Database Authentication Fails

**Error:** `FATAL: password authentication failed for user "aotriton"`

**Causes:**
1. PostgreSQL not running on configured port
2. Wrong credentials in config.rc
3. PostgreSQL container using old volume with different credentials

**Solution:**
```bash
# Check what port PostgreSQL is listening on
docker port aotriton_pgsql.<suffix>

# If port is wrong, delete old volume and restart
.tune/bin/srvctl <workdir> stop
docker volume rm <POSTGRES_DOCKER_VOLUME>
.tune/bin/srvctl <workdir> start
.tune/bin/initdb <workdir> --recreate
```

# Architecture Notes

## Tuner v3.5 vs Tuner v3.0 (Celery)

| Aspect | Tuner v3.0 (Celery) | Tuner v3.5 (Local Queue) |
|--------|---------------------|--------------------------|
| Message Broker | RabbitMQ | Unix sockets (per-node local broker) |
| Task Distribution | Celery workers pull from RabbitMQ | PostgreSQL readers push to local broker |
| Worker Framework | Celery | Custom GenericWorker |
| GPU Isolation | Ray ActorPool | Direct subprocess (exaid) |
| Database | PostgreSQL (results only) | PostgreSQL (queue + results) |
| Debugging | Requires Ray/Celery tools | Standard logs, no special tools |
| Dependencies | Celery, RabbitMQ, Ray | psycopg, PostgreSQL only |

## Local Queue Architecture

Each GPU worker node runs:

```
PostgreSQL (server node)
    ↓ (4 PG reader workers fetch tasks)
Local Broker (Unix socket /wkdir/run/broker.sock)
    ↓ (distributes to queues)
├─ gpu_queue → N GPU workers (one per GPU)
├─ cpu_queue → 4 CPU workers (postprocess)
└─ result_queue → (results sent back to PG readers)
```

**Message flow:**
1. PG reader fetches task from `task_queue_<arch>`, marks as "running"
2. PG reader sends `tune_kernel` message to broker
3. Broker forwards to `gpu_queue`
4. GPU worker picks up task, runs preprocess → probe → tune_hsaco (parallel)
5. GPU worker sends `hsaco_result` messages back to broker
6. Broker forwards to `cpu_queue`
7. CPU worker aggregates results, writes to `tuning_results` table
8. PG reader marks task as "completed" in `task_queue_<arch>`

# SLURM Support

**TODO: SLURM integration for Tuner v3.5**

For HPC environments, SLURM support will allow:
- Job submission via `sbatch` instead of Docker containers
- Shared filesystem deployment instead of per-node rsync
- GRES-based GPU allocation
- Bad node tracking and automatic exclusion

Planned workflow:
```bash
# Setup
.tune/bin/initproj <workdir>  # Enable SLURM when prompted
# TODO: Register SLURM GRES configurations via WebUI
.tune/bin/libbld <workdir>
.tune/bin/prepwkdir <workdir>
# TODO: Build SLURM venv script
.tune/bin/deploy <workdir>  # Rsync to SLURM shared filesystem

# Start services and submit jobs
.tune/bin/srvctl <workdir> start
.tune/bin/srun <workdir>  # Submit SLURM batch jobs

# Dispatch tasks (same as Docker workflow)
# TODO: WebUI task dispatch

# Monitor and manage
ssh <SLURM_LOGIN_NODE> squeue -u $USER
ssh <SLURM_LOGIN_NODE> scancel <job_id>
```

Key differences from Docker workflow:
- Workers run as SLURM jobs instead of Docker containers
- Python venv on shared NFS instead of Docker image per node
- Time limits managed by SLURM allocation
- Bad nodes tracked in database and excluded via `--exclude`

**Status:** Design complete, implementation pending
