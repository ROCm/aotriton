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
- SSH access from dev node to all GPU workers

Linux is assumed for all nodes.

## Software Requirements

**All nodes:**
- `ssh` - for remote access
- `docker` - for containerized execution (podman may work but is untested)
- `task-spooler` (`tsp`) - for queuing Docker image builds (optional but recommended)

**Dev node:**
- `sqlite3` - for worker database management
- `rsync` - for deploying workdir to GPU workers

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

# 3. Generate mTLS certificates and start WebUI
.tune/webui/gencerts/generate_all_certs.sh ~/wkdir.tuning   # first time only
~/venv/bin/python .tune/dashboard.py ~/wkdir.tuning
#    Open https://<CELERY_SERVICE_HOST>:8888 — import client.p12 into browser first

# 4. Register GPU workers
#    WebUI → Workers tab → + Add Worker

# 5. Prepare workdir — REQUIRED before every build when dev repo has changed
#    (copies aotriton.src from upstream/main into <workdir>/aotriton.src)
#    WebUI → Builds tab → Deployment → Prepare Workdir
#    (or: .tune/bin/prepwkdir ~/wkdir.tuning)

# 6. Deploy to remote build node (if using one; required before building there)
#    WebUI → Builds tab → Deployment → Deploy to Remote Build Node

# 7. Build tuning version of AOTriton libraries
#    WebUI → Builds tab → Build Tuning Version → Build Libraries (All)
#    (or: .tune/bin/libbld ~/wkdir.tuning)

# 8. Fetch tuning libraries from remote build node (if applicable)
#    WebUI → Builds tab → Build Tuning Version → Fetch from Remote Build Node

# 9. Deploy to GPU workers
#    WebUI → Workers tab → Bulk Actions → Deploy to All Workers
#    (or: .tune/bin/deploy ~/wkdir.tuning)

# 10. Build Docker images on each worker
#     WebUI → Workers tab → Bulk Actions → Build Images on All Workers

# 11. Start PostgreSQL and initialize database schema
#     WebUI → Servers tab → PostgreSQL → Start Server
#     WebUI → Servers tab → Database Schema → Initialize Schema
#     (or: .tune/bin/srvctl ~/wkdir.tuning start && .tune/bin/initdb ~/wkdir.tuning)

# 12. Start GPU workers (select GPUs first)
#     WebUI → Workers tab → per-host → Start
#     (or: .tune/bin/wkctl ~/wkdir.tuning start)

# 13. Dispatch tuning tasks
#     .tune/bin/dispatch ~/wkdir.tuning flash --arch gfx942

# 14. Monitor progress
#     WebUI → Dashboard (auto-refreshes every 10s)

# 15. Export results after tuning completes
#     WebUI → Servers tab → Bake LUT

# --- Correctness Testing (after Bake LUT) ---

# 16. Prepare workdir (required if dev repo changed since last prepare)
#     WebUI → Builds tab → Deployment → Prepare Workdir

# 17. Deploy to remote build node (push updated aotriton.src and baked LUT at installed/database/ before building)
#     WebUI → Builds tab → Deployment → Deploy to Remote Build Node

# 18. Build testing libraries
#     WebUI → Builds tab → Build Testing Version → Build Test Libraries (All)

# 19. Fetch testing libraries from remote build node (if applicable)
#     WebUI → Builds tab → Build Testing Version → Fetch from Remote Build Node

# 20. Deploy testing libraries to workers
#     WebUI → Workers tab → Bulk Actions → Deploy to All Workers

# 21. Run tests
#     WebUI → Testing tab → Run Test / Run Partial Test

# 22. Stop workers and services when done
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

## Working Directory Structure

After setup, the working directory contains:

```
<workdir>/
├── config.rc                   # Tuning infrastructure configuration (bash syntax)
├── workers.db                  # SQLite registry of worker hosts and GPU metadata
├── secrets/                    # mTLS certificates (CA, server, client.p12)
│   ├── ca.crt / ca.key
│   ├── server.crt / server.key
│   ├── client.crt / client.key
│   └── client.p12              # Import into browser for WebUI access
├── aotriton.src/               # Snapshot of the AOTriton source (from upstream/main)
│                               # Updated by Prepare Workdir; NOT auto-synced with dev repo
├── installed/
│   ├── <arch>/                 # Tuning-instrumented libraries for arch (e.g. gfx942/)
│   ├── test/<arch>/            # Testing-instrumented libraries for arch
│   ├── database/               # Baked LUT: sharded kernel selection database
│   │   └── amd/<arch>/...      # Per-arch SQLite shards produced by Decompose DB
│   └── adiffs/<arch>.txt       # Downloaded accuracy-diff file (from Download adiffs)
├── image.build/                # Generated Dockerfile and build scripts
│   └── Dockerfile
├── image.scripts/              # Helper scripts copied into the Docker image
├── scratch/
│   ├── triton/                 # Cached Triton wheel build
│   └── centraldb.sqlite3       # Intermediate unified SQLite before sharding
├── build/                      # Local CMake build artifacts (not synced to workers)
├── run/                        # Runtime state (not synced to workers)
│   ├── broker.sock             # Unix socket for local broker (on each worker node)
│   ├── logs/                   # Worker process logs
│   └── tests/                  # Test output directory
│       └── partial/            # Partial test run outputs (sel<N>.txt, adiffs.txt)
└── <per-pass test output>/
    ├── ut_pass<N>.out          # Full pytest log for pass N
    └── sel<N>.txt              # Failing test IDs for pass N (drives partial re-run)
```

**Key facts:**
- `aotriton.src/` is a **separate copy** that does not auto-sync with the dev repo — re-run Prepare Workdir whenever you make source changes.
- `build/`, `scratch/`, `run/`, and `secrets/` are **never rsynced** to worker nodes.
- `installed/database/` is synced only to the **remote build node** (so the test build can embed the LUT). It is **not** synced to GPU workers — the deploy script sends only `installed/<arch>/` to each worker.

## Start the WebUI

The WebUI is launched from the cloned AOTriton repository on the dev node. All
commands below assume the current directory is the repo root.

**Prerequisites:** install `cheroot` into the dev environment:

```bash
pip install cheroot
```

### Generate mTLS Certificates (first time)

The WebUI uses mutual TLS — both the server and the browser must present
certificates signed by the same CA. Certificates are stored in
`<workdir>/secrets/` and are tied to the hostname in `CELERY_SERVICE_HOST`
from `config.rc`.

```bash
.tune/webui/gencerts/generate_all_certs.sh <working_directory>
```

The script is interactive:
- If you already have certificates from another workdir, enter that path to
  copy them (avoids re-distributing `client.p12` to users).
- Otherwise, confirm generation to create a new CA, server certificate, and
  client certificate bundle.

Generated files in `<workdir>/secrets/`:

| File | Purpose |
|------|---------|
| `ca.crt` / `ca.key` | Certificate Authority (keep `ca.key` private) |
| `server.crt` / `server.key` | Server certificate (loaded by dashboard.py) |
| `client.crt` / `client.key` | Client certificate |
| `client.p12` | PKCS12 bundle — import this into every browser that needs access |

**Browser import (one time per browser):**
- **Chrome**: Settings → Privacy & Security → Manage certificates → Import → select `client.p12`
- **Firefox**: Preferences → Privacy & Security → Certificates → View Certificates → Your Certificates → Import

### Launch the WebUI

```bash
~/venv/bin/python .tune/dashboard.py <working_directory> [--port 8888]
```

The server starts on `https://0.0.0.0:8888` by default. Open
`https://<CELERY_SERVICE_HOST>:8888` in the browser; select the imported
`admin` certificate when prompted.

### Guest Mode (read-only, no auth)

For sharing a read-only progress view without certificate distribution:

```bash
~/venv/bin/python .tune/dashboard.py <working_directory> --guest [--port 9999]
```

Guest mode serves plain HTTP on port 9999 with no client certificate required.
It exposes only the Dashboard (tuning progress) and is safe to share on an
internal network.

## Register GPU Workers

Open the WebUI in a browser (`https://<dev-node>:<port>`). Navigate to the
**Workers** tab.

1. In the **Basic Configuration** section, set the **Default Workdir** (path on
   remote workers, e.g. `/home/user/wkdir`) and click **Update Default Workdir**.
2. In the worker table at the bottom, fill in hostname and GPU arch (e.g.
   `gfx942`) and click **+ Add Worker**. Optionally set a per-host workdir
   override via the **Update** button in the worker row.
3. Click **Detect GPU** on each worker row to populate GPU metadata automatically,
   then click the refresh icon next to it to display the result.
4. Check the **Tuner** role checkbox for each worker that will run tuning tasks.

The worker list is stored in `<working_directory>/workers.db` (SQLite). You can
also edit it directly:

```bash
sqlite3 <working_directory>/workers.db
INSERT INTO workers (hostname, arch) VALUES ('gpu-01.example.com', 'gfx942');
INSERT OR REPLACE INTO config (key, value) VALUES ('default_workdir', '/home/user/wkdir');
```

## Prepare Working Directory

In the **Builds** or **Workers** tab, under **Deployment**, click **Prepare Workdir**.

This clones (or updates) `<workdir>/aotriton.src` from `upstream/main`,
copies `image.scripts/`, and generates `image.build/Dockerfile`.

CLI equivalent:

```bash
.tune/bin/prepwkdir <working_directory>
```

> **Important:** The workdir maintains its own copy of the AOTriton source at
> `<workdir>/aotriton.src`. This copy is **not** automatically kept in sync
> with the dev repo you cloned. You must re-run **Prepare Workdir** every time
> you make changes to the dev repo that should be reflected in the build or on
> the worker nodes — otherwise those changes will be silently ignored.

## Deploy Working Directory to Remote Build Node

If you are using a remote build node (configured under **Remote Build Node** in
the Builds tab), deploy the workdir to it before building:

In the **Builds** tab, under **Deployment**, click
**Deploy to Remote Build Node (hostname)**.

This is a hard requirement before building on the remote node — the build node
must have the latest `<workdir>/aotriton.src` (synced by Prepare above) to
compile from.

Skip this step if building directly on the dev node.

## Build Tuning Version of AOTriton Libraries

In the WebUI **Builds** tab, under **Build Tuning Version of AOTriton Libraries**,
click **Build Libraries (All)** to build for all registered architectures, or
click an individual arch button to build only that arch.

This builds a Triton wheel (cached in `<workdir>/scratch/triton/`) and the
tuning-instrumented AOTriton libraries used by GPU workers during kernel search.
Build artifacts land in `<workdir>/installed/<arch>/`.

CLI equivalent:

```bash
.tune/bin/libbld <working_directory>
```

## Fetch Tuning Libraries from Remote Build Node

If the build ran on a remote build node, click **Fetch from Remote Build Node**
in the **Build Tuning Version of AOTriton Libraries** section to download the
built artifacts back to the local workdir.

Skip this step if the build ran directly on the dev node.

## Deploy Working Directory to GPU Workers

In the **Workers** tab, under **Bulk Actions**, click **Deploy to All Workers**.
For a single host, click the per-row **Deploy** button.

This rsyncs all workdir files to each registered worker, skipping `build/`,
`scratch/`, `run/`, `secrets/`. Architecture-specific files (`installed/<arch>/`
and `aotriton.src/`) are synced with `--delete`.

CLI equivalent:

```bash
.tune/bin/deploy <working_directory>
```

## Build Worker Docker Images

In the **Workers** tab, under **Bulk Actions**, click **Build Images on All Workers**.
For a single host, click the per-row **Build Image** button.

Build progress streams live in the Command Output panel. The image is tagged as
`${CELERY_WORKER_IMAGE}` on the remote host.

To follow a build from the CLI:

```bash
.tune/single/build_image.sh <working_directory> <hostname> --follow
```

## Initialize Database Schema

In the WebUI **Servers** tab, under **PostgreSQL**, click **Start Server** first.
Then under **Database Schema**, click **Initialize Schema**.

This runs `v3python/tune/pq/schema.sql` against the database and creates
per-arch partitions (`task_queue_gfx942`, etc.) for each worker in `workers.db`.
Re-run whenever new GPU architectures are added.

To drop all data and recreate from scratch, click **Recreate Schema (Danger)**.

CLI equivalents:

```bash
.tune/bin/srvctl <working_directory> start
.tune/bin/initdb <working_directory>
.tune/bin/initdb <working_directory> --recreate  # destructive
```

**Connecting to PostgreSQL directly:**

```bash
.tune/bin/psql <working_directory>       # interactive shell
.tune/bin/psql <working_directory> -c "SELECT COUNT(*) FROM task_queue;"
```

## Start GPU Workers

In the **Workers** tab, select which GPUs to use (checkboxes in each worker row),
then click **Start** on the host. Use **Stop & Start** to restart with a new GPU
selection. The status widget shows container ID and GPU process counts.

**Bulk actions**: **Start All**, **Stop All**, **Restart All**, and
**Stop All & Start All** are available in the **Bulk Actions** section.

Each node starts:
- 1 local broker (Unix socket `/wkdir/run/broker.sock`)
- N GPU workers (one per selected GPU)
- 4 PostgreSQL reader workers
- 4 CPU workers (postprocess)

**Office hours scheduling**: in the **Office Hours Scheduling** section, set a
default start/stop schedule and click **Save Default Schedule**, then
**Arm Scheduled Workers** to activate timers. Per-host overrides are set in the
worker row's schedule controls.

CLI equivalents:

```bash
.tune/bin/wkctl <working_directory> start
.tune/bin/wkctl <working_directory> stop
.tune/bin/wkctl <working_directory> restart
# Target specific hosts:
.tune/bin/wkctl --host gpu-01.example.com <working_directory> start
```

Check worker logs on the remote host:

```bash
tail -f <worker_workdir>/run/logs/gpu-worker-0.log
tail -f <worker_workdir>/run/logs/pg-reader-gfx942-0.log
tail -f <worker_workdir>/run/logs/cpu-worker-0.log
```

## Dispatch Tuning Tasks

Use the CLI to dispatch tasks (WebUI dispatch is not yet available):

```bash
# All flash tasks for gfx942
.tune/bin/dispatch ~/wkdir.tuning flash --arch gfx942

# Float16 only, specific sequence lengths
.tune/bin/dispatch ~/wkdir.tuning flash --arch gfx942 \
  --dtype float16 --seqlen_q 128 256 --seqlen_k 128 256
```

Tasks are inserted into `task_queue_<arch>`. PostgreSQL readers pick them up,
the broker distributes them to GPU workers, and results are written to
`tuning_results`.

## Monitor Tuning Progress

The WebUI **Dashboard** shows a per-arch table (pending / running / completed /
failed / cancelled / total / ETA) that auto-refreshes every 30 seconds.

For raw queries:

```bash
.tune/bin/psql ~/wkdir.tuning -c "SELECT * FROM queue_progress;"
.tune/bin/psql ~/wkdir.tuning -c "SELECT * FROM completion_eta;"
```

## Export Tuning Results

After tuning completes, run the export pipeline from the **Servers** tab under
**Bake LUT**. The steps can be run individually or as a combined pipeline:

- **Bake LUT** — runs Rebuild Accuracy Table → Compute Best Results → Export
  Best Results → Sancheck → Decompose DB in one shot.
- **Bake LUT (Incremental)** — same pipeline but passes `--incremental` to
  Compute Best Results (skips already-computed tasks).

Individual steps (for debugging or partial re-runs):

| Button | What it does |
|--------|-------------|
| **Rebuild Accuracy Table** | Recreates the most-accurate materialized view from scratch |
| **Update Accuracy Table** | Incremental MV update using `scratch/retry_task_ids.txt` |
| **Compute Best Results** | Selects fastest `hsaco_index` meeting 3× accuracy threshold |
| **Export Best Results** | Writes `scratch/centraldb.sqlite3` from `best_tuning_results` |
| **Sancheck** | Verifies LUT integrity against `scratch/centraldb.sqlite3` |
| **Decompose DB** | Shards `scratch/centraldb.sqlite3` into `installed/database/amd/<arch>/...` |

CLI equivalents:

```bash
.tune/bin/compute_best_results <workdir> [--incremental] [--fix <pass>]
.tune/bin/bake_lut <workdir>
```

# Running Correctness Tests

After Bake LUT, run correctness tests to verify the tuning database against
reference implementations. The complete testing pipeline is:

**Bake LUT → Prepare (optional) → Deploy → Build Testing Libraries →
Fetch from Remote Build Node → Deploy to Workers → Run Tests**

## Prepare Working Directory (Optional)

If the AOTriton source has changed since the last prepare, run **Prepare Workdir**
again from the **Builds** or **Workers** tab → **Deployment** section.

This updates `<workdir>/aotriton.src` from `upstream/main`. The updated source
will be pushed to the remote build node in the next step.

## Deploy Working Directory to Remote Build Node

In the **Builds** tab, under **Deployment**, click
**Deploy to Remote Build Node (hostname)**.

This pushes the updated `<workdir>/aotriton.src` (from Prepare above) and the
baked LUT (`installed/database/`) to the remote build node before building the
test libraries. Both are required: the source for compilation, and the LUT so
the test build embeds the correct kernel selection database.

## Build Testing Version of AOTriton Libraries

In the **Builds** tab, under **Build Testing Version of AOTriton Libraries**,
click **Build Test Libraries (All)** or an individual arch button.

This builds the test-instrumented libraries (`installed/test/<arch>/`) used by
the Testing tab to run pytest correctness checks.

CLI equivalent:

```bash
.tune/bin/testbld <working_directory>
```

## Fetch Testing Libraries from Remote Build Node

If the test build ran on a remote build node, click **Fetch from Remote Build
Node** in the **Build Testing Version of AOTriton Libraries** section to pull
the built `installed/test/<arch>/` artifacts to the local workdir.

Skip this step if the build ran directly on the dev node.

## Deploy Testing Libraries to Workers

Click **Deploy to All Workers** again to push the newly built test libraries
(`installed/test/<arch>/`) to the worker nodes.

## Run Tests

Navigate to the **Testing** tab. Each registered Tester host shows controls for
selecting pass number, test level, and backend.

**Signature**: displays the SHA of the current test build on that host — use
the **⟳ Refresh** button to verify the deployed libraries match expectations.

### Pass Number

An arbitrary integer label for the run. Output files are named by pass
(`ut_pass<N>.out`, `sel<N>.txt`, etc.) so multiple passes can coexist in the
same output directory without overwriting each other.

### Test Level (`FOR_RELEASE`)

Controls which test suites are collected, mapped to the `FOR_RELEASE` env var:

| Level | Tests included (`test_backward.py`) |
|-------|-------------------------------------|
| 0 | `test_fast` — quick smoke test only |
| 1 | + `test_regular_bwd`, `test_op_bwd_with_matrix_bias`, `test_gqa` |
| 2 | + `test_irregulars` (prime/irregular sequence lengths) |
| 3 | + `test_hdim_qk_ne_vo` (asymmetric head dimensions) |

`test_varlen.py` is always run alongside `test_backward.py`; its `FOR_RELEASE`
is treated as a boolean (level ≥ 1 enables full varlen coverage).

**Release criterion**: a flash attention tuning database is considered releasable
when it passes **test level 2, split backend** with only a reasonable number of
failures. All failures must be documented in `test/adiffs/<arch>.txt` (see
below). The CI script `run-ci-test.sh` sets `USE_ADIFFS_TXT` to that file,
causing known-failing tests to be skipped or marked `xfail` instead of
`FAILED`.

### Backend

| Value | What it tests |
|-------|--------------|
| `split` | Split-kernel backward (primary release target, `BWD_IMPL=0`) |
| `fused` | Fused backward kernel (`BWD_IMPL=1`) |
| `aiter` | AITER ASM backend (`BWD_IMPL=2`; bias/GQA tests skipped) |
| `v3` | V3 API output correctness (`V3_API=1`) |

### Full Test Run

Set **Pass**, **Level**, and **Backend**, then click **Run Test**. Output is
streamed via **Tail Output**. After the run, click **Show Failures** to open a
window listing all `FAILED` test lines. The run also writes `sel<pass>.txt`
(and `sel<pass>.varlen.txt`) for use by partial runs.

### Partial Test Run

After a full run, click **Run Partial Test** to re-run only the tests that
failed in the previous pass. This uses `sel<pass>.txt` to drive
`--select-from-file` in pytest.

- **Show Partial Failures**: open a failures window filtered to the partial run.
- **Tail Partial Output**: stream the partial run's output log.

### Record and Document Accuracy Diffs

Click **Record adiffs** to run a partial test with `RECORD_ADIFFS_TO` set,
capturing per-test accuracy diff values to
`<remote_workdir>/run/tests/partial/adiffs.txt`.

Click **Download adiffs** to SSH-fetch that file and save it locally to
`<local_workdir>/installed/adiffs/<arch>.txt`, served as a browser download.

To generate an adiffs entry for OOM failures (which produce no accuracy data):

```bash
.tune/bin/amend_sel_to_adiffs.sh sel1.txt [--error_reason OOM] >> adiffs.txt
```

Once the adiffs file is complete, copy it into the repository at
`test/adiffs/<arch>.txt` and commit it. The CI script `.ci/run-ci-test.sh`
sets `USE_ADIFFS_TXT=$(realpath test/adiffs/${native_arch}.txt)` before calling
`run-test.sh`, so documented failures are automatically skipped (`OOM` → `pytest.skip`,
`NAN` → `pytest.xfail`) in gated CI runs.

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
