# Overview

The tuning tool `test/tune_flash.py` was deprecated in favor of a distributed
tuning framework based on [celery](https://github.com/celery/celery).

# Prerequisites

## System

* A group of GPU workers
* A host that are accessible to all GPU workers
  - TBD: Required ports for potential firewall configurations.
  - This host will be referred as the "Server" in the following text.
* A host that can access all GPU workers with `ssh`
  - This host will be referred as the "Dev Node" in the following text.

For simplicity, instructions in the following text and scripts created under
.celery all assume dev node and server are the same host. If they are different
ones, please perform instructions below and `rsync` accordingly.

Linux is assumed for all nodes.

## Software

* For all nodes
  + ssh
  + docker
    - `podman` may work, but untested.
* Dev node software:
  + `sqlite3`
* A common docker image serve as the runtime environment, which should contain
  + python >= 3.10
  + git
  + bash
  + A venv with torch pre-installed, or its wheel file `torch-*.whl` available at `/`
    - For the latter case, although the tuning process does not need Triton,
      but `/triton-*.whl` should be available as well since it's  a dependency
    - Please be advised PEP-668 prohibits installing pip packages to system
      managed sites (e.g., `/usr/lib/python3.11/`). A venv is recommended
      regardless the usage of container.

# Steps

## Clone the AOTriton repository **IMPORTANT**

**Certain scripts depending on the configuration.**

```
git clone --recursive https://github.com/ROCm/aotriton.git -b main -o upstream
```

The remote point to `ROCm/aotriton` remote must be named as `upstream`.
The `main` branch should be cloned as well.

*If you're working with a branch, the branch must be forked from `upstream/main`.*

**The following sections assume current working directory has been changed to
the newly cloned aotriton/ directory.**

## Create a working directory on the Dev Node

`bash .celery/create-project-directory.sh <working directory>`

This is an interactive process. Just follow the prompt.

## Add GPU worker to working directory configuration

First, set the default working directory that will be used on GPU workers:

```bash
.celery/manage-workers.py <working directory> set-default-workdir <path on GPU workers>
```

Then add GPU workers by specifying the architecture and one or more hostnames:

```bash
.celery/manage-workers.py <working directory> add <arch> <hostname1> [<hostname2> ...]
```

For example, to add multiple gfx90a workers:

```bash
.celery/manage-workers.py <working directory> add gfx90a gpu-01.example.com gpu-02.example.com
```

To list all registered workers:

```bash
.celery/manage-workers.py <working directory> list
```

For more options and supported architectures, run:

```bash
.celery/manage-workers.py --help
```

## Install packages needed by host OS

```bash
.celery/install-hostos-packages.sh <working directory>
```


## Build AOTriton for all Target Architectures

**This step must be done within environment that is compatible with the CELERY_WORKER_IMAGE_BASE**

*Can be done in any node (including GPU Workers), but ideally on dev node
since this is the only node that are guaranteed having access all worker nodes,
per prerequisites.*

```bash
.celery/build-for-tuning.sh <working directory>
```

This script will:

1. Build a Triton wheel from `third_party/triton/` and store it in `<working directory>/scratch/triton/`
   - The wheel is cached and reused on subsequent runs if the Triton version and Python version match
   - If either version changes, the wheel is rebuilt automatically

2. Query the worker database and build a tuning version of AOTriton for each registered architecture
   - Builds are stored in `<working directory>/build/<arch>/` (not synced to workers)
   - Installed files are stored in `<working directory>/installed/<arch>/` (synced to workers)
   - Each architecture uses the same Triton wheel from step 1

The script is idempotent - it will skip rebuilding the Triton wheel if a compatible one already exists.

## Prepare the working directory on dev node

```bash
.celery/prepare-workdir.sh <working directory>
```

This script will:

1. Clone or update AOTriton source to `<working directory>/aotriton.src`
  - Performs a shallow clone from the `upstream/main` branch
  - If already cloned, performs `git pull` to update

2. Copy the following directory from `.celery` to `<working directory>`
  - `image.scripts`

3. Create `<working directory>/image.build/Dockerfile` from `config.rc`, which
  - Start from the `${CELERY_WORKER_IMAGE_BASE}`
  - Create venv according to `${CELERY_WORKER_PYTHON}` if the python file doesn't exist
  - Install requirements-tuning.txt
  - Run all scripts starting with "two digits+dash" under `<working directory>/image.scripts`

### Customization for different scenarios

The `Dockerfile` will run all scripts matching `[0-9][0-9]-*.sh` under
`image.scripts`. Here you have the flexibility to customize the image creation.

For example, if the base image has venv at `/root/venv` but the venv does not
have torch installed, the torch installation can be completed by adding the
following script as `<working directory>/image.scripts/90-install_torch.sh`

```bash
/root/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

## Deploy the working directory to worker nodes

```bash
.celery/deploy-workdir.sh <working directory>
```

This script will deploy the working directory to each registered GPU worker via rsync:
  + Syncs all files except
    - build (build artifacts are not synced)
    - scratch (temporary files)
    - run (runtime state)
    - installed (synced separately per architecture)
  + Only syncs `installed/<arch>` matching the worker's registered architecture
  + Uses each worker's configured working directory (default or custom override)

## Prepare the worker image from base worker image on each system

```bash
.celery/build-worker-image.sh <working directory>
```

In each GPU worker node:
* Run `docker build -f <working directory>/image.build/Dockerfile -t ${CELERY_WORKER_IMAGE} $WORKDIR`
  + The easiest practice is to prepare the image at dev node, but a writable docker registry may not be available
* This process is done through ssh+tsp to maximize parallism and avoid fragile ssh connections
  + tsp can be installed from package `task-spooler`, which is automated with `.celery/install-hostos-packages.sh`

## Start Server and GPU Worker

### Start Server

On the server node (or dev node if they are the same), start the RabbitMQ and PostgreSQL services:

```bash
.celery/srvctl.sh <working directory> start
```

This will start:
- RabbitMQ message broker (for task queue)
- PostgreSQL database (for result backend)

Both services run as Docker containers in detached mode with `--network=host`.

To stop the server services:

```bash
.celery/srvctl.sh <working directory> stop
```

To restart the server services:

```bash
.celery/srvctl.sh <working directory> restart
```

### Start GPU Workers

On the dev node, start all GPU workers:

```bash
.celery/wkctl.sh <working directory> start
```

This script will:
- SSH to each registered GPU worker
- Launch a Docker container with the worker image
- Start the Celery worker service inside the container
- Record the container ID in `<workdir>/run/worker.containerid`

### Stop GPU Workers

To stop all workers:

```bash
.celery/wkctl.sh <working directory> stop
```

This will stop the Celery worker service and remove the containers.

### Restart GPU Workers

To restart the Celery worker service without recreating containers:

```bash
.celery/wkctl.sh <working directory> restart
```

## Add Tuning Tasks to the Message Queue

After the server and workers are running, you can dispatch tuning tasks using the `dispatch-tasks.sh` script:

```bash
.celery/dispatch-tasks.sh <working directory> [options] <module> [module-options]
```

### Common Options

- `--arch ARCH [ARCH ...]`: Target architecture(s). If not specified, uses all registered workers.
- `--max_hsaco N`: Maximum number of hsaco kernels to tune per entry (default: all)
- `--wait`: Wait for all tasks to complete
- `--verbose`, `-v`: Verbose output

### Module-Specific Options

Each tuning module (e.g., `flash`) has its own parameter choices. To see available options for a module:

```bash
.celery/dispatch-tasks.sh <working directory> flash -h
```

This will show all parameter choices for the flash module, such as:
- `--dtype`: Filter by dtype. Choices: ['float16', 'bfloat16', 'float32']
- `--hdim`: Filter by hdim. Choices: [16, 32, 48, 64, ...]
- `--seqlen_q`: Filter by seqlen_q. Choices: [16, 32, 64, 128, ...]
- `--seqlen_k`: Filter by seqlen_k. Choices: [16, 32, 64, 128, ...]
- `--causal`: Filter by causal. Choices: [0, 1]
- `--dropout_p`: Filter by dropout_p. Choices: [0.0, 0.5]
- `--bias_type`: Filter by bias_type. Choices: [0, 1]

### Examples

Dispatch all flash tasks for gfx942:
```bash
.celery/dispatch-tasks.sh /path/to/workdir --arch gfx942 flash
```

Dispatch only float16 tasks with specific sequence lengths:
```bash
.celery/dispatch-tasks.sh /path/to/workdir --arch gfx942 flash \
  --dtype float16 --seqlen_q 128 256 --seqlen_k 128 256
```

Dispatch to multiple architectures:
```bash
.celery/dispatch-tasks.sh /path/to/workdir --arch gfx942 gfx90a flash
```

Limit number of hsaco kernels to tune per entry:
```bash
.celery/dispatch-tasks.sh /path/to/workdir --max_hsaco 5 flash --dtype float16
```

Wait for all tasks to complete with verbose output:
```bash
.celery/dispatch-tasks.sh /path/to/workdir --arch gfx942 --wait --verbose flash \
  --dtype float16 --hdim 64
```

If `--arch` is not specified, the script will dispatch tasks to all registered architectures in the workers database.

# Steps (SLURM)

For HPC environments using SLURM job scheduling, AOTriton provides native SLURM support as an alternative to the Docker-based worker deployment described above.

## Prerequisites

* A SLURM cluster with GPU nodes
* SSH access to the SLURM login node
* A shared filesystem accessible from all SLURM compute nodes (e.g., NFS, Lustre)
* Python 3.10+ available on SLURM nodes
* The dev node should have SSH access to the SLURM login node

## Create a working directory on the Dev Node

```bash
bash .celery/create-project-directory.sh <working directory>
```

During the interactive setup:
- When prompted "Enable SLURM support? (y/N):", answer `y`
- Provide the SLURM login node hostname (e.g., `slurm-login.hpc.example.com`)
- Provide the SLURM worker directory path (must be absolute and accessible to all compute nodes, e.g., `/home/username/aotriton-workdir`)

This will configure `config.rc` with:
```bash
SLURM_LOGIN_NODE="slurm-login.hpc.example.com"
SLURM_WORKER_DIR="/home/username/aotriton-workdir"
```

## Register SLURM Batch Configurations

Instead of registering individual GPU workers, register SLURM GRES (Generic RESource) constraints that will be used for job submission. Each registered entry will result in one SLURM job being submitted.

Add a single GRES configuration:

```bash
.celery/manage-workers.py <working directory> slurm-add "gpu:gfx942-mi300x:8"
```

Add multiple identical configurations (useful for launching multiple jobs with the same GRES):

```bash
.celery/manage-workers.py <working directory> slurm-add "gpu:gfx942-mi300x:8" --count 3
```

This will register 3 entries, resulting in 3 SLURM jobs being submitted with the same GRES constraint.

The GRES constraint should match your SLURM cluster's GPU configuration. Common formats:
- `gpu:8` - Request 8 GPUs of any type
- `gpu:mi300x:8` - Request 8 MI300X GPUs specifically
- `gpu:gfx942-mi300x:4` - Request 4 GPUs with specific architecture label

To list registered configurations:

```bash
.celery/manage-workers.py <working directory> slurm-list
```

Example output:
```
=== SLURM Batch Configurations ===

ID     GRES                                     Created
----------------------------------------------------------------------
1      gpu:gfx942-mi300x:8                      2026-04-10 12:00:00
2      gpu:gfx942-mi300x:8                      2026-04-10 12:00:00
3      gpu:gfx942-mi300x:8                      2026-04-10 12:00:00
4      gpu:gfx1100w:4                           2026-04-10 12:01:00

Total: 4 configuration(s)
```

To remove configurations by ID:

```bash
# Remove a single configuration
.celery/manage-workers.py <working directory> slurm-remove 1

# Remove multiple configurations
.celery/manage-workers.py <working directory> slurm-remove 2 3 4
```

## Mark Bad Nodes (Optional)

If certain SLURM nodes are experiencing issues, you can mark them as bad to exclude them from job submissions:

```bash
.celery/manage-workers.py <working directory> slurm-bad-add node-05 node-12 --reason "hardware failure"
```

Bad nodes will be automatically excluded using `sbatch --exclude` when submitting jobs.

To list bad nodes:

```bash
.celery/manage-workers.py <working directory> slurm-bad-list
```

To unmark nodes:

```bash
.celery/manage-workers.py <working directory> slurm-bad-remove node-05 node-12
```

## Build AOTriton for all Target Architectures

Same as the Docker workflow:

```bash
.celery/build-for-tuning.sh <working directory>
```

This builds:
1. A Triton wheel in `<working directory>/scratch/triton/`
2. AOTriton for all supported architectures (not just registered SLURM configs)

## Prepare the working directory on dev node

Same as the Docker workflow:

```bash
.celery/prepare-workdir.sh <working directory>
```

## Build SLURM Python Virtual Environment

Instead of building a Docker image, create a Python virtual environment on the SLURM login node:

```bash
.celery/build-slurm-venv.sh <working directory>
```

This script will:
1. SSH to the SLURM login node
2. Create a venv at `$SLURM_WORKER_DIR/installed/venv`
3. Install PyTorch from the official ROCm repository
4. Install the Triton wheel from `<working directory>/scratch/triton/`
5. Install requirements from `requirements-tuning.txt`
6. Apply Celery patches and install amdsmi

**Note:** The venv is shared across all SLURM compute nodes via the shared filesystem.

## Deploy the working directory to SLURM

```bash
.celery/deploy-workdir.sh <working directory>
```

This script will:
- Rsync the working directory to `$SLURM_LOGIN_NODE:$SLURM_WORKER_DIR`
- Deploy **all** AOTriton-supported architectures (not just registered SLURM configs)
- Exclude build/, scratch/, and run/ directories from the common sync
- Sync all architecture builds from `installed/` to the SLURM worker directory

## Start Server

On the server node (or dev node), start RabbitMQ and PostgreSQL services:

```bash
.celery/srvctl.sh <working directory> start
```

## Submit SLURM Jobs

Submit SLURM jobs for all registered GRES configurations:

```bash
.celery/srun.sh <working directory>
```

Or with a custom time limit:

```bash
.celery/srun.sh --time 08:00:00 <working directory>
.celery/srun.sh --time 2-00:00:00 <working directory>  # 2 days
```

Default time limit is 24 hours.

This script will:
1. SSH to the SLURM login node
2. Query the `slurm_batch` table for registered GRES constraints
3. Submit one `sbatch` job per GRES configuration
4. Automatically exclude bad nodes using `--exclude`
5. Record job IDs to `$SLURM_WORKER_DIR/run/slurm/jobs-<timestamp>.txt`

**Job Configuration:**
- Time limit: Configurable via `--time` (default: 24:00:00)
- Signal: SIGTERM sent 30 minutes before timeout for graceful shutdown
- Logs: `$SLURM_WORKER_DIR/run/celery-<hostname>/logs/`
- PIDs: `$SLURM_WORKER_DIR/run/celery-<hostname>/pids/`

**Note:** The hostname-specific directory structure avoids conflicts when multiple compute nodes write to the shared filesystem.

## Monitor and Manage Jobs

View running jobs:

```bash
ssh <SLURM_LOGIN_NODE> squeue -u $USER
```

View job details:

```bash
ssh <SLURM_LOGIN_NODE> scontrol show job <job_id>
```

Cancel jobs:

```bash
# Cancel a specific job
ssh <SLURM_LOGIN_NODE> scancel <job_id>

# Cancel all aotriton jobs
ssh <SLURM_LOGIN_NODE> scancel -u $USER -n aotriton
```

Cancel jobs from recorded job IDs file:

```bash
ssh <SLURM_LOGIN_NODE> "cd $SLURM_WORKER_DIR && scancel \$(cut -d'|' -f1 < run/slurm/jobs-<timestamp>.txt)"
```

## Add Tuning Tasks to the Message Queue

Same as the Docker workflow:

```bash
.celery/dispatch-tasks.sh <working directory> [options] <module> [module-options]
```

Example:

```bash
.celery/dispatch-tasks.sh /path/to/workdir --arch gfx942 flash --dtype float16
```

## Key Differences from Docker Workflow

| Aspect | Docker Workflow | SLURM Workflow |
|--------|----------------|----------------|
| Worker Management | Individual GPU nodes registered | GRES constraints registered |
| Deployment | Docker image per node | Shared Python venv on NFS |
| Worker Start | `wkctl.sh start` | `srun.sh [--time <time>]` |
| Worker Stop | `wkctl.sh stop` | `scancel <job_id>` |
| Logs/PIDs | Shared directory | Hostname-specific directories |
| Bad Node Handling | Manual exclusion | Automatic via `slurm_bad_nodes` table |
| Time Limits | Container lifetime | SLURM job time allocation |
| Architecture Support | Only registered workers | All AOTriton architectures deployed |

## Workflow Summary

For quick reference, the complete SLURM workflow is:

```bash
# Setup
.celery/create-project-directory.sh <workdir>  # Enable SLURM when prompted
.celery/manage-workers.py <workdir> slurm-add "gpu:gfx942-mi300x:8" --count 2
.celery/build-for-tuning.sh <workdir>
.celery/prepare-workdir.sh <workdir>
.celery/build-slurm-venv.sh <workdir>
.celery/deploy-workdir.sh <workdir>

# Start services
.celery/srvctl.sh <workdir> start
.celery/srun.sh --time 24:00:00 <workdir>

# Dispatch tasks
.celery/dispatch-tasks.sh <workdir> --arch gfx942 flash

# Monitor
ssh <SLURM_LOGIN_NODE> squeue -u $USER

# List and remove SLURM configurations
.celery/manage-workers.py <workdir> slurm-list
.celery/manage-workers.py <workdir> slurm-remove 1 2

# Cleanup
ssh <SLURM_LOGIN_NODE> scancel -u $USER
.celery/srvctl.sh <workdir> stop
```

