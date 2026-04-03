# Overview

The tuning tool `test/tune_flash.py` was deprecated in favor of a distributed
tuning framework based on [celery](https://github.com/celery/celery).

# Prerequisites

## System

* A group GPU workers
* A host that are accessible to all GPU workers
  - TBD: Required ports for potential firewall configurations.
  - This host will be referred as the "Server" in the following text.
* A host that can access all GPU workers with `ssh`
  - This host will be referred as the "Dev Node" in the following text.
  - Dev Node and Server can be the same host.

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


Other packages needed by the host OS on the nodes can be installed with `.celery/install-hostos-packages.sh`

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
   - Builds are stored in `<working directory>/build/<arch>/`
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
image.scripts. Here you have the flexibility to customize the image creation.

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
    - build
    - scratch
  + Only syncs `build/<arch>` matching the worker's registered architecture
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

TODO

## Add Tuning Tasks to the Message Queue

TODO

## Analysis the code

