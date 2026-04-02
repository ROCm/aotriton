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

* ssh
* docker
* A common docker image serve as the runtime environment
* patched celery wheel (see section below to build)

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

## Configure Server and GPU Workers

TODO

## Build AOTriton for all Target Architectures

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

## Prepare and Deploy the working directory

```bash
.celery/deploy-workdir.sh <working directory>
```

This script will:

1. Clone or update AOTriton source to `<working directory>/aotriton.src`
   - Performs a shallow clone from the `upstream/main` branch
   - If already cloned, performs `git pull` to update

2. Deploy the working directory to each registered GPU worker via rsync
   - Syncs all files except build directories
   - Only syncs `build/<arch>` matching the worker's registered architecture
   - Uses each worker's configured working directory (default or custom override)

## Configure the venv in GPU worker

**IMPORTANT: This step assume `triton-*.whl` and `torch-*.whl` are available at /
of the container image. Either add them to the container image, or add <working
directory>/hooks/pre-venv.sh to copy them to root**

In each GPU worker node:
* Launch the container with image specified by `${CELERY_WORKER_IMAGE}` with mounting the `<working directory>` to `/wkdir`
* Inside the container
  - Create venv at `/wkdir/venv`
  - Run script `/wkdir/inpod.create_venv.sh` if exists.
  - Install triton and torch from `/triton-*.whl` and `/torch-*.whl`
  - Install other dependencies from `/wkdir/aotriton.src/requirements-tuning.txt`

## Start Server and GPU Worker

TODO

## Add Tuning Tasks to the Message Queue

TODO

## Analysis the code

