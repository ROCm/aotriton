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

TODO

## Deploy Tuning Environment to Server and All GPU Workers

The deploy includes:

* The working directory
* AOTriton source tree (for various scripts)
* AOTriton binary (only on GPU Workers)
* A venv for

TODO

## Start Server and GPU Worker

TODO

## Add Tuning Tasks to the Message Queue

TODO

## Analysis the code

