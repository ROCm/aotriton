#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Prerequisite:
#   1. Launch a container with ${CELERY_WORKER_IMAGE}. Ideally with --network host
#   2. Clone aotriton into the container
#   3. Copy the workdir from controller node to the container
#   4. Run this script inside the container with workdir
# 2/3 can be prepared without docker cp by binding a working directory to
# container

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi
if [ "$#" -ne 2 ]; then
  echo 'Missing arguments. Usage: start-worker.sh <action> <dir>' >&2
  echo '<action> can be start|stop|restart' >&2
  exit 1
fi

action="$1"
dir="$2"

if [ $action != "stop" ]; then
  # Reserved for final release
  # import torch takes too much time for debugging purpose
  torch_lib=$(python -c "import torch; from pathlib import Path; print((Path(torch.__file__).parent/'lib').as_posix())")
  export LD_LIBRARY_PATH=${torch_lib}:$LD_LIBRARY_PATH
  python -c 'import pyaotriton'
  if [ $? -ne 0 ]; then
    echo "Cannot import pyaotriton in python. Forget to set PYTHONPATH ?" >&2
    exit 1
  fi
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/../.ci/common-vars.sh"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"


if [ ! -d "$dir" ]; then
  echo 'Not a directory: $dir' >&2
  exit 1
fi
rcfile="$dir/config.rc"

. "$rcfile"

cd ${SCRIPT_DIR}/..

CPUQ="$(hostname -s)_cpuqueue"
GPUQ="$(hostname -s)_gpuqueue"

# Here we propose a triple queue design to address shortcomings of Celery
#
# 1. We want to saturate the GPU
# 2. We want load balancing among GPU nodes
# 3. We need a pre-processing step to prepare testing data on /dev/shm/
# 4. We do not know the number of sub-tasks for a given tuning task, until the
#    compiled hsaco kernels are "probed"
# 5. We need a post-processing step to clean up testing data
#
# Load balancing requires blocking fetching tasks, otherwise a single node can
# consume all tasks in the broker. GPU saturation requires multiple fetching
# workers, otherwise GPU will be idle
# when the single fetcher is waiting for the current task to complete.
# Here 4 fetcher_* workers are created.
#
# 3-5 requires a two-level celery task, the first level (tune_kernel) prepares
# the data, and the second level (do_tune_kernel) creates actual hsaco tuning
# tasks from the prepared data.  The "probing" is the first step in the second
# level, running on GPU, and its output will be used to create a chord to tune
# hsaco kernels, with post-processing as the callback.
#
# However, since it seems impossible to return chord's AsyncResult as the
# result of do_tune_kernel, it is then necessary to wait the chord's complete.
# The waiting requires another queue and a set of dedicated workers here
# referred as dispatchers.
#
# NOTE: DO NOT USE ADVANCED -Q SYNTAX LIKE -Q:1-4 OR -Q:1,2,3,4. IT SEEMS
# CELERY MULTI HAS PARSING BUGS

AOTRITON_CELERY_WORKDIR=$dir \
celery multi ${action} \
  fetcher_0 fetcher_1 fetcher_2 fetcher_3 \
  dispatcher_0 dispatcher_1 dispatcher_2 dispatcher_3 \
  `seq -s ' ' -f 'gpu_%g' 0 $((ngpus -1))` -A v3python.celery -l info -c 1 \
  -Q:1 ${native_arch} \
  -Q:2 ${native_arch} \
  -Q:3 ${native_arch} \
  -Q:4 ${native_arch} \
  -Q:5 ${CPUQ} \
  -Q:6 ${CPUQ} \
  -Q:7 ${CPUQ} \
  -Q:8 ${CPUQ} \
  -Q ${GPUQ} \
  --pidfile=$dir/run/celery/pids/%n.pid \
  --logfile=$dir/run/celery/logs/%n__%i.log
