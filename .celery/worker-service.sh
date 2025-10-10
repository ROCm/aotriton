#!/bin/bash

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

export AOTRITON_CELERY_WORKDIR=$dir
export AOTRITON_CELERY_CPUQ="$(hostname -s)_cpuqueue"
export AOTRITON_CELERY_GPUQ="$(hostname -s)_gpuqueue"

celery multi ${action} dispatcher `seq -s ' ' -f 'gpu_%g' 0 $((ngpus -1))` -A v3python.celery -l info -c 1 \
  -Q:1 ${native_arch} \
  -Q ${AOTRITON_CELERY_GPUQ} \
  --pidfile=$dir/run/celery/pids/%n.pid \
  --logfile=$dir/run/celery/logs/%n%i.log
