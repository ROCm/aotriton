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
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/../.ci/common-vars.sh"

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: start-worker.sh <dir>' >&2
  exit 1
fi
dir="$1"

if [ ! -d "$dir" ]; then
  echo 'Not a directory: $dir' >&2
  exit 1
fi
rcfile="$dir/config.rc"

. "$rcfile"

cd ${SCRIPT_DIR}/..

export AOTRITON_CELERY_WORKDIR=$dir
export AOTRITON_CELERY_LQ="$(hostname -s)_localqueue"

celery multi start `seq -s ' ' -f 'gpu%g' 0 $((ngpus -1))` -A v3python.celery -l info --concurrency 1 \
  --pidfile=$dir/run/celery/pids/%n.pid \
  --logfile=$dir/run/celery/logs/%n%i.log
