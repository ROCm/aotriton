#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Graceful shutdown worker on one host
# Usage: stopwait_worker.sh <workdir> <hostname>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname>" >&2
  echo "" >&2
  echo "  Cancel task fetching, drain the local queue, then stop the worker container." >&2
  echo "  Waits up to 10 minutes for in-flight tasks to complete." >&2
  echo "  Use wkctl stopwait to operate on all workers at once." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

if [ ! -f "$RUNFILE" ]; then
  echo "Worker not running or run file missing" >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(cat "$RUNFILE")
HOSTNAME=$(hostname -s)

echo "Gracefully stopping workers in container: $WORKER_CONTAINER_ID"
docker exec "$WORKER_CONTAINER_ID" bash -c "
source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate
cd /wkdir/aotriton.src

echo 'Step 1: Cancel broker consumers on fetcher workers...'
celery -A v3python.celery control cancel_consumer $ARCH \
  -d fetcher_0@$HOSTNAME \
  -d fetcher_1@$HOSTNAME \
  -d fetcher_2@$HOSTNAME \
  -d fetcher_3@$HOSTNAME

echo 'Step 2: Waiting for local queues to drain (max 10 min)...'
timeout 600 bash -c 'while celery -A v3python.celery inspect active | grep -q \"task\"; do sleep 10; done' || true

echo 'Step 3: Stopping all workers gracefully...'
bash /wkdir/aotriton.src/.tune/remote/worker_service.sh stopwait /wkdir
"

echo "Stopping and removing container: $WORKER_CONTAINER_ID"
docker stop "$WORKER_CONTAINER_ID"
docker rm "$WORKER_CONTAINER_ID"

rm -rf /dev/shm/aotriton-tuner
rm "$RUNFILE"
echo "Worker stopped and container removed"
EOF
