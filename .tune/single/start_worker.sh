#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start worker on one host
# Usage: start_worker.sh <workdir> <hostname>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname>" >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" "$CELERY_WORKER_IMAGE" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
CELERY_WORKER_IMAGE="$3"
RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

mkdir -p "$WORKER_WORKDIR/run"

if [ -f "$RUNFILE" ]; then
  echo "Worker already running or stale run file exists. Run stop first." >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(docker run -d \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -e PYTHONPATH=/wkdir/installed/$ARCH/lib \
  --mount type=bind,source=$(realpath $WORKER_WORKDIR),target=/wkdir \
  "$CELERY_WORKER_IMAGE" \
  bash -c 'source /wkdir/config.rc && source $(dirname $CELERY_WORKER_PYTHON)/activate && bash /wkdir/aotriton.src/.celery/worker-service.sh start /wkdir && exec sleep infinity')

if [ -z "$WORKER_CONTAINER_ID" ]; then
  echo "Failed to start container" >&2
  exit 1
fi

echo "$WORKER_CONTAINER_ID" > "$RUNFILE"
echo "Started container: $WORKER_CONTAINER_ID"
EOF
