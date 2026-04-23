#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start worker on one host
# Usage: start_worker.sh <workdir> <hostname> [-- <extra args>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2

# Collect extra args after '--'
EXTRA_ARGS=()
if [ "$1" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
fi

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [-- <extra args>]" >&2
  echo "" >&2
  echo "  Start a worker container on <hostname> via SSH." >&2
  echo "" >&2
  echo "  WARNING: This script does NOT read GPU selection from the workers DB." >&2
  echo "  GPU assignment must be passed explicitly via extra args (e.g. -- --multi_gpu 0 1)." >&2
  echo "  Use wkctl start to apply GPU selection automatically." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# Add --hostname to extra args
EXTRA_ARGS=(--hostname "$HOSTNAME" "${EXTRA_ARGS[@]}")

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" "$CELERY_WORKER_IMAGE" "${EXTRA_ARGS[@]}" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
CELERY_WORKER_IMAGE="$3"
shift 3
EXTRA_ARGS=("$@")

RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

mkdir -p "$WORKER_WORKDIR/run"

if [ -f "$RUNFILE" ]; then
  echo "Worker already running or stale run file exists. Run stop first." >&2
  exit 1
fi

set -x
WORKER_CONTAINER_ID=$(docker run -d \
  --init \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -e PYTHONPATH=/wkdir/installed/$ARCH/lib \
  -e PYTHONPYCACHEPREFIX=/wkdir/run/pycache \
  --mount type=bind,source=$(realpath $WORKER_WORKDIR),target=/wkdir \
  "$CELERY_WORKER_IMAGE" \
  bash -c "source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd /wkdir/aotriton.src && bash .tune/remote/worker_service.sh start /wkdir $ARCH ${EXTRA_ARGS[*]} && exec sleep infinity")

if [ -z "$WORKER_CONTAINER_ID" ]; then
  echo "Failed to start container" >&2
  exit 1
fi

echo "$WORKER_CONTAINER_ID" > "$RUNFILE"
echo "Started container: $WORKER_CONTAINER_ID"
EOF
