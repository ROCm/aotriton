#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Restart worker on one host
# Usage: restart_worker.sh <workdir> <hostname> [-- <extra args>]

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
  echo "  Restart the worker service inside an already-running container on <hostname>." >&2
  echo "" >&2
  echo "  WARNING: This script does NOT read GPU selection from the workers DB." >&2
  echo "  GPU assignment must be passed explicitly via extra args (e.g. -- --multi_gpu 0 1)." >&2
  echo "  Use wkctl restart to apply GPU selection automatically." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" "${EXTRA_ARGS[@]}" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
shift 2
EXTRA_ARGS=("$@")

RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

if [ ! -f "$RUNFILE" ]; then
  echo "Worker not running or run file missing" >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(cat "$RUNFILE")

echo "Restarting worker service in container: $WORKER_CONTAINER_ID"
docker exec "$WORKER_CONTAINER_ID" bash -c "source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd /wkdir/aotriton.src && bash .tune/remote/worker_service.sh restart /wkdir $ARCH ${EXTRA_ARGS[*]}"
echo "Worker restarted"
EOF
