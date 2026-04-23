#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Stop worker on one host
# Usage: stop_worker.sh <workdir> <hostname>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2

# Parse optional --graceful flag
GRACEFUL=false
while [ $# -gt 0 ]; do
    case "$1" in
        --graceful)
            GRACEFUL=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 <workdir> <hostname> [--graceful]" >&2
            exit 1
            ;;
    esac
done

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [--graceful]" >&2
  echo "" >&2
  echo "  Stop the worker container on <hostname> via SSH." >&2
  echo "  --graceful  Signal worker service to finish in-flight tasks before stopping." >&2
  echo "  Use wkctl stop / wkctl graceful to operate on all workers at once." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" "$GRACEFUL" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
GRACEFUL="$3"
RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

if [ ! -f "$RUNFILE" ]; then
  echo "Worker not running or run file missing" >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(cat "$RUNFILE")

if [ "$GRACEFUL" = "true" ]; then
  echo "Gracefully stopping worker service in container: $WORKER_CONTAINER_ID"
  docker exec "$WORKER_CONTAINER_ID" bash -c "source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd /wkdir/aotriton.src && bash .tune/remote/worker_service.sh stop /wkdir $ARCH --graceful"
else
  echo "Stopping worker service in container: $WORKER_CONTAINER_ID"
  docker exec "$WORKER_CONTAINER_ID" bash -c "source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd /wkdir/aotriton.src && bash .tune/remote/worker_service.sh stop /wkdir $ARCH"
fi

echo "Stopping and removing container: $WORKER_CONTAINER_ID"
docker rm -f "$WORKER_CONTAINER_ID"

sudo rm -rf /dev/shm/aotriton-tuner
rm "$RUNFILE"
echo "Worker stopped and container removed"
EOF
