#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Docker image on one host
# Usage: build_image.sh <workdir> <hostname> [--follow]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
FOLLOW=""

# Parse optional --follow flag
if [ "$3" = "--follow" ]; then
  FOLLOW="true"
fi

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [--follow]" >&2
  echo "" >&2
  echo "  Submit a Docker image build job via tsp on <hostname>." >&2
  echo "  --follow  Tail the build output in real-time (blocks until done)." >&2
  echo "  Without --follow, the job runs in background; check with tsp on the host." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# Certain nodes need --network=host to access internet
if [ -n "$FOLLOW" ]; then
  # Use tsp -t to tail/follow output in real-time
  ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$CELERY_WORKER_IMAGE" <<'EOF'
WORKER_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"

jobid=$(tsp docker build --network=host -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR)
echo "Job ID: $jobid"
if [ "$(tsp -s "$jobid")" = "queued" ]; then
  echo "Waiting for tsp job $jobid to start..."
  while [ "$(tsp -s "$jobid")" = "queued" ]; do sleep 5; done
fi
tsp -t $jobid
EOF
else
  ssh -n "$HOSTNAME" "tsp docker build --network=host -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR"
fi
