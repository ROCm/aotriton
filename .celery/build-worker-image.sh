#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir>

Build worker container image on each registered GPU worker node via ssh+tsp.

Arguments:
  <workdir>  Project working directory

This script will:
  - SSH to each registered worker
  - Use tsp (task-spooler) to queue docker build jobs
  - Build the image defined in <workdir>/image.build/Dockerfile
  - Tag the image as \${CELERY_WORKER_IMAGE}

Prerequisites:
  - tsp (task-spooler) installed on all workers
  - Docker access on all workers
  - Working directory already deployed via deploy-workdir.sh
EOF
  exit 1
fi

WORKDIR="$1"
CONFIG_RC="$WORKDIR/config.rc"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ] || [ ! -f "$CONFIG_RC" ]; then
  echo "Error: Invalid workdir, workers.db, or config.rc not found" >&2
  exit 1
fi

# Source config
. "$CONFIG_RC"

if [ -z "$CELERY_WORKER_IMAGE" ]; then
  echo "Error: CELERY_WORKER_IMAGE not set in config.rc" >&2
  exit 1
fi

# Get default working directory for workers
REMOTE_WORKDIR=$(sqlite3 "$WORKDIR/workers.db" "SELECT value FROM config WHERE key = 'default_workdir';" 2>/dev/null)
if [ -z "$REMOTE_WORKDIR" ]; then
  echo "Error: Default working directory not set. Use manage-workers.py set-default-workdir" >&2
  exit 1
fi

# Build image on each worker
sqlite3 "$WORKDIR/workers.db" "SELECT hostname, COALESCE(workdir_override, '') FROM workers ORDER BY hostname;" | while IFS='|' read -r hostname workdir_override; do
  # Determine remote workdir for this worker
  if [ -n "$workdir_override" ]; then
    WORKER_WORKDIR="$workdir_override"
  else
    WORKER_WORKDIR="$REMOTE_WORKDIR"
  fi

  echo "Queuing docker build on $hostname (workdir: $WORKER_WORKDIR)"

  ssh -n "$hostname" "tsp docker build -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR"
done

echo "All docker build jobs queued. Monitor with: ssh <hostname> tsp"
