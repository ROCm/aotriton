#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

usage() {
  cat >&2 <<EOF
Usage: $0 [--host <hostname>...] <workdir>

Build worker container image on each registered GPU worker node via ssh+tsp.

Arguments:
  <workdir>  Project working directory

Options:
  --host <hostname>  Build image only on specified host(s). Can be specified multiple times.
                     If not specified, builds on all registered workers.

This script will:
  - SSH to each registered worker (or only specified hosts)
  - Use tsp (task-spooler) to queue docker build jobs
  - Build the image defined in <workdir>/image.build/Dockerfile
  - Tag the image as \${CELERY_WORKER_IMAGE}

Prerequisites:
  - tsp (task-spooler) installed on all workers
  - Docker access on all workers
  - Working directory already deployed via deploy-workdir.sh

Examples:
  # Build on all registered workers
  $0 /path/to/workdir

  # Build only on specific hosts
  $0 --host gpu-01.example.com --host gpu-02.example.com /path/to/workdir
EOF
  exit 1
}

# Parse arguments
TARGET_HOSTS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      TARGET_HOSTS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "Error: Unknown option: $1" >&2
      usage
      ;;
    *)
      WORKDIR="$1"
      shift
      ;;
  esac
done

if [ -z "$WORKDIR" ]; then
  echo "Error: Missing workdir argument" >&2
  usage
fi
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

# Function to check if hostname should be processed
should_build() {
  local hostname="$1"
  # If no target hosts specified, build on all
  if [ ${#TARGET_HOSTS[@]} -eq 0 ]; then
    return 0
  fi
  # Otherwise, check if hostname is in target list
  for target in "${TARGET_HOSTS[@]}"; do
    if [ "$hostname" = "$target" ]; then
      return 0
    fi
  done
  return 1
}

# Build image on selected workers
BUILD_COUNT=0
while IFS='|' read -r hostname workdir_override; do
  # Skip if not in target hosts
  if ! should_build "$hostname"; then
    continue
  fi

  # Determine remote workdir for this worker
  if [ -n "$workdir_override" ]; then
    WORKER_WORKDIR="$workdir_override"
  else
    WORKER_WORKDIR="$REMOTE_WORKDIR"
  fi

  echo "Queuing docker build on $hostname (workdir: $WORKER_WORKDIR)"

  # Certain nodes need --network=host to access internet
  ssh -n "$hostname" "tsp docker build --network=host -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR"

  BUILD_COUNT=$((BUILD_COUNT + 1))
done < <(sqlite3 "$WORKDIR/workers.db" "SELECT hostname, COALESCE(workdir_override, '') FROM workers ORDER BY hostname;")

if [ "$BUILD_COUNT" -eq 0 ]; then
  echo "Warning: No workers matched the specified criteria" >&2
  exit 1
fi

echo "Docker build jobs queued on $BUILD_COUNT worker(s). Monitor with: ssh <hostname> tsp"
