#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Automated redeploy and restart script for the tuning infrastructure

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<EOF
Usage: $0 <workdir> [options]

Automated redeploy and restart of the entire tuning infrastructure.

Arguments:
  <workdir>  Project working directory

Options:
  --rebuild_library  Rebuild AOTriton binaries (default: skip)
  --rebuild_image    Rebuild worker Docker images (deletes old images first)
  --skip_deploy      Skip workdir deployment (useful if only restarting services)
  --help             Show this help message

This script performs the following steps:
  1. Stop all GPU workers
  2. Stop server services (RabbitMQ, PostgreSQL)
  3. Prepare working directory (update source, scripts)
  4. Rebuild AOTriton binaries if --rebuild_library
  5. Deploy working directory to workers (unless --skip_deploy)
  6. Rebuild worker images if --rebuild_image (deletes old images first)
  7. Start server services
  8. Start all GPU workers

Examples:
  # Full redeploy with image rebuild
  $0 /path/to/workdir --rebuild_image

  # Redeploy without rebuilding images
  $0 /path/to/workdir

  # Just restart services without redeploying files
  $0 /path/to/workdir --skip_deploy
EOF
  exit 1
}

# Parse arguments
WORKDIR=""
REBUILD_IMAGE=false
REBUILD_LIBRARY=false
SKIP_DEPLOY=false

while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild_image)
      REBUILD_IMAGE=true
      shift
      ;;
    --rebuild_library)
      REBUILD_LIBRARY=true
      shift
      ;;
    --skip_deploy)
      SKIP_DEPLOY=true
      shift
      ;;
    --help|-h)
      usage
      ;;
    *)
      if [ -z "$WORKDIR" ]; then
        WORKDIR="$1"
      else
        echo "Error: Unknown argument '$1'" >&2
        usage
      fi
      shift
      ;;
  esac
done

if [ -z "$WORKDIR" ]; then
  echo "Error: workdir argument required" >&2
  usage
fi

# Validate workdir
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found at $WORKDIR" >&2
  exit 1
fi

# Load config
CONFIG_RC="$WORKDIR/config.rc"
if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

. "$CONFIG_RC"

if [ -z "$CELERY_WORKER_IMAGE" ]; then
  echo "Error: CELERY_WORKER_IMAGE not set in config.rc" >&2
  exit 1
fi

set -e  # Exit on error

echo "Redeploy: rebuild_image=$REBUILD_IMAGE rebuild_library=$REBUILD_LIBRARY skip_deploy=$SKIP_DEPLOY"

# Step 1: Stop workers
echo "[1/9] Stop workers"
"$SCRIPT_DIR/wkctl.sh" "$WORKDIR" stop 2>/dev/null || true

# Step 2: Stop server
echo "[2/9] Stop server"
"$SCRIPT_DIR/srvctl.sh" "$WORKDIR" stop 2>/dev/null || true

# Step 3: Prepare workdir
echo "[3/9] Prepare workdir"
"$SCRIPT_DIR/prepare-workdir.sh" "$WORKDIR"

# Step 4: Build AOTriton (if requested)
if [ "$REBUILD_LIBRARY" = true ]; then
  echo "[4/9] Build AOTriton"
  "$SCRIPT_DIR/build-for-tuning.sh" "$WORKDIR"
else
  echo "[4/9] Skip library build"
fi

# Step 5: Deploy workdir (unless skipped)
if [ "$SKIP_DEPLOY" = false ]; then
  echo "[5/9] Deploy workdir"
  "$SCRIPT_DIR/deploy-workdir.sh" "$WORKDIR"
else
  echo "[5/9] Skip deploy"
fi

# Step 6-7: Rebuild worker images if requested
if [ "$REBUILD_IMAGE" = true ]; then
  echo "[6/9] Delete old images"
  "$SCRIPT_DIR/ssh-all.sh" "$WORKDIR" docker rmi -f "$CELERY_WORKER_IMAGE" 2>/dev/null || true
  echo "[7/9] Build worker images"
  "$SCRIPT_DIR/build-worker-image.sh" "$WORKDIR"

  # Wait for all tsp jobs to complete
  echo "[7/9] Waiting for docker builds to complete..."
  HOSTNAMES=($(sqlite3 "$WORKDIR/workers.db" "SELECT DISTINCT hostname FROM workers ORDER BY hostname;"))
  for hostname in "${HOSTNAMES[@]}"; do
    ssh "$hostname" "tsp -w" >/dev/null 2>&1 || true
  done
  echo "[7/9] All docker builds completed"
else
  echo "[6/9] Skip image rebuild"
fi

# Step 8: Start server
echo "[8/9] Start server"
"$SCRIPT_DIR/srvctl.sh" "$WORKDIR" start

# Step 9: Start workers
echo "[9/9] Start workers"
"$SCRIPT_DIR/wkctl.sh" "$WORKDIR" start

echo "Done. Dispatch: .celery/dispatch-tasks.sh $WORKDIR <module> [options]"
