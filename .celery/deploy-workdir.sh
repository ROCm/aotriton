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

Prepare and deploy working directory to all registered GPU workers.

Arguments:
  <workdir>  Project working directory (created by create-project-directory.sh)

Steps:
  1. Clone or update AOTriton source to <workdir>/aotriton.src
  2. Rsync working directory to each registered worker
     - Only syncs build/<arch> matching worker's architecture

Requirements:
  - SSH access to all registered workers
  - rsync installed on both local and remote nodes
EOF
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
AOTRITON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="$1"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Get default working directory for workers
REMOTE_WORKDIR=$(sqlite3 "$WORKDIR/workers.db" "SELECT value FROM config WHERE key = 'default_workdir';" 2>/dev/null)
if [ -z "$REMOTE_WORKDIR" ]; then
  echo "Error: Default working directory not set. Use manage-workers.py set-default-workdir" >&2
  exit 1
fi

# Step 1: Prepare the working directory
AOTRITON_SRC="$WORKDIR/aotriton.src"

if [ -d "$AOTRITON_SRC/.git" ]; then
  echo "Updating AOTriton source..."
  cd "$AOTRITON_SRC"
  git pull upstream main
else
  echo "Cloning AOTriton source..."
  cd "$AOTRITON_ROOT"
  UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null)
  if [ -z "$UPSTREAM_URL" ]; then
    echo "Error: 'upstream' remote not found in AOTriton repository" >&2
    exit 1
  fi
  git clone --depth 1 --single-branch --branch main "$UPSTREAM_URL" "$AOTRITON_SRC"
fi

# Step 2: Deploy to each worker
sqlite3 "$WORKDIR/workers.db" "SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers ORDER BY hostname;" | while IFS='|' read -r hostname arch workdir_override; do
  # Determine remote workdir for this worker
  if [ -n "$workdir_override" ]; then
    WORKER_WORKDIR="$workdir_override"
  else
    WORKER_WORKDIR="$REMOTE_WORKDIR"
  fi

  echo "Deploying to $hostname ($arch) -> $WORKER_WORKDIR"

  # Create remote workdir
  ssh "$hostname" "mkdir -p $WORKER_WORKDIR"

  # Rsync everything except build directories
  rsync -az --info=progress2 --exclude 'build/' --exclude '.git/' \
    "$WORKDIR/" "$hostname:$WORKER_WORKDIR/"

  # Rsync only the specific architecture build directory
  if [ -d "$WORKDIR/build/$arch" ]; then
    rsync -az --info=progress2 "$WORKDIR/build/$arch/" "$hostname:$WORKER_WORKDIR/build/$arch/"
  fi
done

echo "Deployment completed"
