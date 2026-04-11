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

Deploy working directory to all registered GPU workers.

Arguments:
  <workdir>  Project working directory (created by create-project-directory.sh)

This script will:
  - Rsync working directory to each registered worker
  - Exclude build/, scratch/, and run/ directories
  - Only sync build/<arch> matching worker's architecture

Requirements:
  - SSH access to all registered workers
  - rsync installed on both local and remote nodes
EOF
  exit 1
fi

WORKDIR="$1"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Load config for SLURM deployment
CONFIG_RC="$WORKDIR/config.rc"
if [ -f "$CONFIG_RC" ]; then
  . "$CONFIG_RC"
fi

# Get default working directory for workers
REMOTE_WORKDIR=$(sqlite3 "$WORKDIR/workers.db" "SELECT value FROM config WHERE key = 'default_workdir';" 2>/dev/null)
if [ -z "$REMOTE_WORKDIR" ]; then
  echo "Error: Default working directory not set. Use manage-workers.py set-default-workdir" >&2
  exit 1
fi

deploy() {
  local hostname="$1"
  local arch="$2"
  local workdir_override="$3"
  # Determine remote workdir for this worker
  if [ -n "$workdir_override" ]; then
    local WORKER_WORKDIR="$workdir_override"
  else
    local WORKER_WORKDIR="$REMOTE_WORKDIR"
  fi
  echo "Deploying to $hostname ($arch) -> $WORKER_WORKDIR"
  # CAVEAT FOR ADDING SSH COMMAND:
  #   MUST PASS -n otherwise stdin will be consumed
  # (ssh -n "$hostname" "mkdir -p $WORKER_WORKDIR")

  # Rsync everything except top-level build, scratch, and run directories
  # Note need to keep .git so git can work in wkdir/aotriton.src
  set +x
  rsync -az \
    --info=progress2 \
    --exclude '/build/' \
    --exclude '/scratch/' \
    --exclude '/run/' \
    --exclude '/installed/' \
    --mkpath \
    "$WORKDIR/" "$hostname:$WORKER_WORKDIR/"

  echo $?

  # Rsync architecture-specific build directories
  local subdir=""
  if [ "$arch" != "ALL" ]; then
    subdir="/$arch"
  fi
  if [ -d "$WORKDIR/installed$subdir" ]; then
    rsync -azR --info=progress2 "$WORKDIR/./installed$subdir" "$hostname:$WORKER_WORKDIR/./"
  fi
  echo "Deployed to $hostname ($arch) -> $WORKER_WORKDIR"
}

sqlite3 "$WORKDIR/workers.db" "SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers ORDER BY hostname;" | while IFS='|' read h a w; do
  deploy "$h" "$a" "$w"
done

# Deploy to SLURM if configured
if [ -n "$SLURM_LOGIN_NODE" ]; then
  deploy "$SLURM_LOGIN_NODE" "ALL" "$SLURM_WORKER_DIR"
fi

echo "Deployment completed"
