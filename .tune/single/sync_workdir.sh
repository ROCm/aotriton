#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Sync workdir to one host (main files + architecture-specific files)
# Usage: sync_workdir.sh <workdir> <hostname>

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

# Create directory structure
ssh "$HOSTNAME" mkdir -p "$WORKER_WORKDIR"

# Sync main directories (exclude build, installed, run, scratch, secrets, aotriton.src)
# aotriton.src synced below with architecture-specific files
rsync -az --info=progress2 \
  --exclude '/build/' \
  --exclude '/installed/' \
  --exclude '/run/' \
  --exclude '/scratch/' \
  --exclude '/secrets/' \
  --exclude '/aotriton.src/' \
    --mkpath \
  "$WORKDIR/" "$HOSTNAME:$WORKER_WORKDIR/"

# Sync architecture-specific files and aotriton.src with --delete
# --delete ensures exact copy, removing stale .pyc, deleted files, old Ray code
# We minimize rsync calls since some deployments have long SSH authentication time
# TODO: Re-use SSH connection between multiple rsyncs (e.g., SSH ControlMaster)
if [ "$arch" = "ALL" ]; then
  SUBDIR=""
else
  SUBDIR="/$arch"
fi

if [ -d "$WORKDIR/installed$SUBDIR" ]; then
  # Sync both installed/$arch and aotriton.src in single rsync
  rsync -azR --info=progress2 --delete \
    "$WORKDIR/./installed$SUBDIR" \
    "$WORKDIR/./aotriton.src" \
    "$HOSTNAME:$WORKER_WORKDIR/./"
else
  # Sync aotriton.src only if installed/$arch doesn't exist
  rsync -az --info=progress2 --delete \
    "$WORKDIR/aotriton.src/" \
    "$HOSTNAME:$WORKER_WORKDIR/aotriton.src/"
fi
