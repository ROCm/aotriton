#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Sync workdir to one host (main files + architecture-specific files)
# Usage: sync_workdir.sh <workdir> <hostname> [--remote_workdir <path>|--buildnode]
#
# --remote_workdir <path>
#   Override the remote workdir instead of looking it up from workers.db.
#   When set, arch is treated as ALL (sync all installed/ subdirs).
#   Use this for hosts not registered as GPU workers (e.g. build nodes).
#
# --buildnode
#   Look up buildnode::workdir_override (falling back to default_workdir) from
#   workers.db config table. Sets SUBDIR to 'database' so only installed/database/
#   is synced to the remote (the build node needs the sharded DB, not GPU binaries).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2

REMOTE_WORKDIR_OVERRIDE=""
BUILDNODE_MODE=0
TESTNODE_MODE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --remote_workdir) REMOTE_WORKDIR_OVERRIDE="$2"; shift 2 ;;
    --buildnode)      BUILDNODE_MODE=1; shift ;;
    --testnode)       TESTNODE_MODE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [--remote_workdir <path>]" >&2
  echo "" >&2
  echo "  Rsync workdir and arch-specific installed/ files to <hostname>." >&2
  echo "  Excludes build/, run/, scratch/, secrets/. Uses --delete on installed/ and aotriton.src/." >&2
  echo "" >&2
  echo "  --remote_workdir <path>  Override remote workdir (skips workers.db lookup, syncs all installed/)." >&2
  exit 1
fi

load_config "$WORKDIR"

if [ -n "$REMOTE_WORKDIR_OVERRIDE" ]; then
  arch="ALL"
  WORKER_WORKDIR="$REMOTE_WORKDIR_OVERRIDE"
elif [ "$BUILDNODE_MODE" -eq 1 ]; then
  WORKER_WORKDIR="$(get_buildnode_workdir "$WORKDIR")"
else
  # Get arch and workdir_override for this hostname
  WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
  IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"
  WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"
fi

# Sync main directories (exclude build, installed, run, scratch, secrets, aotriton.src)
# --mkpath creates $WORKER_WORKDIR if it doesn't exist
# aotriton.src synced below with architecture-specific files
rsync -az --checksum --info=progress2 \
  --exclude '/build/' \
  --exclude '/installed/' \
  --exclude '/run/' \
  --exclude '/scratch/' \
  --exclude '/secrets/' \
  --exclude '/aotriton.src/' \
    --mkpath \
  "$WORKDIR/" "$HOSTNAME:$WORKER_WORKDIR/"

# Sync architecture-specific files and aotriton.src with --delete
# --delete ensures exact copy, removing stale files, deleted files, old Ray code
# --exclude '*.pyc' prevents deleting bytecode (may be root-owned from container)
# We minimize rsync calls since some deployments have long SSH authentication time
# TODO: Re-use SSH connection between multiple rsyncs (e.g., SSH ControlMaster)
if [ "$BUILDNODE_MODE" -eq 1 ]; then
  SUBDIR="/database"
elif [ "$TESTNODE_MODE" -eq 1 ]; then
  SUBDIR="/test/$arch"
elif [ "$arch" = "ALL" ]; then
  SUBDIR=""
else
  SUBDIR="/$arch"
fi

# Always use --delete so aotriton.src is an exact copy.
# When SUBDIR is empty (ALL), protect installed/ from deletion so remote
# arch dirs not present locally are not wiped.
FILTER_ARGS=()
[ -z "$SUBDIR" ] && FILTER_ARGS+=("--filter=P /installed/***")

SOURCES=()
[ -d "$WORKDIR/installed$SUBDIR" ] && SOURCES+=("$WORKDIR/./installed$SUBDIR")
SOURCES+=("$WORKDIR/./aotriton.src")

rsync -azR --checksum --info=progress2 --delete "${FILTER_ARGS[@]}" --exclude '*.pyc' \
  "${SOURCES[@]}" \
  "$HOSTNAME:$WORKER_WORKDIR/./"
