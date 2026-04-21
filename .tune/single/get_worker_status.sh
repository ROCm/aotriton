#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Get worker status (container ID and GPU process count)
# Usage: get_worker_status.sh <workdir> <hostname>

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

# Get workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"
WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# Copy remote script to worker and execute
REMOTE_SCRIPT="$TUNE_ROOT/remote/get_status.sh"

set +e
ssh -T -o LogLevel=ERROR "$HOSTNAME" "bash -s" < "$REMOTE_SCRIPT" "$WORKER_WORKDIR"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
  echo "status=error"
  echo "message=Remote script failed"
  exit 1
fi
