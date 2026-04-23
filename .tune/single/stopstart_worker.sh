#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Stop then start worker on one host
# Usage: stopstart_worker.sh <workdir> <hostname> [-- <extra args for start>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKDIR="$1"
HOSTNAME="$2"
shift 2

# Collect extra args after '--'
EXTRA_ARGS=()
if [ "$1" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
fi

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [-- <extra args>]" >&2
  echo "" >&2
  echo "  Stop then start the worker container on <hostname> (ignores stop errors)." >&2
  echo "" >&2
  echo "  WARNING: This script does NOT read GPU selection from the workers DB." >&2
  echo "  GPU assignment must be passed explicitly via extra args (e.g. -- --multi_gpu 0 1)." >&2
  echo "  Use wkctl restart to apply GPU selection automatically." >&2
  exit 1
fi

# Stop worker (ignore errors if not running)
"$SCRIPT_DIR/stop_worker.sh" "$WORKDIR" "$HOSTNAME" || true

# Start worker with extra args
"$SCRIPT_DIR/start_worker.sh" "$WORKDIR" "$HOSTNAME" "${EXTRA_ARGS[@]}"
