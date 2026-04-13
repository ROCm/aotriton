#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Docker image on one host
# Usage: build_image.sh <workdir> <hostname>

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

# Certain nodes need --network=host to access internet
ssh -n "$HOSTNAME" "tsp docker build --network=host -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR"
