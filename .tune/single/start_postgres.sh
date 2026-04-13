#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start PostgreSQL container
# Usage: start_postgres.sh <workdir>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"

WORKDIR="$1"

if [ -z "$WORKDIR" ]; then
  echo "Usage: $0 <workdir>" >&2
  exit 1
fi

load_config "$WORKDIR"

# Container name with suffix
POSTGRES_CONTAINER="aotriton_pgsql.${CONTAINER_SUFFIX}"

# PID file
mkdir -p "$WORKDIR/run"
PIDF="$WORKDIR/run/container.pids"

echo "Starting PostgreSQL..."
POSTGRES_ID=$(docker run --ipc=host \
  --network=host \
  -d \
  --rm \
  --ulimit nofile=65536:65536 \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -v "${POSTGRES_DOCKER_VOLUME}:/var/lib/postgresql/data" \
  --name "${POSTGRES_CONTAINER}" \
  "${POSTGRES_DOCKER_IMAGE}" \
  postgres -c max_connections=500 -c shared_buffers=2GB)

if [ -z "$POSTGRES_ID" ]; then
  echo "Error: Failed to start PostgreSQL" >&2
  exit 1
fi
echo "$POSTGRES_ID" >> "$PIDF"
echo "Started PostgreSQL: $POSTGRES_ID"
