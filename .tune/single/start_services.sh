#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start PostgreSQL container
# Usage: start_services.sh <workdir>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"

WORKDIR="$1"

if [ -z "$WORKDIR" ]; then
  echo "Usage: $0 <workdir>" >&2
  echo "" >&2
  echo "  Start the PostgreSQL container for the tuning infrastructure." >&2
  echo "  Container name and image are read from config.rc." >&2
  echo "  Fails if services are already running (stale PID file)." >&2
  exit 1
fi

load_config "$WORKDIR"

# Container names with suffix
POSTGRES_CONTAINER="aotriton_pgsql.${CONTAINER_SUFFIX}"

# PID file
mkdir -p "$WORKDIR/run"
PIDF="$WORKDIR/run/container.pids"

if [ -f "$PIDF" ]; then
  echo "Error: Services already running or stale PID file exists. Run stop first." >&2
  exit 1
fi

echo "Starting server services..."

# Pull PostgreSQL image if not present
echo "Pulling PostgreSQL image: ${POSTGRES_DOCKER_IMAGE}"
docker pull "${POSTGRES_DOCKER_IMAGE}"

# Start PostgreSQL
echo "Starting PostgreSQL on port ${POSTGRES_PORT}..."
POSTGRES_ID=$(docker run --ipc=host \
  --network=host \
  -d \
  --rm \
  --ulimit nofile=65536:65536 \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e PGPORT="${POSTGRES_PORT}" \
  -v "${POSTGRES_DOCKER_VOLUME}:/var/lib/postgresql/data" \
  --name "${POSTGRES_CONTAINER}" \
  "${POSTGRES_DOCKER_IMAGE}" \
  postgres -c max_connections=500 -c shared_buffers=2GB)

if [ -z "$POSTGRES_ID" ]; then
  echo "Error: Failed to start PostgreSQL" >&2
  rm -f "$PIDF"
  exit 1
fi
echo "$POSTGRES_ID" >> "$PIDF"
echo "Started PostgreSQL: $POSTGRES_ID"

echo "Server services started successfully"
