#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Stop RabbitMQ and PostgreSQL
# Usage: stop_services.sh <workdir>

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

# Container names with suffix
RABBITMQ_CONTAINER="aotriton_rabbitmq.${CONTAINER_SUFFIX}"
POSTGRES_CONTAINER="aotriton_pgsql.${CONTAINER_SUFFIX}"

# PID file
PIDF="$WORKDIR/run/container.pids"

echo "Stopping server services..."

if [ -f "$PIDF" ]; then
  CONTAINER_IDS=$(cat "$PIDF")
  if [ -n "$CONTAINER_IDS" ]; then
    echo "Stopping containers: $CONTAINER_IDS"
    docker stop $CONTAINER_IDS
    rm -f "$PIDF"
    echo "Server services stopped and containers removed"
  else
    echo "PID file is empty"
    rm -f "$PIDF"
  fi
else
  # Try to stop by container name
  echo "No PID file found. Attempting to stop by container name..."
  docker stop "${RABBITMQ_CONTAINER}" "${POSTGRES_CONTAINER}" 2>/dev/null || true
  docker rm "${RABBITMQ_CONTAINER}" "${POSTGRES_CONTAINER}" 2>/dev/null || true
  echo "Server services stopped (if they were running)"
fi
