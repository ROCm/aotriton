#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start RabbitMQ container
# Usage: start_rabbitmq.sh <workdir>

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
RABBITMQ_CONTAINER="aotriton_rabbitmq.${CONTAINER_SUFFIX}"

# PID file
mkdir -p "$WORKDIR/run"
PIDF="$WORKDIR/run/container.pids"

echo "Starting RabbitMQ..."
RABBITMQ_ID=$(docker run --ipc=host \
  --network=host \
  -d \
  --rm \
  --ulimit nofile=65536:65536 \
  -e RABBITMQ_DEFAULT_USER="${RABBITMQ_DEFAULT_USER}" \
  -e RABBITMQ_DEFAULT_PASS="${RABBITMQ_DEFAULT_PASS}" \
  --name "${RABBITMQ_CONTAINER}" \
  rabbitmq:4-management)

if [ -z "$RABBITMQ_ID" ]; then
  echo "Error: Failed to start RabbitMQ" >&2
  exit 1
fi
echo "$RABBITMQ_ID" >> "$PIDF"
echo "Started RabbitMQ: $RABBITMQ_ID"
