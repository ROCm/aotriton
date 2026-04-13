#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start RabbitMQ container
# Usage: start_rabbitmq.sh <workdir>

set -e

WORKDIR="$1"
. "$WORKDIR/config.rc"

docker run --ipc=host --network=host -d --rm \
    --ulimit nofile=65536:65536 \
    --hostname "$CELERY_SERVICE_HOST" \
    -e RABBITMQ_DEFAULT_USER="$RABBITMQ_DEFAULT_USER" \
    -e RABBITMQ_DEFAULT_PASS="$RABBITMQ_DEFAULT_PASS" \
    -e RABBITMQ_VM_MEMORY_HIGH_WATERMARK=2048MiB \
    -e RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS="+P 1048576" \
    --name "$RABBITMQ_CONTAINER" \
    "$RABBITMQ_DOCKER_IMAGE"

echo "RabbitMQ started"
