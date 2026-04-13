#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start PostgreSQL container
# Usage: start_postgres.sh <workdir>

set -e

WORKDIR="$1"
. "$WORKDIR/config.rc"

docker run --ipc=host --network=host -d --rm \
    --ulimit nofile=65536:65536 \
    -e POSTGRES_USER="$POSTGRES_USER" \
    -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
    -v "${POSTGRES_DOCKER_VOLUME}:/var/lib/postgresql/data" \
    --name "$POSTGRES_CONTAINER" \
    "$POSTGRES_DOCKER_IMAGE" \
    postgres -c max_connections=500 -c shared_buffers=2GB

echo "PostgreSQL started"
