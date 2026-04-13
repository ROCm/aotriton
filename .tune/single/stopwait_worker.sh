#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Graceful shutdown worker on one host
# Usage: stopwait_worker.sh <hostname> <arch> <workdir>

set -e

HOSTNAME="$1"
ARCH="$2"
WORKER_WORKDIR="$3"

CONTAINER_NAME="aotriton_celeryworker.${ARCH}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$CONTAINER_NAME" <<'EOF'
WORKER_WORKDIR="$1"
CONTAINER_NAME="$2"

docker exec "$CONTAINER_NAME" bash /wkdir/aotriton.src/.tune/remote/worker_service.sh stopwait /wkdir
docker stop "$CONTAINER_NAME"
EOF
