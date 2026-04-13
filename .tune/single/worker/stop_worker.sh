#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Stop worker on one host
# Usage: stop_worker.sh <hostname> <arch> <workdir>

set -e

HOSTNAME="$1"
ARCH="$2"
WORKER_WORKDIR="$3"

CONTAINER_NAME="aotriton_celeryworker.${ARCH}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$CONTAINER_NAME" <<'EOF'
WORKER_WORKDIR="$1"
CONTAINER_NAME="$2"

docker exec "$CONTAINER_NAME" bash /wkdir/aotriton.src/.tune/single/worker/celery_service.sh stop /wkdir
docker stop "$CONTAINER_NAME"
EOF
