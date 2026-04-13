#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start worker on one host
# Usage: start_worker.sh <hostname> <arch> <workdir> <image>

set -e

HOSTNAME="$1"
ARCH="$2"
WORKER_WORKDIR="$3"
CELERY_WORKER_IMAGE="$4"

CONTAINER_NAME="aotriton_celeryworker.${ARCH}"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$CELERY_WORKER_IMAGE" "$CONTAINER_NAME" "$ARCH" <<'EOF'
WORKER_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"
CONTAINER_NAME="$3"
ARCH="$4"

# Load config
. "$WORKER_WORKDIR/config.rc"

# Start container
docker run -d --rm \
  --network=host \
  --name "$CONTAINER_NAME" \
  --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -e AOTRITON_ARCH="$ARCH" \
  -v "$WORKER_WORKDIR:/wkdir" \
  "$CELERY_WORKER_IMAGE"
EOF
