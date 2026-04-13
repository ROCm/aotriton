#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Docker image on one host
# Usage: build_image.sh <hostname> <workdir> <image_name>

set -e

HOSTNAME="$1"
WORKER_WORKDIR="$2"
CELERY_WORKER_IMAGE="$3"

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$CELERY_WORKER_IMAGE" <<'EOF'
WORKER_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"

cd "$WORKER_WORKDIR"
tsp docker build -f Dockerfile -t "$CELERY_WORKER_IMAGE" .
EOF
