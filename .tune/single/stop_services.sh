#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Stop RabbitMQ and PostgreSQL
# Usage: stop_services.sh <workdir>

set -e

WORKDIR="$1"
. "$WORKDIR/config.rc"

docker stop "$RABBITMQ_CONTAINER" 2>/dev/null || true
docker stop "$POSTGRES_CONTAINER" 2>/dev/null || true

echo "Services stopped"
