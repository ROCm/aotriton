#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Docker image on one host
# Usage: build_image.sh <workdir> <hostname> [--arch <arch>] [--follow]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2 || true

FOLLOW=""
ARCH=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --follow) FOLLOW="true"; shift ;;
    --arch)   ARCH="$2"; shift 2 ;;
    *)        echo "Error: unrecognized argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [--arch <arch>] [--follow]" >&2
  echo "" >&2
  echo "  Submit a Docker image build job via tsp on <hostname>." >&2
  echo "  --arch    GPU arch the image's torch must target (default: this" >&2
  echo "            host's arch in the worker registry)." >&2
  echo "  --follow  Tail the build output in real-time (blocks until done)." >&2
  echo "  Without --follow, the job runs in background; check with tsp on the host." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get workdir_override for this hostname. Check the worker-registry table
# first (a registered tuning worker acting as its own build target); a
# build node is explicitly NOT required to be a registered worker (see
# remotebld/get_buildnode_workdir), so fall back to the separate
# buildnode::* config when the hostname matches the configured build node
# instead -- otherwise this would silently resolve to DEFAULT_WORKDIR, which
# may not even exist on that host.
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

if [ -z "$workdir_override" ]; then
  BUILDNODE_HOSTNAME=$(sqlite3 "$WORKDIR/workers.db" \
    "SELECT COALESCE(value,'') FROM config WHERE key='buildnode::hostname'" 2>/dev/null || true)
  if [ -n "$BUILDNODE_HOSTNAME" ] && [ "$HOSTNAME" = "$BUILDNODE_HOSTNAME" ]; then
    workdir_override="$(get_buildnode_workdir "$WORKDIR")"
  fi
fi

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# The arch the image's torch is built for. An explicit --arch (imgbld passes
# the one it already read for this host) wins; otherwise use this host's own
# registry row, which the lookup above already returned.
#
# This is resolved HERE, on the server, and shipped to the remote as a docker
# build-arg, rather than probed inside the image build: a worker is not
# guaranteed to have amd-smi or rocminfo available -- they may be missing
# entirely, or installed into a TheRock venv at a location this script has no
# way to guess -- whereas the registry knows the arch of every worker by
# construction. A build node that is not a registered worker has no arch row;
# that case passes nothing and lets the Dockerfile's own ARG default stand.
ARCH="${ARCH:-$arch}"
BUILD_ARGS=""
if [ -n "$ARCH" ]; then
  BUILD_ARGS="--build-arg ROCM_GPU_ARCH=$ARCH"
  echo "Building for GPU arch: $ARCH"
fi

# Certain nodes need --network=host to access internet
if [ -n "$FOLLOW" ]; then
  # Use tsp -t to tail/follow output in real-time
  # ARCH goes last on purpose: when it is empty ssh drops the trailing empty
  # argument entirely, and ${3:-} below absorbs that without a sentinel.
  ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$CELERY_WORKER_IMAGE" "$ARCH" <<'EOF'
WORKER_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"
ARCH="${3:-}"

BUILD_ARGS=""
if [ -n "$ARCH" ]; then
  BUILD_ARGS="--build-arg ROCM_GPU_ARCH=$ARCH"
fi

jobid=$(tsp docker build --network=host $BUILD_ARGS -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR)
echo "Job ID: $jobid"
if [ "$(tsp -s "$jobid")" = "queued" ]; then
  echo "Waiting for tsp job $jobid to start..."
  while [ "$(tsp -s "$jobid")" = "queued" ]; do sleep 5; done
fi
tsp -t $jobid
EOF
else
  ssh -n "$HOSTNAME" "tsp docker build --network=host $BUILD_ARGS -f $WORKER_WORKDIR/image.build/Dockerfile -t $CELERY_WORKER_IMAGE $WORKER_WORKDIR"
fi
