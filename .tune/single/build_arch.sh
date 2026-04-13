#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build AOTriton libraries for one architecture (local execution)
# Usage: build_arch.sh <workdir> <arch>

set -e

WORKDIR="$1"
ARCH="$2"

if [ -z "$WORKDIR" ] || [ -z "$ARCH" ]; then
  echo "Usage: $0 <workdir> <arch>" >&2
  exit 1
fi

BUILD_DIR="$WORKDIR/build/$ARCH"
INSTALL_DIR="$WORKDIR/installed/$ARCH"

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

cd "$BUILD_DIR"

cmake "$WORKDIR/aotriton.src" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DAOTRITON_ENABLE_CKDB=ON \
    -DCMAKE_HIP_ARCHITECTURES="$ARCH"

ninja install
