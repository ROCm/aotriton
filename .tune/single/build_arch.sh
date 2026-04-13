#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build AOTriton libraries for one architecture
# Usage: build_arch.sh <arch> <workdir>

set -e

ARCH="$1"
WORKDIR="$2"

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
