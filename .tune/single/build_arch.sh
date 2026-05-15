#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build AOTriton libraries for one architecture (local execution)
# Usage: build_arch.sh <workdir> <arch> <triton_wheel>

set -e

WORKDIR="$1"
ARCH="$2"
TRITON_WHEEL="$3"

if [ -z "$WORKDIR" ] || [ -z "$ARCH" ] || [ -z "$TRITON_WHEEL" ]; then
  echo "Usage: $0 <workdir> <arch> <triton_wheel>" >&2
  echo "" >&2
  echo "  Build AOTriton libraries for <arch> locally using cmake+ninja." >&2
  echo "  Output is installed into <workdir>/installed/<arch>/." >&2
  echo "  Run build_triton_wheel.sh first to obtain the <triton_wheel> path." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AOTRITON_ROOT="$(realpath "${TUNE_ROOT}/..")"

BUILD_DIR="$WORKDIR/build/$ARCH"
INSTALL_DIR="$WORKDIR/installed/$ARCH"

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

cd "$BUILD_DIR"

cmake "$AOTRITON_ROOT" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DAOTRITON_TARGET_ARCH="$ARCH" \
    -DAOTRITON_NAME_SUFFIX=123 \
    -DAOTRITON_BUILD_FOR_TUNING=ON \
    -DAOTRITON_USE_LOCAL_TRITON_WHEEL="$TRITON_WHEEL" \
    -G Ninja

ninja install/strip
