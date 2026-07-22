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

# shellcheck disable=SC1091
. "$TUNE_ROOT/lib/aotriton_version.sh"
ALTWHEEL_CONFIG="$(get_resolved_altwheel_yaml "$AOTRITON_ROOT" "$WORKDIR")"
BUILD_TUNE_ARGS=()
[ -n "$ALTWHEEL_CONFIG" ] && BUILD_TUNE_ARGS+=(--altwheel_config "$ALTWHEEL_CONFIG")

# Delegate to .ci/build-tune.sh, pointed at this workdir's build/install dirs.
AOTRITON_BUILD_PATH="$BUILD_DIR" \
AOTRITON_INSTALL_PATH="$INSTALL_DIR" \
  bash "$AOTRITON_ROOT/.ci/build-tune.sh" "${BUILD_TUNE_ARGS[@]}" "$ARCH" "$TRITON_WHEEL"
