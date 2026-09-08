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
#
# PYTORCH_ROCM_ARCH is torch's, not AOTriton's: CMakeLists.txt's
# find_package(Torch) pulls in LoadHIP.cmake, which hard-errors with "No GPU
# arch specified for ROCm build" unless it is set. Torch would otherwise probe
# the local GPU, and probing is wrong here twice over -- the build container
# need not have a GPU at all, and libbld builds every arch in the registry from
# one host, so even a host that has one would answer for the wrong target on
# all but one pass through its loop. $ARCH is the arch actually being built,
# which is the only correct answer. Set per invocation rather than exported, so
# each pass gets its own.
AOTRITON_BUILD_PATH="$BUILD_DIR" \
AOTRITON_INSTALL_PATH="$INSTALL_DIR" \
PYTORCH_ROCM_ARCH="$ARCH" \
  bash "$AOTRITON_ROOT/.ci/build-tune.sh" "${BUILD_TUNE_ARGS[@]}" "$ARCH" "$TRITON_WHEEL"
