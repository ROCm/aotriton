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
# pin_line first -- find_flydsl.sh's helpers are written against it.
# shellcheck disable=SC1091
. "$AOTRITON_ROOT/.ci/common-pin.sh"
# shellcheck disable=SC1091
. "$TUNE_ROOT/lib/find_flydsl.sh"
ALTWHEEL_CONFIG="$(get_resolved_altwheel_yaml "$AOTRITON_ROOT" "$WORKDIR")"
BUILD_TUNE_ARGS=()
[ -n "$ALTWHEEL_CONFIG" ] && BUILD_TUNE_ARGS+=(--altwheel_config "$ALTWHEEL_CONFIG")

# The FlyDSL wheel, when this tree needs a locally built one. Resolved here
# rather than threaded in as a fourth positional, so libbld's call site is
# unchanged -- it already loops over arches handing us the one Triton wheel,
# and the FlyDSL wheel is arch-independent in exactly the same way.
#
# Note this is required even though flyc kernels are untunable today: the
# tripwire at CMakeLists.txt:206 fires on any image-mode configure, so a tuning
# build needs the wheel just to reach codegen.
if flydsl_required "$AOTRITON_ROOT"; then
  PYVER=$(python --version 2>&1 | cut -d' ' -f2 | cut -d. -f1,2)
  if ! FLYDSL_WHEEL=$(find_flydsl_wheel "$AOTRITON_ROOT" "$WORKDIR" "$PYVER"); then
    echo "Error: third_party/flydsl-llvm.txt is non-empty, so this build needs a" >&2
    echo "locally built FlyDSL wheel, but none for python ${PYVER} is cached in" >&2
    echo "$WORKDIR/scratch/flydsl/." >&2
    echo "" >&2
    echo "Without it cmake fails at configure (CMakeLists.txt's flydsl-llvm.txt" >&2
    echo "tripwire). Run remotebld, which pre-builds it via a sibling container," >&2
    echo "or build it directly with:" >&2
    echo "  .tune/single/prebuild_wheel.sh --select all <workdir>" >&2
    exit 1
  fi
  echo "Using FlyDSL wheel: $FLYDSL_WHEEL"
  BUILD_TUNE_ARGS+=(--flydsl_wheel "$FLYDSL_WHEEL")
fi

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
