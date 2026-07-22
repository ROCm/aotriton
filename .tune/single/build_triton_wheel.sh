#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Triton wheel (cached by version)
# Usage: build_triton_wheel.sh <workdir>

set -ex

WORKDIR="$1"

if [ -z "$WORKDIR" ]; then
  echo "Usage: $0 <workdir>" >&2
  echo "" >&2
  echo "  Build the Triton wheel pinned by third_party/triton and cache it in <workdir>/scratch/triton/." >&2
  echo "  Skips rebuild if a wheel matching the current git revision already exists." >&2
  echo "  Prints the wheel path to stdout on success (suitable for \$(…) capture)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AOTRITON_ROOT="$(realpath "${TUNE_ROOT}/..")"

# Setup
TRITON_WHEEL_OUTPUT_DIR="$WORKDIR/scratch/triton"
mkdir -p "$TRITON_WHEEL_OUTPUT_DIR"

set -e # MUST NOT FAIL
# Reads the gitlink SHA straight from the git tree object -- works whether or
# not third_party/triton is actually checked out locally. This script does
# not assume it is (matching .ci, which never checks it out either): the
# build path below clones a fresh copy directly instead of relying on any
# local submodule state.
TRITON_HASH=$(git -C "$AOTRITON_ROOT" rev-parse HEAD:third_party/triton)
TRITON_GIT8="${TRITON_HASH:0:8}"
set +e

# Check if a triton wheel already exists for the current python + this
# commit. Glob matches the canonical +git<hash8> scheme that
# .ci/runc-build-triton-wheel.sh produces (see docs/plans on wheel naming
# unification) -- the same scheme a sibling wheel-build container launched
# by .tune/bin/remotebld would already have populated on the common path.
has_triton_wheel() {
  local triton_dir="$1"
  local hash8="$2"

  # Get current python version
  local python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

  local wheel=$(ls "$triton_dir"/triton-*cp${python_version/./}*+git${hash8}*.whl 2>/dev/null | head -n 1)
  echo "$wheel" >&2

  if [ -f "$wheel" ]; then
    echo "$wheel"
    return 0
  fi

  return 1
}

TRITON_WHEEL=$(has_triton_wheel "$TRITON_WHEEL_OUTPUT_DIR" "$TRITON_GIT8")
echo "TRITON_WHEEL detected: $TRITON_WHEEL" >&2

# Build triton wheel if not found. This is the no-Docker-available fallback
# path: on the common path (builds launched via remotebld), a sibling
# container already pre-built and cached the wheel here before this script
# ever runs; this only builds when that didn't happen (e.g. bare-metal
# libbld usage that bypasses remotebld entirely).
if [ -z "$TRITON_WHEEL" ]; then
  echo "Building triton wheel (no pre-built wheel found in cache)..." >&2

  TRITON_REMOTE=$(git config -f "$AOTRITON_ROOT/.gitmodules" --get submodule.third_party/triton.url)

  bash "$AOTRITON_ROOT/.ci/runc-build-triton-wheel.sh" \
    "$TRITON_REMOTE" "$TRITON_HASH" "$TRITON_WHEEL_OUTPUT_DIR" \
    "" "$WORKDIR/scratch/triton-build"

  TRITON_WHEEL=$(ls "$TRITON_WHEEL_OUTPUT_DIR"/triton-*+git${TRITON_GIT8}*.whl 2>/dev/null | head -n 1)
  if [ -z "$TRITON_WHEEL" ] || [ ! -f "$TRITON_WHEEL" ]; then
    echo "Error: Triton wheel not found" >&2
    exit 1
  fi
fi

# Output the wheel path
echo "$TRITON_WHEEL"
