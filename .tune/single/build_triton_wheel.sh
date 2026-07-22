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
# Gitlink SHA read -- works without third_party/triton checked out locally.
TRITON_HASH=$(git -C "$AOTRITON_ROOT" rev-parse HEAD:third_party/triton)
TRITON_GIT8="${TRITON_HASH:0:8}"
set +e

# Glob matches the +git<hash8> naming .ci/runc-build-triton-wheel.sh produces.
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

# Fallback build path (remotebld normally pre-builds via a sibling container).
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
