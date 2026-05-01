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
  echo "  Build the Triton wheel from third_party/triton and cache it in <workdir>/scratch/triton/." >&2
  echo "  Skips rebuild if a wheel matching the current git revision already exists." >&2
  echo "  Prints the wheel path to stdout on success (suitable for \$(…) capture)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AOTRITON_ROOT="$(realpath "${TUNE_ROOT}/..")"

# Setup
TRITON_SRC_DIR="$AOTRITON_ROOT/third_party/triton"
TRITON_WHEEL_OUTPUT_DIR="$WORKDIR/scratch/triton"
mkdir -p "$TRITON_WHEEL_OUTPUT_DIR"

echo "Sync triton source..." >&2
(cd "$AOTRITON_ROOT"; git submodule sync && git submodule update --init --recursive --force) >&2
set -e # MUST NOT FAIL
TRITON_GIT12=$(cd "$TRITON_SRC_DIR" && git rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")
set +e
TRITON_WHEEL_VERSION_SUFFIX="+tunerwheel.$TRITON_GIT12"

# Check if triton wheel already exists for current version and python
has_triton_wheel() {
  local triton_dir="$1"
  local triton_signature="$2"

  # Get current triton version and python version
  local python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

  # Check if matching wheel exists
  local wheel=$(ls "$triton_dir"/triton-*cp${python_version/./}*.whl 2>/dev/null | head -n 1)
  echo "$wheel" >&2

  if [ -f "$wheel" ]; then
    # Check if wheel name contains the triton version
    if [[ "$wheel" == *"$triton_signature"* ]]; then
      echo "$wheel"
      return 0
    fi
  fi

  return 1
}

TRITON_WHEEL=$(has_triton_wheel "$TRITON_WHEEL_OUTPUT_DIR" "${TRITON_WHEEL_VERSION_SUFFIX}")
echo "TRITON_WHEEL detected: $TRITON_WHEEL" >&2

# Build triton wheel if not found
if [ -z "$TRITON_WHEEL" ]; then

  echo "Building triton wheel..." >&2
  # Must set TRITON_WHEEL_VERSION_SUFFIX triton's setup.py use .is_dir() to
  # detect .git and thus cannot append +git<hash8> when being built as a submodule.
  (cd "$TRITON_SRC_DIR"; TRITON_WHEEL_VERSION_SUFFIX=${TRITON_WHEEL_VERSION_SUFFIX} pip wheel . -w "$TRITON_WHEEL_OUTPUT_DIR")

  TRITON_WHEEL=$(ls "$TRITON_WHEEL_OUTPUT_DIR"/triton-*.whl 2>/dev/null | head -n 1)
  if [ -z "$TRITON_WHEEL" ] || [ ! -f "$TRITON_WHEEL" ]; then
    echo "Error: Triton wheel not found" >&2
    exit 1
  fi
fi

# Output the wheel path
echo "$TRITON_WHEEL"
