#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if ! command -v sqlite3 &> /dev/null; then
  cat <<EOF >&2
Command 'sqlite3' could not be found. Install it with
dnf install sqlite3
or
snap install sqlite3
EOF
fi

if [ "$#" -ne 1 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir>

Build triton wheel and AOTriton tuning binaries for all registered
architectures by manage-workers.py.

Arguments:
  <workdir>  Project working directory (created by create-project-directory.sh)

Output:
  - Triton wheel: <workdir>/scratch/triton/
  - AOTriton builds: <workdir>/build/<arch>/
EOF
  exit 1
fi

# set -x

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
AOTRITON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="$1"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Setup
TRITON_SRC_DIR="$AOTRITON_ROOT/third_party/triton"
TRITON_WHEEL_OUTPUT_DIR="$WORKDIR/scratch/triton"
mkdir -p "$TRITON_WHEEL_OUTPUT_DIR"
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
  echo $wheel >&2

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
echo "TRITON_WHEEL detected: $TRITON_WHEEL"

# Step 1: Build triton wheel
if [ -z "$TRITON_WHEEL" ]; then
  echo "Building triton wheel: $TRITON_WHEEL"
  # Must set TRITON_WHEEL_VERSION_SUFFIX triton's setup.py use .is_dir() to
  # detect .git and thus cannot append +git<hash8> when being built as a submodule.
  (cd "$TRITON_SRC_DIR"; TRITON_WHEEL_VERSION_SUFFIX=${TRITON_WHEEL_VERSION_SUFFIX} pip wheel . -w "$TRITON_WHEEL_OUTPUT_DIR")

  TRITON_WHEEL=$(ls "$TRITON_WHEEL_OUTPUT_DIR"/triton-*.whl 2>/dev/null | head -n 1)
  if [ -z "$TRITON_WHEEL" ] || [ ! -f "$TRITON_WHEEL" ]; then
    echo "Error: Triton wheel not found" >&2
    exit 1
  fi
fi

# Step 2: Build AOTriton for each architecture
. "$AOTRITON_ROOT/.ci/common-vars.sh"

ARCHS=($(sqlite3 "$WORKDIR/workers.db" "SELECT DISTINCT arch FROM workers ORDER BY arch;"))
echo "ARCHS detected: ${ARCHS[@]}"

ABSWORKDIR=$(realpath "$WORKDIR")

for arch in "${ARCHS[@]}"; do
  BUILD_DIR="$ABSWORKDIR/build/$arch"
  INSTALL_DIR="$ABSWORKDIR/installed/$arch"
  mkdir -p "$BUILD_DIR"

  (
    cd "$BUILD_DIR"
    cmake "$AOTRITON_ROOT" \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DAOTRITON_TARGET_ARCH="$arch" \
      -DAOTRITON_NAME_SUFFIX=123 \
      -DAOTRITON_BUILD_FOR_TUNING=ON \
      -DAOTRITON_USE_LOCAL_TRITON_WHEEL="$TRITON_WHEEL" \
      -G Ninja
    ninja install/strip
  ) || exit 1
done
