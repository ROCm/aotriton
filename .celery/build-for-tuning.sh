#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
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

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
AOTRITON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="$1"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Setup
TRITON_DIR="$WORKDIR/scratch/triton"
mkdir -p "$TRITON_DIR"

# Check if triton wheel already exists for current version and python
has_triton_wheel() {
  local triton_dir="$1"
  local triton_src="$2"

  # Get current triton version and python version
  local triton_version=$(cd "$triton_src" && git describe --tags --always 2>/dev/null || echo "unknown")
  local python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

  # Check if matching wheel exists
  local wheel=$(ls "$triton_dir"/triton-*cp${python_version/./}*.whl 2>/dev/null | head -n 1)

  if [ -f "$wheel" ]; then
    # Check if wheel name contains the triton version
    if [[ "$wheel" == *"$triton_version"* ]] || [[ "$triton_version" == "unknown" ]]; then
      echo "$wheel"
      return 0
    fi
  fi

  return 1
}

# Step 1: Build triton wheel
if TRITON_WHEEL=$(has_triton_wheel "$TRITON_DIR" "$AOTRITON_ROOT/third_party/triton"); then
  echo "Using existing triton wheel: $TRITON_WHEEL"
else
  cd "$AOTRITON_ROOT/third_party/triton/python"
  pip wheel . -w "$TRITON_DIR"

  TRITON_WHEEL=$(ls "$TRITON_DIR"/triton-*.whl | head -n 1)
  if [ ! -f "$TRITON_WHEEL" ]; then
    echo "Error: Triton wheel not found" >&2
    exit 1
  fi
fi

# Step 2: Build AOTriton for each architecture
. "$AOTRITON_ROOT/.ci/common-vars.sh"

ARCHS=($(sqlite3 "$WORKDIR/workers.db" "SELECT DISTINCT arch FROM workers ORDER BY arch;"))

for arch in "${ARCHS[@]}"; do
  BUILD_DIR="$WORKDIR/build/$arch"
  mkdir -p "$BUILD_DIR"

  (
    cd "$BUILD_DIR"
    cmake "$AOTRITON_ROOT" \
      -DCMAKE_INSTALL_PREFIX=./install_dir \
      -DCMAKE_BUILD_TYPE=Release \
      -DAOTRITON_TARGET_ARCH="$arch" \
      -DAOTRITON_NAME_SUFFIX=123 \
      -DAOTRITON_BUILD_FOR_TUNING=ON \
      -DAOTRITON_USE_LOCAL_TRITON_WHEEL="$TRITON_WHEEL" \
      -G Ninja
    ninja install/strip
  ) || exit 1
done
