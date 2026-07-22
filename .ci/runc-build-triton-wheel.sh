#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build a Triton wheel from a fresh git checkout (never a submodule gitlink,
# so Triton's setup.py can auto-detect "+git<hash8>"). Runs inside a
# container -- .ci/build_triton_wheels.sh's ephemeral one, or .tune's
# worker container as its no-Docker fallback.
#
# Usage: runc-build-triton-wheel.sh <triton_source> <hash> <output_dir> [<version_suffix>] [<scratch_dir>]

set -ex

TRITON_SOURCE="$1"
HASH="$2"
OUTPUT_DIR="$3"
VERSION_SUFFIX="${4:-}"
SCRATCH_DIR="${5:-}"

if [ -z "$TRITON_SOURCE" ] || [ -z "$HASH" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <triton_source> <hash> <output_dir> [<version_suffix>] [<scratch_dir>]" >&2
  exit 1
fi

if [ -z "$SCRATCH_DIR" ]; then
  SCRATCH_DIR="$(mktemp -d)"
fi

git config --global --add safe.directory '*'

SHORT="${HASH:0:8}"
rm -rf "$SCRATCH_DIR"
git init "$SCRATCH_DIR"
git -C "$SCRATCH_DIR" remote add origin "$TRITON_SOURCE"
git -C "$SCRATCH_DIR" fetch --depth=1 origin "$HASH"
git -C "$SCRATCH_DIR" checkout FETCH_HEAD

mkdir -p "$OUTPUT_DIR"
TRITON_WHEEL_VERSION_SUFFIX="$VERSION_SUFFIX" python -m pip wheel "$SCRATCH_DIR" -w "$OUTPUT_DIR"

ls "$OUTPUT_DIR"/triton-*+*"${SHORT}"*.whl
