#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build a single Triton wheel from a fresh git checkout. Runs inside a
# container: either the ephemeral aotriton:base(-pyX.Y) container spun up by
# .ci/build_triton_wheels.sh (bind-mounted in), or .tune's long-lived worker
# container via .tune/single/build_triton_wheel.sh (its no-Docker-available
# fallback path). No docker/mirror-volume specifics live here — those stay
# in the callers.
#
# Cloning into a FRESH real git checkout (never a submodule gitlink) lets
# Triton's own setup.py auto-detect and contribute "+git<hash8>" to the
# wheel's local version identifier.
#
# Usage: runc-build-triton-wheel.sh <triton_source> <hash> <output_dir> [<version_suffix>] [<scratch_dir>]
#   <triton_source>   git-fetchable remote (e.g. file:///mirror, a local
#                     mirror path, or a plain URL)
#   <hash>            commit to check out
#   <output_dir>      directory to write the built wheel into
#   <version_suffix>  optional TRITON_WHEEL_VERSION_SUFFIX value
#   <scratch_dir>     optional clone/build scratch dir (default: mktemp -d)

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
