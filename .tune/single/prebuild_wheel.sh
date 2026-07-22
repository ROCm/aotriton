#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-build the Triton wheel(s) for this workdir into
# <workdir>/scratch/triton/, via .ci/build_triton_wheels.sh. Needs Docker;
# called by remotebld as a sibling step before testbld/libbld (no
# Docker-in-Docker), which then just hits this cache.
#
# Usage: prebuild_wheel.sh <workdir>

set -euo pipefail

WORKDIR="$1"
if [ -z "$WORKDIR" ]; then
  echo "Usage: $0 <workdir>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AOTRITON_ROOT="$(realpath "$TUNE_ROOT/..")"

# shellcheck disable=SC1091
source "$WORKDIR/config.rc"

if [ -z "${CELERY_WORKER_PYTHON:-}" ] || [ -z "${CELERY_WORKER_IMAGE:-}" ]; then
  echo "Error: CELERY_WORKER_PYTHON/CELERY_WORKER_IMAGE not set in config.rc" >&2
  exit 1
fi

# CELERY_WORKER_PYTHON only exists inside CELERY_WORKER_IMAGE, not on the
# host -- run it in a throwaway container instead of invoking it directly.
PYVER=$(docker run --rm "$CELERY_WORKER_IMAGE" "$CELERY_WORKER_PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)

aotriton_major=$(grep '^set(AOTRITON_VERSION_MAJOR_INT' "$AOTRITON_ROOT/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
aotriton_minor=$(grep '^set(AOTRITON_VERSION_MINOR_INT' "$AOTRITON_ROOT/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)

# shellcheck disable=SC1091
. "$TUNE_ROOT/lib/aotriton_version.sh"
version_dir="$(get_aotriton_version_dir "$AOTRITON_ROOT")"

# Reuse .ci's per-version altwheel convention if present.
ALTWHEEL_YAML=""
if [ -f "$AOTRITON_ROOT/.ci/${version_dir}.yaml" ]; then
  ALTWHEEL_YAML="$AOTRITON_ROOT/.ci/${version_dir}.yaml"
fi

mapfile -t HASHES < <(bash "$AOTRITON_ROOT/.ci/resolve-triton-hashes.sh" "$AOTRITON_ROOT" "$ALTWHEEL_YAML")

echo "Pre-building Triton wheel(s) for python ${PYVER}: ${HASHES[*]}"

bash "$AOTRITON_ROOT/.ci/build_triton_wheels.sh" \
  --wheel_output_dir "$WORKDIR/scratch/triton" \
  --version_suffix "+aotriton${aotriton_major}.${aotriton_minor}" \
  --python "$PYVER" \
  --altwheel_yaml "$ALTWHEEL_YAML" \
  "${HASHES[@]}"

# Resolve hashes -> wheel paths for build_arch.sh/testbld to pass as
# -DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE. host/container dir are the same
# path here (no docker translation, unlike .ci's release pipeline).
if [ -n "$ALTWHEEL_YAML" ]; then
  if ! command -v yq &> /dev/null; then
    echo "Error: 'yq' is required to resolve altwheel YAML $ALTWHEEL_YAML (dnf install yq / snap install yq)" >&2
    exit 1
  fi

  RESOLVED_YAML="$WORKDIR/scratch/triton/resolved_altwheel.yaml"
  cp "$ALTWHEEL_YAML" "$RESOLVED_YAML"
  if [ -z "$(yq -r '.venvs.default // ""' "$RESOLVED_YAML")" ]; then
    yq -i ".venvs.default = \"${HASHES[0]}\"" "$RESOLVED_YAML"
  fi

  # shellcheck disable=SC1091
  . "$AOTRITON_ROOT/.ci/include-altwheel.sh"
  replace_hash "$RESOLVED_YAML" "$WORKDIR/scratch/triton" "$WORKDIR/scratch/triton" "${HASHES[@]}"

  echo "Resolved altwheel config: $RESOLVED_YAML"
fi
