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
# -DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE.
if [ -n "$ALTWHEEL_YAML" ]; then
  if ! command -v yq &> /dev/null; then
    echo "Error: 'yq' is required to resolve altwheel YAML $ALTWHEEL_YAML (dnf install yq / snap install yq)" >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  . "$AOTRITON_ROOT/.ci/common-altwheel.sh"

  RESOLVED_YAML="$WORKDIR/scratch/triton/resolved_altwheel.yaml"
  cp "$ALTWHEEL_YAML" "$RESOLVED_YAML"

  # AOTriton's own codegen (python/codegen/root.py's _load_altwheel_config)
  # requires every venvs.* value to be a plain string (a wheel path, or a
  # "python:X.Y" marker) -- it calls value.startswith(...) unconditionally,
  # so a {hash, origin} map (our source syntax for a non-default origin)
  # would crash it. Rebuild every entry -- including "default", added here
  # if the source yaml didn't set one -- as the actual wheel path, whatever
  # its source syntax was.
  resolve_wheel_for_hash() {
    local hash="$1" short
    short="${hash:0:8}"
    ls "$WORKDIR/scratch/triton"/triton-*+git${short}*.whl 2>/dev/null | head -n1
  }

  default_hash=$(altwheel_venv_hash "$RESOLVED_YAML" ".venvs.default")
  [ -z "$default_hash" ] && default_hash="${HASHES[0]}"
  default_wheel=$(resolve_wheel_for_hash "$default_hash")
  if [ -z "$default_wheel" ]; then
    echo "Error: no built wheel found for altwheel default venv (hash ${default_hash:0:8})" >&2
    exit 1
  fi
  yq -i ".venvs.default = \"${default_wheel}\"" "$RESOLVED_YAML"

  for key in $(yq -r '.venvs | keys | .[]' "$RESOLVED_YAML"); do
    [ "$key" = "default" ] && continue
    hash=$(altwheel_venv_hash "$RESOLVED_YAML" ".venvs.${key}")
    wheel=$(resolve_wheel_for_hash "$hash")
    if [ -z "$wheel" ]; then
      echo "Error: no built wheel found for altwheel venv '$key' (hash ${hash:0:8})" >&2
      exit 1
    fi
    yq -i ".venvs.${key} = \"${wheel}\"" "$RESOLVED_YAML"
  done

  echo "Resolved altwheel config: $RESOLVED_YAML"
fi
