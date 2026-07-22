#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-build the Triton wheel(s) needed for this workdir into
# <workdir>/scratch/triton/, via .ci/build_triton_wheels.sh.
#
# This spins up its own ephemeral aotriton:base-pyX.Y container(s) and must
# therefore run on a host with Docker access -- called by .tune/bin/remotebld
# as a sibling step *before* launching the testbld/libbld container, never
# from inside an already-running worker container (no Docker-in-Docker).
#
# On the common path, this means testbld/libbld's own wheel resolution
# (.tune/single/build_triton_wheel.sh) finds a cache hit and never has to
# build anything itself; its own build path remains only as the fallback for
# bare-metal libbld usage that bypasses remotebld entirely.
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

if [ -z "${CELERY_WORKER_PYTHON:-}" ]; then
  echo "Error: CELERY_WORKER_PYTHON not set in config.rc" >&2
  exit 1
fi

# Resolve the actual Python version by invoking the interpreter, rather than
# parsing it out of CELERY_WORKER_IMAGE_BASE's name (fragile).
PYVER=$("$CELERY_WORKER_PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)

# Same grep pattern .ci/common-vars.sh uses to read the AOTriton version.
aotriton_major=$(grep '^set(AOTRITON_VERSION_MAJOR_INT' "$AOTRITON_ROOT/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
aotriton_minor=$(grep '^set(AOTRITON_VERSION_MINOR_INT' "$AOTRITON_ROOT/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)

# shellcheck disable=SC1091
. "$TUNE_ROOT/lib/aotriton_version.sh"
version_dir="$(get_aotriton_version_dir "$AOTRITON_ROOT")"

# Altwheel YAML: reuse .ci's existing per-version convention
# (.ci/<VERSION_DIR>.yaml, e.g. .ci/0.11.1b.yaml) instead of inventing a
# second one. Use it if present, else no altwheel (default hash only).
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
  "${HASHES[@]}"

# When an altwheel YAML applies, produce a resolved copy -- git hashes
# replaced by the actual wheel paths just built above -- so
# build_arch.sh/testbld can pass it straight to AOTriton's own cmake
# (-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=...), which already knows how to
# pick the right venv per arch/kernel-family via its own rule matcher
# (v3python/codegen/root.py); no per-arch selection logic needed here.
# host_wheel_dir == container_wheel_dir since .tune's build runs against
# this same path directly (no docker path-translation layer, unlike .ci's
# release pipeline where replace_hash() rewrites to a container-internal path).
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
