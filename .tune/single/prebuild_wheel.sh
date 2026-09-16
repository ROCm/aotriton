#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-build the compiler artifacts this workdir's builds need, into
# <workdir>/scratch/, via the .ci/ build scripts. Needs Docker; called by
# remotebld as a sibling step before testbld/libbld (no Docker-in-Docker),
# which then just hit these caches.
#
#   scratch/triton/         Triton wheel(s)         .ci/build_triton_wheels.sh
#   scratch/llvm-tarballs/  LLVM/MLIR tarball       .ci/build_llvm_tarball.sh
#   scratch/flydsl/         FlyDSL compiler wheel   .ci/build_flydsl_wheel.sh
#
# The two FlyDSL stages run only when the tree needs a locally built FlyDSL
# wheel -- see flydsl_required() in .tune/lib/find_flydsl.sh, which owns that
# decision. When it says no, this script does exactly what it always did.
#
# Usage: prebuild_wheel.sh [--select triton|flydsl-llvm|flydsl-wheel|all] <workdir>
#
# --select drives one stage on its own, for debugging; default is `all`, which
# runs them in dependency order (the FlyDSL wheel is built against the tarball).

set -euo pipefail

WORKDIR=""
SELECT="all"
while [ "$#" -gt 0 ]; do
  case "$1" in
    # `shift 2` with one argument left FAILS WITHOUT SHIFTING, which spins this
    # loop forever on a trailing `--select`. Same guard, for the same reason, as
    # .ci/build_llvm_tarball.sh:48-56.
    --select)
      [ "$#" -ge 2 ] || { echo "Error: --select requires a value." >&2; exit 1; }
      SELECT="$2"; shift 2 ;;
    *) WORKDIR="$1"; shift ;;
  esac
done

case "$SELECT" in
  triton|flydsl-llvm|flydsl-wheel|all) ;;
  *) echo "Error: --select must be one of triton, flydsl-llvm, flydsl-wheel, all (got '$SELECT')." >&2
     exit 1 ;;
esac

if [ -z "$WORKDIR" ]; then
  echo "Usage: $0 [--select triton|flydsl-llvm|flydsl-wheel|all] <workdir>" >&2
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

# shellcheck disable=SC1091
. "$TUNE_ROOT/lib/aotriton_version.sh"
# pin_line first: find_flydsl.sh's helpers are written against it, the same way
# .ci/releasesuite-git-head.sh sources common-pin.sh before reading any pin.
# shellcheck disable=SC1091
. "$AOTRITON_ROOT/.ci/common-pin.sh"
# shellcheck disable=SC1091
. "$TUNE_ROOT/lib/find_flydsl.sh"
read -r aotriton_major aotriton_minor <<< "$(get_aotriton_major_minor "$AOTRITON_ROOT")"
version_dir="$(get_aotriton_version_dir "$AOTRITON_ROOT")"

# CELERY_WORKER_PYTHON only exists inside CELERY_WORKER_IMAGE, not on the
# host -- run it in a throwaway container instead of invoking it directly.
# The version is fixed for the life of a given image tag, so cache it keyed
# on that tag: remotebld calls this script before every single build, and
# without caching each call pays a full container spin-up just to learn a
# constant.
PYVER_CACHE="$WORKDIR/scratch/.pyver_cache"
if [ -f "$PYVER_CACHE" ] && [ "$(cut -d' ' -f1 "$PYVER_CACHE" 2>/dev/null)" = "$CELERY_WORKER_IMAGE" ]; then
  PYVER=$(cut -d' ' -f2 "$PYVER_CACHE")
else
  PYVER=$(docker run --rm "$CELERY_WORKER_IMAGE" "$CELERY_WORKER_PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
  mkdir -p "$(dirname "$PYVER_CACHE")"
  echo "$CELERY_WORKER_IMAGE $PYVER" > "$PYVER_CACHE"
fi

LLVM_TARBALL_DIR="$WORKDIR/scratch/llvm-tarballs"
FLYDSL_WHEEL_DIR="$WORKDIR/scratch/flydsl"

stage_triton() {
  # Reuse .ci's per-version altwheel convention if present.
  local ALTWHEEL_YAML=""
  if [ -f "$AOTRITON_ROOT/.ci/${version_dir}.yaml" ]; then
    ALTWHEEL_YAML="$AOTRITON_ROOT/.ci/${version_dir}.yaml"
  fi

  local HASHES
  mapfile -t HASHES < <(bash "$AOTRITON_ROOT/.ci/resolve-triton-hashes.sh" "$AOTRITON_ROOT" "$ALTWHEEL_YAML")

  echo "Pre-building Triton wheel(s) for python ${PYVER}: ${HASHES[*]}"

  bash "$AOTRITON_ROOT/.ci/build_triton_wheels.sh" \
    --wheel_output_dir "$WORKDIR/scratch/triton" \
    --version_suffix "+aotriton${aotriton_major}.${aotriton_minor}" \
    --python "$PYVER" \
    --altwheel_yaml "$ALTWHEEL_YAML" \
    "${HASHES[@]}"

  # Resolve hashes -> wheel paths for build_arch.sh/testbld to pass as
  # -DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE, via .ci/common-altwheel.sh's
  # altwheel_resolve_config (also used by .ci/releasesuite-git-head.sh). This
  # script runs on the HOST (remotebld's sibling pre-build step, before the
  # testbld/libbld container launches) -- $WORKDIR is the host path, used to
  # actually find the built wheels here. But the resolved yaml is read later
  # INSIDE the worker container, where remotebld always mounts the same
  # workdir at the fixed path /wkdir (see remotebld's --mount target=/wkdir,
  # identical across all 4 of its branches) -- so the wheel paths written
  # into the yaml must use /wkdir, not $WORKDIR, or pip inside the container
  # looks for a file that only exists at that path on the host.
  if [ -n "$ALTWHEEL_YAML" ]; then
    if ! command -v yq &> /dev/null; then
      echo "Error: 'yq' is required to resolve altwheel YAML $ALTWHEEL_YAML (dnf install yq / snap install yq)" >&2
      exit 1
    fi

    # shellcheck disable=SC1091
    . "$AOTRITON_ROOT/.ci/common-altwheel.sh"

    local RESOLVED_YAML="$WORKDIR/scratch/triton/resolved_altwheel.yaml"
    cp "$ALTWHEEL_YAML" "$RESOLVED_YAML"
    altwheel_resolve_config "$RESOLVED_YAML" "$WORKDIR/scratch/triton" "/wkdir/scratch/triton" "${HASHES[0]}" "$PYVER"

    echo "Resolved altwheel config: $RESOLVED_YAML"
  fi
}

# The LLVM/MLIR tarball the FlyDSL wheel is linked against. Hours on a cold
# cache, seconds on a warm one -- build_llvm_tarball.sh keys it on the resolved
# commit and returns early on a hit, so calling this every time is cheap.
stage_flydsl_llvm() {
  echo "Pre-building the LLVM/MLIR tarball for FlyDSL (python ${PYVER})"
  bash "$AOTRITON_ROOT/.ci/build_llvm_tarball.sh" \
    --tarball_output_dir "$LLVM_TARBALL_DIR" \
    --python "$PYVER"
}

# The FlyDSL compiler wheel itself. The tarball is read from the cache rather
# than built here: --select exists so a stage can be driven alone, and quietly
# spending an hour on LLVM inside the stage someone picked *because* they
# wanted only the wheel would defeat that. .ci/build_flydsl_wheel.sh takes the
# tarball as a required input for the same reason -- it is a sibling input, not
# a private dependency (see its "The LLVM half is the CALLER's" note).
stage_flydsl_wheel() {
  local ref tarball
  ref="$(flydsl_ref_from_pin "$AOTRITON_ROOT")" || exit 1

  # Newest by mtime: the name carries the commit, so any tarball here is
  # legitimate, and build_flydsl_wheel.sh re-derives the LLVM identity from the
  # filename for its own cache key -- picking the freshest is right when a pin
  # has moved and both are still on disk.
  # `|| true` is load-bearing under `set -euo pipefail`: with no tarball dir at
  # all, ls exits 2, pipefail propagates it through `| head`, and the script
  # dies on the assignment -- taking the explanatory error below with it and
  # leaving the caller a bare exit 2.
  tarball=$(ls -t "$LLVM_TARBALL_DIR"/llvm-*.tar.gz 2>/dev/null | head -n 1) || true
  if [ -z "$tarball" ]; then
    echo "Error: no LLVM tarball in $LLVM_TARBALL_DIR, and this stage does not build one." >&2
    echo "  Run: $0 --select flydsl-llvm $WORKDIR" >&2
    echo "  (or --select all, which runs both stages in order)" >&2
    exit 1
  fi

  echo "Pre-building the FlyDSL wheel ${ref} for python ${PYVER} against $(basename "$tarball")"
  bash "$AOTRITON_ROOT/.ci/build_flydsl_wheel.sh" \
    --wheel_output_dir "$FLYDSL_WHEEL_DIR" \
    --flydsl_commit "$ref" \
    --llvm_tarball "$tarball" \
    --python "$PYVER" \
    --version_suffix ".aotriton${aotriton_major}.${aotriton_minor}"
}

# flydsl_required is consulted once, here, so an empty pin makes both FlyDSL
# stages vanish -- including when they were asked for by name, which is the
# honest answer to "build me a wheel this tree does not need".
FLYDSL_NEEDED=0
if flydsl_required "$AOTRITON_ROOT"; then
  FLYDSL_NEEDED=1
fi

run_flydsl_stage() {
  if [ "$FLYDSL_NEEDED" -eq 0 ]; then
    echo "third_party/flydsl-llvm.txt is empty: the pinned FlyDSL wheel is usable," \
         "so there is nothing to build. Skipping $1."
    return 0
  fi
  "$1"
}

case "$SELECT" in
  triton)       stage_triton ;;
  flydsl-llvm)  run_flydsl_stage stage_flydsl_llvm ;;
  flydsl-wheel) run_flydsl_stage stage_flydsl_wheel ;;
  all)
    stage_triton
    run_flydsl_stage stage_flydsl_llvm
    run_flydsl_stage stage_flydsl_wheel
    ;;
esac
