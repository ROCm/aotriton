#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Builds subjects: AOTriton releases (or the checkout, "head"), each with the
# perfmon harness built against it. One subject is
#
#   <prefix>/<arch>/aotriton-<tag>+rocm<ver>/
#     bin/runner                    main.cc linked with the adapter and the core
#     lib/libperfmon_flash.so       the adapter of this release's API
#     lib/libaotriton_pmon_v2.so    the release's runtime (suffix _pmon)
#     lib/aotriton.images/          the release's own kernel images
#     include/                      the release's headers
#
# next to <prefix>/core/ (build_core.sh), which every subject links and none
# contains. All of it resolves through $ORIGIN RUNPATHs, so the tree can be
# moved as a whole.
#
# Usage: build_subject.sh [options] --rocm <ver> <arch> <prefix> <tag>...
#   --rocm <ver>      the ROCm the toolchain must be, e.g. 10.1.0; it is part of
#                     every name, so a different toolchain is refused
#   <arch>            e.g. gfx942; the AOTriton target arch
#   <prefix>          e.g. ${AOTRITON_WORKDIR}/installed/perfmon
#   <tag>...          release tags, e.g. 0.13b 0.14.2b, or head (a full build of
#                     this checkout: HEAD has no published images)
#   --origin <url>    where release tags are cloned from
#   --src_prefix <d>  clones go to <d>/<tag>
#   --build_prefix <d>  build trees go to <d>/<arch>/<tag>/; never under
#                     <prefix>, which holds products only
#   --cache_dir <d>   downloaded release tarballs, shared by tags and arches
#
# Each tag builds in its own subshell, so one failure does not stop the rest;
# the exit status is nonzero if any failed. build_core.sh must have run for the
# same ROCm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/scripts/lib/rocm.sh"

ORIGIN="https://github.com/ROCm/aotriton.git"
SRC_PREFIX="${TMPDIR:-/tmp}/perfmon-src"
BUILD_PREFIX="${TMPDIR:-/tmp}/perfmon-build"
CACHE_DIR="${TMPDIR:-/tmp}/perfmon-cache"
ROCM=""

usage() {
  sed -n "/^# Usage:/,/^# same ROCm\./p" "${BASH_SOURCE[0]}" | sed "s/^# \{0,1\}//" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --rocm)         ROCM="$2"; shift 2 ;;
    --origin)       ORIGIN="$2"; shift 2 ;;
    --src_prefix)   SRC_PREFIX="$2"; shift 2 ;;
    --build_prefix) BUILD_PREFIX="$2"; shift 2 ;;
    --cache_dir)    CACHE_DIR="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Error: unknown option: $1" >&2; usage ;;
    *) break ;;
  esac
done
if [ -z "${ROCM}" ] || [ "$#" -lt 3 ]; then
  usage
fi
ARCH="$1"
PREFIX="${2%/}"
shift 2
TAGS=("$@")
if [ -z "${PREFIX}" ]; then
  echo "Error: empty prefix." >&2
  exit 1
fi

perfmon_resolve_rocm_path
perfmon_require_rocm "${ROCM}"

CORE_ROOT="${PREFIX}/core"
if [ ! -f "${CORE_ROOT}/lib/libperfmon_core+rocm${ROCM}.so" ] || \
   [ ! -f "${CORE_ROOT}/include/perfmon/perfmon_abi.h" ]; then
  echo "Error: no core for ROCm ${ROCM} under ${CORE_ROOT}." >&2
  echo "       Run: perfmon/build_core.sh --rocm ${ROCM} ${PREFIX}" >&2
  exit 1
fi

# The adapter and runner need only HIP's host API but are built with hipcc, as
# the core is. CXX overrides.
HIPCC="${CXX:-}"
if [ -z "${HIPCC}" ]; then
  if [ -x "${ROCM_PATH}/bin/hipcc" ]; then HIPCC="${ROCM_PATH}/bin/hipcc"; else HIPCC=hipcc; fi
fi

build_one_subject() {
  local TAG="$1"
  local SUBJECT_DIR="${PREFIX}/${ARCH}/aotriton-${TAG}+rocm${ROCM}"
  local TAG_BUILD="${SCRIPT_DIR}/scripts/build-${TAG}.sh"
  local ADAPTER_DIR="${REPO_ROOT}/modules/flash/perfmon/runner/${TAG}"
  local WORK="${BUILD_PREFIX}/${ARCH}/${TAG}"

  if [ ! -x "${TAG_BUILD}" ]; then
    echo "[build_subject] ERROR: no build script for '${TAG}': ${TAG_BUILD}." >&2
    exit 1
  fi
  if [ ! -f "${ADAPTER_DIR}/adapter.cc" ]; then
    echo "[build_subject] ERROR: no flash adapter for '${TAG}': ${ADAPTER_DIR}/adapter.cc." \
         "The API is not stable across releases: write one against that release's" \
         "include/aotriton/flash.h, or symlink the directory to a release whose API" \
         "is identical." >&2
    exit 1
  fi

  # --- Source ---------------------------------------------------------------
  # Releases are cloned from upstream, not taken from this checkout, which has
  # no release tags. head is this checkout.
  local SRC_DIR
  if [ "${TAG}" = "head" ]; then
    SRC_DIR="${REPO_ROOT}"
  else
    SRC_DIR="${SRC_PREFIX}/${TAG}"
    if [ ! -e "${SRC_DIR}/.git" ]; then
      echo "[build_subject] git clone --depth 1 --branch ${TAG} ${ORIGIN}" >&2
      mkdir -p "$(dirname "${SRC_DIR}")"
      # A partial clone must not be mistaken for a good one by the check above.
      rm -rf "${SRC_DIR}.partial"
      git clone --depth 1 --branch "${TAG}" "${ORIGIN}" "${SRC_DIR}.partial"
      mv "${SRC_DIR}.partial" "${SRC_DIR}"
    fi
  fi

  # --- AOTriton ---------------------------------------------------------------
  # Into a clean subject directory: a leftover library of an earlier build would
  # leave two libaotriton*_v2.so, and the adapter build refuses that.
  case "${SUBJECT_DIR}" in
    "${PREFIX}/${ARCH}/aotriton-"*) rm -rf "${SUBJECT_DIR}" ;;
    *) echo "[build_subject] ERROR: unexpected subject directory ${SUBJECT_DIR}" >&2; exit 1 ;;
  esac
  mkdir -p "${SUBJECT_DIR}"

  # The build tree is kept: the backends check below reads its generated headers.
  local AOTRITON_BUILD="${WORK}/aotriton"
  echo "[build_subject] ${TAG_BUILD} ${SRC_DIR} ${SUBJECT_DIR} ${ARCH}" >&2
  AOTRITON_BUILD_PATH="${AOTRITON_BUILD}" PERFMON_CACHE_DIR="${CACHE_DIR}" \
    "${TAG_BUILD}" "${SRC_DIR}" "${SUBJECT_DIR}" "${ARCH}"

  # --- Backend table ----------------------------------------------------------
  # A hand-written backends.csv (releases before the public backends.h) must
  # agree with what the build itself generated.
  local CSV="${ADAPTER_DIR}/backends.csv"
  if [ -f "${CSV}" ] && grep -qE '^op_attn_(fwd|bwd),' "${CSV}"; then
    python3 -I "${SCRIPT_DIR}/scripts/lib/check_backends_csv.py" "${CSV}" "${AOTRITON_BUILD}"
  fi

  # --- Adapter and runner -------------------------------------------------------
  local FLASH_BUILD="${WORK}/flash"
  echo "[build_subject] adapter ${TAG}/adapter.cc against ${SUBJECT_DIR}" >&2
  # A fresh tree: a cache from an earlier configure could keep old paths.
  rm -rf "${FLASH_BUILD}"
  cmake -S "${REPO_ROOT}/modules/flash/perfmon/runner" -B "${FLASH_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${HIPCC}" \
    -DCMAKE_INSTALL_PREFIX="${SUBJECT_DIR}" \
    -DAOTRITON_ROOT="${SUBJECT_DIR}" \
    -DPERFMON_CORE_ROOT="${CORE_ROOT}" \
    -DPERFMON_ROCM_VERSION="${ROCM}" \
    -DPERFMON_FLASH_TAG="${TAG}"
  cmake --build "${FLASH_BUILD}"
  cmake --install "${FLASH_BUILD}"

  echo "[build_subject] done: ${SUBJECT_DIR}/bin/runner" >&2
}

FAILED=()
for TAG in "${TAGS[@]}"; do
  echo "[build_subject] === ${TAG} (${ARCH}, ROCm ${ROCM}) ===" >&2
  # A subshell, so one failure does not stop the batch. `set -e` is set again
  # inside it: bash ignores it for anything called in an `if` or `||` context,
  # which would let a failed step go unnoticed.
  set +e
  ( set -e; build_one_subject "${TAG}" )
  rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    echo "[build_subject] ok: ${TAG}" >&2
  else
    echo "[build_subject] FAILED: ${TAG}" >&2
    FAILED+=("${TAG}")
  fi
done

if [ "${#FAILED[@]}" -ne 0 ]; then
  echo "[build_subject] ${#FAILED[@]} of ${#TAGS[@]} subject(s) failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "[build_subject] all ${#TAGS[@]} subject(s) built for ${ARCH}" >&2
