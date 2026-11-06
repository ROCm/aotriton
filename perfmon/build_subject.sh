#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Builds subjects: AOTriton releases (or the checkout, "head"), each with the
# perfmon harness built against it. One subject is
#
#   <prefix>/<pool>/aotriton-<tag>+rocm<ver>/
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
# <pool> is MAJOR or an arch. MAJOR is one subject for the five arches of
# AOTRITON_MAJOR_ARCHES (python/gpu_targets.py): one shim built for all of them
# and the release's images of all of them in lib/aotriton.images/, so that
# the artifacts cannot diverge between arches; the dev team always uses it. An
# arch the release has no images for is listed in missing_images.txt in the
# subject. A subject of one arch (<prefix>/<arch>/...) is for external users of
# a non-major arch. At run time a device arch is served by the pool
# perfmon_subject_pool (scripts/lib/arches.sh) names, MAJOR for a major arch.
#
# Tune and perfmon share one way to build on the remote build node (prepwkdir -> <workdir>/aotriton.src -> sync_workdir.sh --buildnode), so
# an operator who knows one app's build flow knows the other's.
#
# Usage: build_subject.sh [options] --rocm <ver> <arch> <prefix> <tag>...
#   --rocm <ver>      the ROCm the toolchain must be, e.g. 10.1.0; it is part of
#                     every name, so a different toolchain is refused
#   <arch>            MAJOR, or one AOTriton target arch, e.g. gfx1101; a major
#                     arch (e.g. gfx942) is refused: it is served by MAJOR
#   <prefix>          e.g. ${AOTRITON_WORKDIR}/installed/perfmon
#   <tag>...          release tags, e.g. 0.13b 0.14.2b, or head: the committed
#                     tip of the operator's branch, as .def/bin/prepwkdir clones
#                     it into <workdir>/aotriton.src and
#                     .def/libexec/sync_workdir.sh <workdir> <buildnode>
#                     --buildnode copies it to the build node (uncommitted work
#                     is not in it). Subject aotriton-head+rocm<ver>, for live
#                     comparison only; <subject>/source.txt says which commit
#                     was built
#   --head_src <d>    the source of head; default $AOTRITON_WORKDIR/aotriton.src
#   --origin <url>    where release tags are cloned from
#   --src_prefix <d>  clones go to <d>/<tag>
#   --build_prefix <d>  build trees go to <d>/<arch>/<tag>/; never under
#                     <prefix>, which holds products only
#   --cache_dir <d>   downloaded release tarballs, shared by tags and arches
#   --triton_wheel <whl>  head only: the local Triton wheel (absolute path) the
#                     build compiles kernels with, as -DAOTRITON_USE_LOCAL_TRITON_WHEEL
#                     (.ci/build_triton_wheels.sh builds one; .ci/build-test.sh's
#                     optional wheel argument)
#   --flydsl_wheel <whl>  head only: the local FlyDSL wheel (absolute path), as
#                     -DAOTRITON_USE_LOCAL_FLYDSL_WHEEL. Required while
#                     third_party/flydsl-llvm.txt pins an LLVM (CMake refuses
#                     otherwise): build one against that LLVM with
#                     .ci/build_flydsl_wheel.sh --wheel_output_dir <dir> --flydsl_commit <sha>
#                     Releases ship images, so neither wheel applies to them.
#                     PERFMON_HEAD_CMAKE_ARGS (more -D options for head) is
#                     still honored, after these.
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
HEAD_SRC="${AOTRITON_WORKDIR:+${AOTRITON_WORKDIR}/aotriton.src}"
TRITON_WHEEL=""
FLYDSL_WHEEL=""

usage() {
  sed -n "/^# Usage:/,/^# same ROCm\./p" "${BASH_SOURCE[0]}" | sed "s/^# \{0,1\}//" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --rocm)         ROCM="$2"; shift 2 ;;
    --head_src)     HEAD_SRC="$2"; shift 2 ;;
    --origin)       ORIGIN="$2"; shift 2 ;;
    --src_prefix)   SRC_PREFIX="$2"; shift 2 ;;
    --build_prefix) BUILD_PREFIX="$2"; shift 2 ;;
    --cache_dir)    CACHE_DIR="$2"; shift 2 ;;
    --triton_wheel) TRITON_WHEEL="$2"; shift 2 ;;
    --flydsl_wheel) FLYDSL_WHEEL="$2"; shift 2 ;;
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

. "${SCRIPT_DIR}/scripts/lib/arches.sh"
if [ "${ARCH}" != "${PERFMON_MAJOR_POOL}" ] && \
   [ "$(perfmon_subject_pool "${ARCH%%:*}")" = "${PERFMON_MAJOR_POOL}" ]; then
  echo "Error: ${ARCH} is a major arch: build ${PERFMON_MAJOR_POOL}, which serves it." >&2
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
  # Releases are cloned from upstream, which this checkout has no tags of. head
  # is the workdir's aotriton.src (prepwkdir), as the tuner builds from.
  local SRC_DIR
  if [ "${TAG}" = "head" ]; then
    SRC_DIR="${HEAD_SRC}"
    if [ -z "${SRC_DIR}" ] || [ ! -d "${SRC_DIR}/.git" ]; then
      echo "[build_subject] ERROR: no source for head: '${SRC_DIR:-<unset>}' is not a git clone." \
           "Run .def/bin/prepwkdir <workdir> and .def/libexec/sync_workdir.sh" \
           "<workdir> <buildnode> --buildnode first (or pass --head_src <dir>)." >&2
      exit 1
    fi
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

  # The wheels head's kernels are compiled with (releases ship images instead).
  local HEAD_ARGS=""
  if [ "${TAG}" = "head" ]; then
    local W
    for W in "${TRITON_WHEEL}" "${FLYDSL_WHEEL}"; do
      if [ -n "${W}" ] && { [ "${W#/}" = "${W}" ] || [ ! -f "${W}" ]; }; then
        echo "[build_subject] ERROR: wheel '${W}' is not an existing file with an absolute path." >&2
        exit 1
      fi
    done
    [ -z "${TRITON_WHEEL}" ] || HEAD_ARGS+=" -DAOTRITON_USE_LOCAL_TRITON_WHEEL=${TRITON_WHEEL}"
    [ -z "${FLYDSL_WHEEL}" ] || HEAD_ARGS+=" -DAOTRITON_USE_LOCAL_FLYDSL_WHEEL=${FLYDSL_WHEEL}"
    HEAD_ARGS+=" ${PERFMON_HEAD_CMAKE_ARGS:-}"
    # Fail before a long configure, as CMakeLists.txt would after it.
    if [ -f "${SRC_DIR}/third_party/flydsl-llvm.txt" ] && \
       grep -qvE '^[[:space:]]*(#|$)' "${SRC_DIR}/third_party/flydsl-llvm.txt" && \
       [[ "${HEAD_ARGS}" != *AOTRITON_USE_LOCAL_FLYDSL_WHEEL=* ]]; then
      echo "[build_subject] ERROR: third_party/flydsl-llvm.txt pins an LLVM, so head needs a FlyDSL" \
           "wheel built against it: pass --flydsl_wheel <abs .whl> (build one with" \
           ".ci/build_flydsl_wheel.sh --wheel_output_dir <dir> --flydsl_commit <sha>)." \
           "A local Triton wheel (--triton_wheel; .ci/build_triton_wheels.sh) is needed too" \
           "unless the build can fetch one." >&2
      exit 1
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
    PERFMON_HEAD_CMAKE_ARGS="${HEAD_ARGS# }" \
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

  # What was built: the commit of the source (head: the aotriton.src clone, which
  # is clean unless someone edited it; dirty = uncommitted changes).
  {
    echo "describe: $(git -C "${SRC_DIR}" describe --always --dirty 2>/dev/null || echo unknown)"
    echo "commit: $(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "changed_files: $(git -C "${SRC_DIR}" status --porcelain 2>/dev/null | wc -l)"
  } > "${SUBJECT_DIR}/source.txt"

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
