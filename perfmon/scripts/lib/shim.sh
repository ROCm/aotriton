# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# What every perfmon/scripts/build-<tag>.sh shares. Source it. What stays in
# the per-tag script is what drifts between releases: how the target arch is
# passed, which kernel-image asset to download, and any option the release
# needs. (.ci/ is not a stable interface across tags: 0.9.2b and 0.10b have no
# .ci/, and the helpers that exist differ in signature, so none is used.)
#
# The build is a shim: only the runtime, with the release's own kernel images
# unpacked next to it, so the kernels measured are exactly the shipped ones.
# A per-tag script is called as
#     build-<tag>.sh <src_dir> <install_dir> <arch>
# with, optionally, in the environment:
#     AOTRITON_BUILD_PATH     the cmake build tree (kept after the build:
#                             build_subject.sh reads the generated
#                             iface.op_attn_*.h from it)
#     AOTRITON_CXX            the compiler (default g++, never hipcc: hipcc is
#                             clang, which rejects narrowing that old releases'
#                             sources contain, and the shim needs only HIP's
#                             host API)
#     PERFMON_CACHE_DIR       where release tarballs are cached
# and leaves <install_dir>/include/aotriton, lib/libaotriton*_v2.so and
# lib/aotriton.images.

_SHIM_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${_SHIM_LIB_DIR}/rocm.sh"
. "${_SHIM_LIB_DIR}/release_asset.sh"

# The suffix is pasted into the library file name (libaotriton_pmon_v2.so) and
# into the C++ namespace (aotriton_pmon), so it must be a valid identifier
# part: '-', '.' and '+' are out, and '__' is reserved by C++.
SHIM_NAME_SUFFIX="_pmon"

# shim_init <tag> <src_dir> <install_dir> <arch>
# Sets TAG, SRC_DIR, INSTALL_DIR, ARCH and BUILD_DIR.
shim_init() {
  if [ "$#" -ne 4 ]; then
    echo "Usage: build-<tag>.sh <src_dir> <install_dir> <arch>" >&2
    exit 1
  fi
  TAG="$1"
  if [ ! -f "$2/CMakeLists.txt" ]; then
    echo "[build-${TAG}] ERROR: $2 is not an AOTriton checkout (no CMakeLists.txt)." >&2
    exit 1
  fi
  SRC_DIR="$(cd "$2" && pwd)"
  mkdir -p "$3"
  INSTALL_DIR="$(cd "$3" && pwd)"
  ARCH="$4"
  if [ -z "${ARCH}" ]; then
    echo "[build-${TAG}] ERROR: empty <arch>." >&2
    exit 1
  fi
  BUILD_DIR="${AOTRITON_BUILD_PATH:-${TMPDIR:-/tmp}/aotriton-${TAG}-shim-${ARCH}}"
  AOTRITON_CXX="${AOTRITON_CXX:-g++}"

  if ! command -v "${AOTRITON_CXX}" >/dev/null 2>&1; then
    echo "[build-${TAG}] ERROR: compiler '${AOTRITON_CXX}' not found; set AOTRITON_CXX." >&2
    exit 1
  fi
  # Old tags only search /opt/rocm on their own, so ROCM_PATH must be in the
  # environment, and in CMAKE_PREFIX_PATH.
  perfmon_resolve_rocm_path
  export CMAKE_PREFIX_PATH="${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  echo "[build-${TAG}] src=${SRC_DIR} install=${INSTALL_DIR} build=${BUILD_DIR} arch=${ARCH}" \
       "CXX=${AOTRITON_CXX} ROCM_PATH=${ROCM_PATH}" >&2
}

# shim_require_lzma
# Kernel Storage V2 reads xz through pkg-config (0.10b and later).
shim_require_lzma() {
  if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists liblzma; then
    echo "[build-${TAG}] ERROR: pkg-config cannot find liblzma; install pkg-config and a" \
         "liblzma development package (liblzma-dev on Debian)." >&2
    exit 1
  fi
}

# shim_build <noimage ON|OFF> [cmake options...]
# Configures and installs with the options every tag needs. The caller adds the
# arch option, whose name differs between tags. A stale build tree is removed
# first (only if it is a cmake tree), since a suffix or compiler change would
# otherwise be silently ignored by its cache.
shim_build() {
  local noimage="$1"
  shift
  if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    rm -rf "${BUILD_DIR}"
  fi
  mkdir -p "${BUILD_DIR}"
  cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${AOTRITON_CXX}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DAOTRITON_NOIMAGE_MODE="${noimage}" \
    -DAOTRITON_NO_PYTHON=ON \
    -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
    -DAOTRITON_NAME_SUFFIX="${SHIM_NAME_SUFFIX}" \
    "$@"
  ninja -C "${BUILD_DIR}" install
}

# shim_install_images <asset_name>
# The images of the release, unpacked into <install_dir>/lib/aotriton.images.
shim_install_images() {
  local tarball
  tarball="$(fetch_release_asset "${TAG}" "$1" "${BUILD_DIR}/images-download")"
  install_images_from_tarball "${tarball}" "${INSTALL_DIR}"
}

# shim_finish
# Fails unless the install holds what build_subject.sh and the adapter need.
shim_finish() {
  if [ ! -f "${INSTALL_DIR}/include/aotriton/flash.h" ]; then
    echo "[build-${TAG}] ERROR: ${INSTALL_DIR}/include/aotriton/flash.h is missing." >&2
    exit 1
  fi
  if ! compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so" >/dev/null; then
    echo "[build-${TAG}] ERROR: no ${INSTALL_DIR}/lib/libaotriton*_v2.so was installed." >&2
    exit 1
  fi
  if [ ! -d "${INSTALL_DIR}/lib/aotriton.images" ]; then
    echo "[build-${TAG}] ERROR: no ${INSTALL_DIR}/lib/aotriton.images: the subject would" \
         "have no kernels to measure." >&2
    exit 1
  fi
  echo "[build-${TAG}] ok: ${INSTALL_DIR}" >&2
}
