#!/bin/bash
# Runs inside the AlmaLinux 8 ROCm Docker container (manylinux_2_28 environment).
# Fed via stdin by build_inside() in releasesuite-git-head.sh; no bind mount needed.
# Positional args: $1=NOIMAGE_MODE $2=WHEEL_CFG $3=ASAN_MODE
#
# Container prerequisites:
#   Mounts:
#     /src/aotriton  — AOTriton source tree (read-only git repo)
#     /output        — destination for output tar archives
#     /cache         — shared cache; /cache/wheels holds pre-built Triton wheels
#   Environment variables:
#     AOTRITON_BUILD_PATH    — build directory, e.g. /scratch/build/aotriton
#                              Recommended: mount /scratch as tmpfs with exec flag
#                              and sufficient memory to hold the full build tree.
#     AOTRITON_INSTALL_PREFIX — install prefix, e.g. /scratch/install
#                               AOTRITON_INSTALL_PATH is derived as $AOTRITON_INSTALL_PREFIX/aotriton
#   Tools (provided by the ROCm AlmaLinux 8 image):
#     hipconfig        — to locate ROCM_PATH
#     gcc-toolset-13   — C++17 compiler via scl enable (non-asan path only)
#     cpp              — preprocessor used to extract the HIP version number

set -ex

# --- Arguments ---
NOIMAGE_MODE="$1"
WHEEL_CFG="$2"
ASAN_MODE="${3:-OFF}"

# --- Validate environment ---
if [ -z "${AOTRITON_BUILD_PATH}" ]; then
  echo "Error: AOTRITON_BUILD_PATH is not set." >&2; exit 1
fi
if [ -z "${AOTRITON_INSTALL_PREFIX}" ]; then
  echo "Error: AOTRITON_INSTALL_PREFIX is not set." >&2; exit 1
fi
export AOTRITON_INSTALL_PATH="${AOTRITON_INSTALL_PREFIX}/aotriton"

# --- Detect ROCm and HIP version ---
GIT_SHORT=$(git -C /src/aotriton rev-parse --short=12 HEAD)
export ROCM_PATH=$(hipconfig --rocmpath)
if [ -z "${ROCM_PATH}" ]; then
  echo "Error: ROCM_PATH is empty. hipconfig --rocmpath failed." >&2
  exit 1
fi
printf '#include <hip/hip_version.h>\nHIP_VERSION_MAJOR . HIP_VERSION_MINOR\n' > /tmp/print_hip_version.h
if [[ "${ASAN_MODE}" == "ON" ]]; then
  # No gcc-toolset/cpp in this path; use clang's preprocessor.
  hipver=$(${ROCM_PATH}/llvm/bin/clang -E -P -x c -I${ROCM_PATH}/include /tmp/print_hip_version.h | tail -n 1 | sed 's/ //g')
else
  hipver=$(scl enable gcc-toolset-13 "cpp -I${ROCM_PATH}/include /tmp/print_hip_version.h" | tail -n 1 | sed 's/ //g')
fi

# --- Build ---
build_args=("${NOIMAGE_MODE}" "ALL")
if [[ "${WHEEL_CFG}" == *.yml || "${WHEEL_CFG}" == *.yaml ]]; then
  cmake_arg="-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=${WHEEL_CFG}"
else
  cmake_arg="-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${WHEEL_CFG}"
fi
build_args+=("${cmake_arg}")

if [[ "${ASAN_MODE}" == "ON" ]]; then
  build_args+=(
    "-DAOTRITON_ENABLE_ASAN_CLANG=ON"
    "-DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang"
    "-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++"
  )
  # Use theRock clang directly — no scl gcc-toolset wrapper.
  bash /src/aotriton/.ci/build-release.sh "${build_args[@]}"
else
  scl enable gcc-toolset-13 -- bash /src/aotriton/.ci/build-release.sh "${build_args[@]}"
fi

# manylinux tag matches the build platform's glibc baseline.
# AlmaLinux 8 (aotriton:base) ships glibc 2.28 -> manylinux_2_28.
# Hardcoded rather than auto-detected: objdump-based "max GLIBC_x.y in
# the .so" measures the highest symbol version *referenced*, not the
# baseline glibc the binary was linked against, so a binary that happens
# to use only old symbols would falsely tag as a lower manylinux.
MANYLINUX_TAG="manylinux_2_28"

asan_suffix=""
if [[ "${ASAN_MODE}" == "ON" ]]; then
  asan_suffix="+asan"
fi

# --- Package (both archives must have aotriton/ as the root directory) ---
if [ ${NOIMAGE_MODE} == "OFF" ]; then
  tarbase=aotriton-${GIT_SHORT}${asan_suffix}-images
  cd "${AOTRITON_INSTALL_PREFIX}"
  for d in $(ls aotriton/lib/aotriton.images/); do
    tarfile=${tarbase}-$d.tar.gz
    tar cz "aotriton/lib/aotriton.images/$d" > /output/${tarfile}
  done
else
  tarfile=aotriton-${GIT_SHORT}${asan_suffix}-${MANYLINUX_TAG}_x86_64-rocm${hipver}-shared.tar.gz
  cd "${AOTRITON_INSTALL_PREFIX}" && tar cz aotriton > /output/${tarfile}
fi
