#!/bin/bash
# Runs inside the AlmaLinux 8 ROCm Docker container (manylinux_2_28 environment).
# Bind-mounted by build_inside() in releasesuite-git-head.sh at /tmp/runc-manylinux-build-tar.sh
# and invoked as: bash -l /tmp/runc-manylinux-build-tar.sh <args>
# Positional args: $1=NOIMAGE_MODE $2=WHEEL_CFG $3=ARCH_LIST
#   ARCH_LIST: "ALL" (default) or a ';'-separated GPU arch list forwarded
#   to build-release.sh as its arch_list arg (becomes AOTRITON_TARGET_ARCH).
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
#     gcc-toolset-13   — C++17 compiler via scl enable
#     cpp              — preprocessor used to extract the HIP version number

set -ex

# --- Arguments ---
NOIMAGE_MODE="$1"
WHEEL_CFG="$2"
ARCH_LIST="${3:-ALL}"

# --- Validate environment ---
if [ -z "${AOTRITON_BUILD_PATH}" ]; then
  echo "Error: AOTRITON_BUILD_PATH is not set." >&2; exit 1
fi
if [ -z "${AOTRITON_INSTALL_PREFIX}" ]; then
  echo "Error: AOTRITON_INSTALL_PREFIX is not set." >&2; exit 1
fi
export AOTRITON_INSTALL_PATH="${AOTRITON_INSTALL_PREFIX}/aotriton"

# pip (running as root) refuses a cache dir not owned by root and silently
# disables caching. PIP_CACHE_DIR=/cache/pip is bind-mounted from the host
# and owned by the host UID, so take ownership inside the container.
if [ -n "${PIP_CACHE_DIR}" ] && [ -d "${PIP_CACHE_DIR}" ]; then
  chown -R "$(id -u):$(id -g)" "${PIP_CACHE_DIR}" || true
fi

# --- Detect ROCm and HIP version ---
GIT_SHORT=$(git -C /src/aotriton rev-parse --short=12 HEAD)
export ROCM_PATH=$(hipconfig --rocmpath)
if [ -z "${ROCM_PATH}" ]; then
  echo "Error: ROCM_PATH is empty. hipconfig --rocmpath failed." >&2
  exit 1
fi
printf '#include <hip/hip_version.h>\nHIP_VERSION_MAJOR . HIP_VERSION_MINOR\n' > /tmp/print_hip_version.h
hipver=$(scl enable gcc-toolset-13 "cpp -I${ROCM_PATH}/include /tmp/print_hip_version.h" | tail -n 1 | sed 's/ //g')

# --- Build ---
# Only image builds embed a Triton wheel. Runtime builds (NOIMAGE_MODE=ON)
# run with AOTRITON_NOIMAGE_MODE=ON and skip Triton entirely, so no wheel
# config is passed (WHEEL_CFG is "NONE" in that case).
build_args=("${NOIMAGE_MODE}" "${ARCH_LIST}")
if [ "${NOIMAGE_MODE}" == "OFF" ]; then
  if [[ "${WHEEL_CFG}" == *.yml || "${WHEEL_CFG}" == *.yaml ]]; then
    cmake_arg="-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=${WHEEL_CFG}"
  else
    cmake_arg="-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${WHEEL_CFG}"
  fi
  build_args+=("${cmake_arg}")
fi
scl enable gcc-toolset-13 -- bash /src/aotriton/.ci/build-release.sh "${build_args[@]}"

# --- Package (both archives must have aotriton/ as the root directory) ---
if [ ${NOIMAGE_MODE} == "OFF" ]; then
  tarbase=aotriton-${GIT_SHORT}-images
  cd "${AOTRITON_INSTALL_PREFIX}"
  for d in $(ls aotriton/lib/aotriton.images/); do
    tarfile=${tarbase}-$d.tar.gz
    tar cz "aotriton/lib/aotriton.images/$d" > /output/${tarfile}
  done
else
  tarfile=aotriton-${GIT_SHORT}-manylinux_2_28_x86_64-rocm${hipver}-shared.tar.gz
  cd "${AOTRITON_INSTALL_PREFIX}" && tar cz aotriton > /output/${tarfile}
fi

# Debug: drop into interactive shell after everything is done so the full
# build and install tree can be inspected. Requires -t from the caller.
if [[ "${SUITE_DEBUG:-0}" == "1" ]]; then
  if [ -t 0 ]; then
    echo "=== DEBUG MODE: build complete. Dropping into interactive shell. ===" >&2
    bash -i </dev/tty >/dev/tty 2>&1 || true
  else
    echo "=== DEBUG MODE: build complete, but no TTY available. Skipping interactive shell. ===" >&2
  fi
fi
