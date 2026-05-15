#!/bin/bash

set -ex

TRITON_LLVM_HASH="$1"
NOIMAGE_MODE="$2"
ALTWHEEL_CFG="$3"

cd /src/aotriton/
GIT_FULL=$(git rev-parse HEAD)
GIT_SHORT=$(git rev-parse --short=12 HEAD)
cd /src/aotriton/third_party/triton
TRITON_SHORT=$(git rev-parse --short=12 HEAD)
export TRITON_WHEEL_VERSION_SUFFIX="+git${TRITON_SHORT}"
export ROCM_PATH=$(hipconfig --rocmpath)
if [ -z "${ROCM_PATH}" ]; then
  echo "Error: ROCM_PATH is empty. hipconfig --rocmpath failed." >&2
  exit 1
fi
hipver=$(scl enable gcc-toolset-13 "cpp -I${ROCM_PATH}/include /input/print_hip_version.h"|tail -n 1|sed 's/ //g')

if [ ${NOIMAGE_MODE} == "OFF" ]; then
  fn="llvm-${TRITON_LLVM_HASH}-almalinux-x64.tar.gz"
  if [ -f "/input/$fn" ]; then
    mkdir -p "$HOME/.triton/llvm"
    cd "$HOME/.triton/llvm"
    echo "Unpacking $fn" && tar xf "/input/$fn"
    echo -n "https://oaitriton.blob.core.windows.net/public/llvm-builds/$fn" > "llvm-${TRITON_LLVM_HASH}-almalinux-x64/version.txt"
  fi
fi

if [ -z "${AOTRITON_BUILD_PATH}" ]; then
  echo "Error: AOTRITON_BUILD_PATH is not set." >&2
  exit 1
fi
if [ -z "${AOTRITON_INSTALL_PATH}" ]; then
  echo "Error: AOTRITON_INSTALL_PATH is not set." >&2
  exit 1
fi
export AOTRITON_CI_SUPPLIED_SHA1=${GIT_FULL}
if [ -z "${ALTWHEEL_CFG}" ]; then
  scl enable gcc-toolset-13 -- bash /src/aotriton/.ci/build-release.sh "${NOIMAGE_MODE}"
else
  scl enable gcc-toolset-13 -- bash /src/aotriton/.ci/build-release.sh "${NOIMAGE_MODE}" "ALL" \
    "-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=${ALTWHEEL_CFG}"
fi

# Both tar archives must have aotriton/ as the root directory.
if [ ${NOIMAGE_MODE} == "OFF" ]; then
  tarbase=aotriton-${GIT_SHORT}-images
  cd "${AOTRITON_INSTALL_PATH}/.."
  for d in $(ls aotriton/lib/aotriton.images/); do
    tarfile=${tarbase}-$d.tar.gz
    tar cz "aotriton/lib/aotriton.images/$d" > /output/${tarfile}
  done
else
  tarfile=aotriton-${GIT_SHORT}-manylinux_2_28_x86_64-rocm${hipver}-shared.tar.gz
  cd "${AOTRITON_INSTALL_PATH}/.." && tar cz aotriton > /output/${tarfile}
fi
