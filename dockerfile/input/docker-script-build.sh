#!/bin/bash

set -ex

TRITON_LLVM_HASH="$1"
NOIMAGE_MODE="$2"

rsync -a --exclude='.git' /src/aotriton/ /root/build/aotriton/
cd /src/aotriton/
GIT_FULL=$(git rev-parse HEAD)
GIT_SHORT=$(git rev-parse --short=12 HEAD)
cd /src/aotriton/third_party/triton
TRITON_SHORT=$(git rev-parse --short=12 HEAD)
export TRITON_WHEEL_VERSION_SUFFIX="+git${TRITON_SHORT}"
hipver=$(scl enable gcc-toolset-13 "cpp -I/opt/rocm/include /input/print_hip_version.h"|tail -n 1|sed 's/ //g')

if [ ${NOIMAGE_MODE} == "OFF" ]; then
  fn="llvm-${TRITON_LLVM_HASH}-almalinux-x64.tar.gz"
  if [ -f "/input/$fn" ]; then
    mkdir -p "$HOME/.triton/llvm"
    cd "$HOME/.triton/llvm"
    echo "Unpacking $fn" && tar xf "/input/$fn"
    echo -n "https://oaitriton.blob.core.windows.net/public/llvm-builds/$fn" > "llvm-${TRITON_LLVM_HASH}-almalinux-x64/version.txt"
  fi
fi

cd /root/build/
export AOTRITON_CI_SUPPLIED_SHA1=${GIT_FULL}
scl enable gcc-toolset-13 -- bash aotriton/.ci/build-release.sh "${NOIMAGE_MODE}"

if [ ${NOIMAGE_MODE} == "OFF" ]; then
  tarbase=aotriton-${GIT_SHORT}-images
  cd /root/build/aotriton/build/installed_dir
  for d in $(ls aotriton/lib/aotriton.images/); do
    tarfile=${tarbase}-$d.tar.gz
    tar cz "aotriton/lib/aotriton.images/$d" > /output/${tarfile}
  done
else
  tarfile=aotriton-${GIT_SHORT}-manylinux_2_28_x86_64-rocm${hipver}-shared.tar.gz
  cd /root/build/aotriton/build/installed_dir && tar cz aotriton > /output/${tarfile}
fi
