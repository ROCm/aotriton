#!/bin/bash

set -ex

TRITON_LLVM_HASH="$1"
NOIMAGE_MODE="$2"

rsync -a --exclude='.git' /src/aotriton/ /root/build/aotriton/
cd /root/build/aotriton/

fn="llvm-${TRITON_LLVM_HASH}-almalinux-x64.tar.gz"
if [ -f "/input/$fn" ]; then
  mkdir -p "$HOME/.triton/llvm"
  cd "$HOME/.triton/llvm"
  echo "Unpacking $fn" && tar xf "/input/$fn"
  echo -n "https://oaitriton.blob.core.windows.net/public/llvm-builds/$fn" > "llvm-${TRITON_LLVM_HASH}-almalinux-x64/version.txt"
fi

cd /root/build/
scl enable gcc-toolset-13 -- bash aotriton/.ci/build-release.sh "${NOIMAGE_MODE}"

if [ ${NOIMAGE_MODE} == "OFF" ]; then
  tarfile=aotriton-${AOTRITON_GIT_NAME}-images.tar.gz
  cd /root/build/aotriton/build/installed_dir && tar cz aotriton/lib/aotriton.images > /output/${tarfile}
else
  hipver=$(scl enable gcc-toolset-13 "cpp -I/opt/rocm/include /input/print_hip_version.h"|tail -n 1|sed 's/ //g')
  tarfile=aotriton-${AOTRITON_GIT_NAME}-manylinux_2_28_x86_64-rocm${hipver}-shared.tar.gz
  cd /root/build/aotriton/build/installed_dir && tar cz aotriton > /output/${tarfile}
fi
