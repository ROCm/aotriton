#!/bin/bash

set -ex

cd /src/triton
TRITON_COMMIT=$(git rev-parse HEAD)
TRITON_SHORT=$(git rev-parse --short=8 HEAD)
TRITON_LLVM_COMMIT=$(cat cmake/llvm-hash.txt)
TRITON_LLVM_SHORT=$(head -c 8 "cmake/llvm-hash.txt")

# Note: only copy .git
rsync -aR /src/./triton/.git /root/build/./
cd /root/build/triton
git checkout ${TRITON_COMMIT}
git reset --hard

FN="llvm-${TRITON_LLVM_SHORT}-almalinux-x64.tar.gz"
URL="https://oaitriton.blob.core.windows.net/public/llvm-builds/$FN"
function check_cached_llvm() {
  local d="$1"
  if [ -f "$d/$FN" ]; then
    mkdir -p "$HOME/.triton/llvm"
    cd "$HOME/.triton/llvm"
    echo "Unpacking $FN" && tar xf "$d/$FN"
    echo -n "${URL}" > "llvm-${TRITON_LLVM_SHORT}-almalinux-x64/version.txt"
    return 0
  else
    return 1
  fi
}

function cache_llvm() {
  local d="$1"
  wget "$url" -O "$d/$fn"
}

check_cached_llvm /input || check_cached_llvm /output || cache_llvm /output || check_cached_llvm

cd /root/build/triton
scl enable gcc-toolset-13 -- python -m pip wheel .
cp triton-*.whl /output
