#!/bin/bash

TRITON_COMMIT="$1"
TRITON_SHORT=$(echo ${TRITON_COMMIT} | head -c 8)
TMP_BRANCH="__ci_internal_use/_triton_wheel"

set -ex

# Note: only copy .git
rsync -aR /src/./triton/.git /root/build/./
cd /root/build/triton
git reset --hard
TRITON_LLVM_HASH=$(head -c 8 "cmake/llvm-hash.txt")

unset patch_file
# Patch
for f in $(ls -r /input/patch-*.sh)
do
  git switch -C ${TMP_BRANCH} ${TRITON_COMMIT}
  set +e  # Can fail
  bash /input/docker-script-patch.sh $f
  RET=$?
  set -e  # Can fail
  if [ $RET -eq 0 ]; then
    patch_file=$(basename -s .sh "$f")
    patch_date=${patch_file#patch-}
    break
  fi
  git reset --hard
done

if [[ -z "${patch_file+x}" ]]; then
  echo "All patch-*.sh file failed"
  exit 1
fi
export TRITON_WHEEL_VERSION_SUFFIX="_base_${TRITON_SHORT}_patch_${patch_date}"

fn="llvm-${TRITON_LLVM_HASH}-almalinux-x64.tar.gz"
url="https://oaitriton.blob.core.windows.net/public/llvm-builds/$fn"
function check_cached_llvm() {
  dir="$1"
  if [ -f "/$dir/$fn" ]; then
    mkdir -p "$HOME/.triton/llvm"
    cd "$HOME/.triton/llvm"
    echo "Unpacking $fn" && tar xf "$dir/$fn"
    echo -n "${url}" > "llvm-${TRITON_LLVM_HASH}-almalinux-x64/version.txt"
    return 0
  else
    return 1
  fi
}

function cached_llvm() {
  dir="$1"
  wget "$url" -O "$dir/$fn"
}

check_cached_llvm /input || check_cached_llvm /output || cache_llvm /output || check_cached_llvm

cd /root/build/triton
scl enable gcc-toolset-13 -- python -m pip wheel .
cp triton-*.whl /output
