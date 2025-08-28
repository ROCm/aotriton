#!/bin/bash

set -ex

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
root_cmake="${SCRIPT_DIR}/../CMakeLists.txt"

aotriton_major=$(grep 'set(AOTRITON_VERSION_MAJOR_INT' "${root_cmake}"|cut -d ' ' -f 2|cut -d ')' -f 1)
aotriton_minor=$(grep 'set(AOTRITON_VERSION_MINOR_INT' "${root_cmake}"|cut -d ' ' -f 2|cut -d ')' -f 1)
default_target_arch=$(grep 'set(AOTRITON_TARGET_ARCH' "${root_cmake}"|cut -d ' ' -f 2|cut -d '"' -f 2)
native_arch=$(rocm_agent_enumerator|grep -v gfx000|head -n 1)
ngpus=$(rocm_agent_enumerator|grep -v gfx000|wc -l)

if [ -f "${SCRIPT_DIR}/../third_party/triton/cmake/llvm-hash.txt" ]; then
  llvm_hash_sha1=$(cat "${SCRIPT_DIR}/../third_party/triton/cmake/llvm-hash.txt")
  llvm_hash_url=$(echo ${llvm_hash_sha1}|head -c 8)
else
  echo "common-vars: llvm_hash is unset due to missing third_party/triton/cmake/llvm-hash.txt. Need git clone --recursive" >&2
fi
