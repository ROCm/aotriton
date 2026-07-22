#!/bin/bash

set -ex

get_root_cmake() {
  local script_dir="$(dirname "${BASH_SOURCE[0]}")"
  echo "${script_dir}/../CMakeLists.txt"
}
root_cmake=$(get_root_cmake)

aotriton_major=$(grep '^set(AOTRITON_VERSION_MAJOR_INT' "${root_cmake}"|cut -d ' ' -f 2|cut -d ')' -f 1)
aotriton_minor=$(grep '^set(AOTRITON_VERSION_MINOR_INT' "${root_cmake}"|cut -d ' ' -f 2|cut -d ')' -f 1)
default_target_arch=$(grep '^set(AOTRITON_TARGET_ARCH' "${root_cmake}"|cut -d ' ' -f 2|cut -d '"' -f 2)
native_arch=$(rocm_agent_enumerator|grep -v gfx000|head -n 1)
ngpus=$(rocm_agent_enumerator|grep -v gfx000|wc -l)

get_llvm_hash() {
  local script_dir="$(dirname "${BASH_SOURCE[0]}")"
  if [ -f "${script_dir}/../third_party/triton/cmake/llvm-hash.txt" ]; then
    cat "${script_dir}/../third_party/triton/cmake/llvm-hash.txt"
  else
    echo ""
  fi
}
add_torch_ldconfig() {
  local torch_lib=$(python -c "import torch; from pathlib import Path; print((Path(torch.__file__).parent/'lib').as_posix())")
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${torch_lib}
}

# theRock (rocm_sdk) ships some libs (e.g. liblzma, openblas) as private,
# renamed copies under lib/rocm_sysdeps/lib that RPATH doesn't always resolve
# -- an acknowledged upstream packaging bug (see ROCm Core SDK 7.12.0 release
# notes) whose sanctioned workaround is adding that dir to LD_LIBRARY_PATH.
# No-op when rocm-sdk isn't installed (e.g. classical ROCm).
add_rocm_sdk_ldconfig() {
  if ! command -v rocm-sdk &>/dev/null; then
    return
  fi
  local rocm_path
  rocm_path=$(rocm-sdk path --root 2>/dev/null) || return
  if [ -d "${rocm_path}/lib/rocm_sysdeps/lib" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${rocm_path}/lib/rocm_sysdeps/lib"
  fi
}

llvm_hash_sha1=$(get_llvm_hash)
if [ -z "${llvm_hash_sha1}" ]; then
  echo "common-vars: llvm_hash is unset due to missing third_party/triton/cmake/llvm-hash.txt. Need git clone --recursive" >&2
  unset llvm_hash_sha1
else
  llvm_hash_url=$(echo ${llvm_hash_sha1}|head -c 8)
fi
