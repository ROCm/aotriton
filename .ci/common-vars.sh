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

# theRock (rocm_sdk) wheels split their shared libs across nested
# _rocm_sdk_devel/lib/, lib/rocm_sysdeps/lib/, lib/host-math/lib/ etc.
# RPATH covers most cross-references between them, but not all (e.g.
# liblzma, openblas), so callers that don't already inherit ROCM_PATH from
# `rocm-sdk path --root` (e.g. plain venvs baked into worker images) can hit
# "cannot open shared object file" at import time. Add every nested dir that
# actually holds a .so directly, found via the sibling rocm_sdk_core package.
add_rocm_sdk_ldconfig() {
  local dirs=$(python -c "
try:
    import rocm_sdk_core
except ImportError:
    pass
else:
    from pathlib import Path
    devel_lib = Path(rocm_sdk_core.__file__).parent.parent / '_rocm_sdk_devel' / 'lib'
    if devel_lib.is_dir():
        print(':'.join(sorted({p.parent.as_posix() for p in devel_lib.rglob('*.so*')})))
" 2>/dev/null)
  if [ -n "${dirs}" ]; then
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${dirs}
  fi
}

llvm_hash_sha1=$(get_llvm_hash)
if [ -z "${llvm_hash_sha1}" ]; then
  echo "common-vars: llvm_hash is unset due to missing third_party/triton/cmake/llvm-hash.txt. Need git clone --recursive" >&2
  unset llvm_hash_sha1
else
  llvm_hash_url=$(echo ${llvm_hash_sha1}|head -c 8)
fi
