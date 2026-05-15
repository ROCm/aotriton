#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo 'Missing arguments. Usage: build-release.sh <noimage mode> [arch list string] [cmake options ...]' >&2
  echo 'Put "ALL" to [arch list string] to build all architectures'
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

echo "${TRITON_WHEEL_VERSION_SUFFIX}"
python_exec="/usr/bin/python3.11"
noimage="$1"
shift

if [ "$#" -ge 1 ]; then
  target_arch="$1"
  shift
else
  target_arch="${default_target_arch}"
fi
if [ "${target_arch}" = "ALL" ]; then
  target_arch="${default_target_arch}"
fi

source_dir="${SCRIPT_DIR}/.."
if [ -n "${AOTRITON_BUILD_PATH}" ]; then
  bdir="${AOTRITON_BUILD_PATH}"
else
  bdir="${source_dir}/build"
fi
if [ -n "${AOTRITON_INSTALL_PATH}" ]; then
  install_prefix="${AOTRITON_INSTALL_PATH}"
else
  install_prefix="${bdir}/installed_dir/aotriton"
fi
mkdir -p "${bdir}"

if [ -z "${ROCM_PATH}" ]; then
  export ROCM_PATH=$(hipconfig --rocmpath 2>/dev/null)
  if [ -z "${ROCM_PATH}" ]; then
    echo "Error: ROCM_PATH is empty. hipconfig --rocmpath failed." >&2
    exit 1
  fi
fi

(cd "${bdir}" && cmake "${source_dir}" -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DPYTHON_EXECUTABLE="${python_exec}" \
  -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  "-DAOTRITON_TARGET_ARCH=${target_arch}" \
  -DAOTRITON_NO_PYTHON=ON \
  -DAOTRITON_NOIMAGE_MODE=${noimage} \
  -G Ninja \
  "$@" \
  && ninja install/strip)
