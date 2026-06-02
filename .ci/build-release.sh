#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

function usage() {
  cat <<EOF >&2
Usage: build-release.sh <noimage_mode> [arch_list] [cmake_options ...]
  noimage_mode   ON|OFF — passed as -DAOTRITON_NOIMAGE_MODE
  arch_list      space-separated GPU arch list, or ALL for all architectures
  cmake_options  extra -D flags forwarded to cmake
                 e.g. -DAOTRITON_USE_LOCAL_TRITON_WHEEL=<path>
                      -DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=<yaml>

Environment variables:
  AOTRITON_BUILD_PATH   cmake build directory; defaults to <source>/build
  AOTRITON_INSTALL_PATH cmake install prefix; defaults to
                        \$AOTRITON_BUILD_PATH/installed_dir/aotriton
                        When called from runc-manylinux-build-tar.sh this is
                        derived from AOTRITON_INSTALL_PREFIX as
                        \$AOTRITON_INSTALL_PREFIX/aotriton
  ROCM_PATH             ROCm installation root; auto-detected via hipconfig if unset
EOF
  exit 1
}

if [ "$#" -lt 1 ]; then
  usage
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

source_dir=$(realpath ${SCRIPT_DIR}/..)
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
