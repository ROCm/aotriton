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

Builds the shippable library: no name suffix and no Python bindings, the same
two settings build-for-torch.sh uses and for the same reason.

Unlike the rest of the build-*.sh family this takes RAW cmake options rather
than named flags, because runc-manylinux-build-tar.sh assembles them that way.
That is why it does not share their option parser -- getopt would reject a bare
-D. It does share the cmake invocation itself.

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
. "${SCRIPT_DIR}/common-build.sh"

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
# A fixed `build`, not the family's build-<version>-<profile>-<arch>: the
# release tarball's layout is keyed to this path. Expressed as the same two
# environment variables _common_build already honours, so an external caller
# (runc-manylinux-build-tar.sh) still overrides both.
export AOTRITON_BUILD_PATH="${AOTRITON_BUILD_PATH:-${source_dir}/build}"
export AOTRITON_INSTALL_PATH="${AOTRITON_INSTALL_PATH:-${AOTRITON_BUILD_PATH}/installed_dir/aotriton}"

if [ -z "${ROCM_PATH}" ]; then
  export ROCM_PATH=$(hipconfig --rocmpath 2>/dev/null)
  if [ -z "${ROCM_PATH}" ]; then
    echo "Error: ROCM_PATH is empty. hipconfig --rocmpath failed." >&2
    exit 1
  fi
fi

_common_build Release '' "${target_arch}" release \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
  -DPYTHON_EXECUTABLE="${python_exec}" \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  -DAOTRITON_NO_PYTHON=ON \
  -DAOTRITON_NOIMAGE_MODE=${noimage} \
  "$@"
