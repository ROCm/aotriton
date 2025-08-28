#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

build_type="pytorch"
target_arch="${native_arch}"

bdir="build-${aotriton_major}.${aotriton_minor}-${build_type}-${target_arch}"
mkdir -p "${SCRIPT_DIR}/../${bdir}"
cd "${SCRIPT_DIR}/../${bdir}"

cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX=installed_dir/aotriton \
  -DCMAKE_BUILD_TYPE=Release \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  "-DAOTRITON_TARGET_ARCH=${target_arch}" \
  -DAOTRITON_NO_PYTHON=ON \
  -G Ninja && ninja install/strip
