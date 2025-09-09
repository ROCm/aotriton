#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 2 ]; then
  echo 'Missing arguments. Usage: build-triton-tester.sh <target arch> <triton wheel file>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

target_arch="$1"
triton_wheel="$(realpath $2)"

bdir="build-triton_tester"
mkdir -p ${SCRIPT_DIR}/../${bdir}
cd ${SCRIPT_DIR}/../${bdir}
# Limit the number of hdims to build
AOTRITON_FLASH_BLOCK_DMODEL='48, 80, 128, 192, 224, 256'
export AOTRITON_FLASH_BLOCK_DMODEL

cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DAOTRITON_USE_LOCAL_TRITON_WHEEL="${triton_wheel}" \
  -DAOTRITON_GPU_BUILD_TIMEOUT=8 \
  -DAOTRITON_TERMINATE_WHEN_GPU_BUILD_TIMEOUT=ON \
  -DCMAKE_INSTALL_PREFIX=installed_dir/aotriton \
  -DCMAKE_BUILD_TYPE=Release \
  "-DAOTRITON_TARGET_ARCH=${target_arch}" \
  -DAOTRITON_NAME_SUFFIX=tRiToN_tEsTeR \
  -G Ninja && ninja install/strip
