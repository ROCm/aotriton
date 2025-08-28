#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ ! -f torch/torch_version.py ]; then
  echo 'Usage: run this script under pytoch source directory to build a dev mode torch for local testing.' >&2
  exit 1
fi

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CMAKE_HIP_COMPILER_LAUNCHER=ccache

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

bdir="build-${aotriton_major}.${aotriton_minor}-pytorch-${native_arch}"
AOTRITON_INSTALLED_PREFIX="$(realpath "${SCRIPT_DIR}"/../"${bdir}"/installed_dir/aotriton)"
if [ ! -d "${AOTRITON_INSTALLED_PREFIX}" ]; then
  echo "Cannot find aotriton install directory ${AOTRITON_INSTALLED_PREFIX}" >&2
  exit 1
fi
export AOTRITON_INSTALLED_PREFIX
export PYTORCH_ROCM_ARCH=${native_arch}
python tools/amd_build/build_amd.py|grep -v skipped || true
export ROCM_PATH=/opt/rocm
USE_ROCM=1 python setup.py develop --user
