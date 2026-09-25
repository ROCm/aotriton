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

# Optional, defaulting to the local GPU: build-for-torch.sh now takes an arch,
# so this has to be able to name the same one. `aotriton_build_dir` is the
# formula build-for-torch.sh builds under, shared rather than restated.
target_arch="${1:-${native_arch}}"
bdir="$(aotriton_build_dir pytorch "${target_arch}")"
AOTRITON_INSTALLED_PREFIX="$(realpath -m "${SCRIPT_DIR}"/../"${bdir}"/installed_dir/aotriton)"
if [ ! -d "${AOTRITON_INSTALLED_PREFIX}" ]; then
  echo "Cannot find aotriton install directory ${AOTRITON_INSTALLED_PREFIX}" >&2
  echo "Build it first: bash .ci/build-for-torch.sh ${target_arch}" >&2
  exit 1
fi
export AOTRITON_INSTALLED_PREFIX
export PYTORCH_ROCM_ARCH=${target_arch}
python tools/amd_build/build_amd.py|grep -v skipped || true
# theRock installs ROCm inside a Python package, so `rocm-sdk path --root` is
# the only thing that knows where it is; /opt/rocm is the classical layout and
# the fallback when rocm-sdk is not installed. This used to be a hardcoded
# /opt/rocm, which on a theRock machine points the torch build at an install
# that is not there -- or, worse, at a stale one that is.
#
# An explicit ROCM_PATH from the caller wins over both, the same precedence
# build-release.sh uses.
if [ -z "${ROCM_PATH}" ]; then
  if command -v rocm-sdk &>/dev/null; then
    ROCM_PATH=$(rocm-sdk path --root 2>/dev/null) || ROCM_PATH=''
  fi
  ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
fi
export ROCM_PATH
USE_ROCM=1 python setup.py develop --user
