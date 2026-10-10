#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-usage.sh"

usage() {
  cat <<'EOF' >&2
Usage: build-for-torch.sh [options] <target arch> [optional pre-compiled triton wheel]

The AOTriton PyTorch links against, for torch-build.sh to consume: NO name
suffix (this build IS torch's libaotriton_v2.so, it does not coexist with one)
and AOTRITON_NO_PYTHON=ON. No mold. <target arch> may be a semicolon-separated
list; it used to be implicit, so an existing no-argument call needs one adding.
EOF
  common_build_usage_options
}

common_build_usage_if_requested "$@"
. "${SCRIPT_DIR}/common-build.sh"

CB_PROG=build-for-torch.sh
common_build_parse "$@"
common_build_take_arch
common_build_take_triton_wheel

# The install layout torch-build.sh expects. Relative, so it lands inside the
# build directory; `:-` so an external caller's own override still wins.
export AOTRITON_INSTALL_PATH="${AOTRITON_INSTALL_PATH:-installed_dir/aotriton}"

# AOTRITON_NO_PYTHON is not just "torch has no use for pyaotriton": leaving the
# bindings on drags in find_package(Torch), which hard-errors without
# PYTORCH_ROCM_ARCH (see .tune/single/build_arch.sh for that fight).
#
# And no -DCMAKE_PREFIX_PATH. CMakeLists.txt already prefers $ENV{ROCM_PATH}
# and falls back to /opt/rocm itself, so passing /opt/rocm on the command line
# -- which this script used to do -- pins the stale install AHEAD of a theRock
# ROCM_PATH, which is the one thing that fallback is there to prevent.
common_build_run Release '' pytorch \
  -DAOTRITON_NO_PYTHON=ON -DAOTRITON_GPU_BUILD_TIMEOUT=0
