#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-usage.sh"

usage() {
  cat <<'EOF' >&2
Usage: build-shim.sh [options] <target arch> [optional pre-compiled triton wheel]

The C++ shim only (AOTRITON_NOIMAGE_MODE=ON): codegen and a C++ compile, no
kernel images -- the fastest check that the tree still builds. Mold linker on.
<target arch> may be a semicolon-separated list.
EOF
  common_build_usage_options
}

common_build_usage_if_requested "$@"
. "${SCRIPT_DIR}/common-build.sh"

CB_PROG=build-shim.sh
CB_MOLD_DEFAULT=true
common_build_parse "$@"
common_build_take_arch
common_build_take_triton_wheel

common_build_run Release 123 shim \
  -DAOTRITON_NOIMAGE_MODE=ON -DAOTRITON_GPU_BUILD_TIMEOUT=0
