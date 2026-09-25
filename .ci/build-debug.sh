#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-usage.sh"

usage() {
  cat <<'EOF' >&2
Usage: build-debug.sh [options] <target arch> [optional pre-compiled triton wheel]

The testing build with CMAKE_BUILD_TYPE=Debug. Shares build-test.sh's build
directory, so keep one or the other. No mold. <target arch> may be a
semicolon-separated list.
EOF
  common_build_usage_options
}

common_build_usage_if_requested "$@"
. "${SCRIPT_DIR}/common-build.sh"

CB_PROG=build-debug.sh
common_build_parse "$@"
common_build_take_arch
common_build_take_triton_wheel

# "test", not "debug": run-test.sh globs for build-<version>-test-*<arch> and
# errors on more than one hit, so a debug build has to land where it will look.
common_build_run Debug 123 test -DAOTRITON_GPU_BUILD_TIMEOUT=0
