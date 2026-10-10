#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-usage.sh"

usage() {
  cat <<'EOF' >&2
Usage: build-tune.sh [--shim] [options] <target arch> [optional pre-compiled triton wheel]

The tuning build: every GPU kernel, AOTRITON_BUILD_FOR_TUNING=ON, no mold.
<target arch> may be a semicolon-separated list.

  --shim  C++ shim only (AOTRITON_NOIMAGE_MODE=ON), into its own build dir
EOF
  common_build_usage_options
}

common_build_usage_if_requested "$@"
. "${SCRIPT_DIR}/common-build.sh"

CB_PROG=build-tune.sh
CB_EXTRA_LONGOPTS=shim
common_build_parse "$@"
common_build_take_arch
common_build_take_triton_wheel

if [ "${CB_OPT[shim]:-false}" = true ]; then
  common_build_run Release 123 tune_shim_only \
    -DAOTRITON_BUILD_FOR_TUNING=ON -DAOTRITON_NOIMAGE_MODE=ON
else
  common_build_run Release 123 tune -DAOTRITON_BUILD_FOR_TUNING=ON
fi
