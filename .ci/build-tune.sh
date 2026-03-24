#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo 'Missing arguments. Usage: build-tune.sh <target arch> [optional pre-compiled triton wheel]' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-build.sh"

target_arch="$1"
shift

# Prepare arguments for common_build
build_args=("${target_arch}" "tune" "-DAOTRITON_BUILD_FOR_TUNING=ON")

# Add optional triton wheel argument if provided
if [ "$#" -ge 1 ]; then
  wheel=$(realpath "$1")
  build_args+=("-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${wheel}")
fi

# Call common_build with collected arguments
common_build "${build_args[@]}"
