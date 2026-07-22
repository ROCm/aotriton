#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

# Parse options using getopt
shim_only=false
altwheel_config=""
TEMP=$(getopt -o '' --long shim,altwheel_config: -n 'build-tune.sh' -- "$@")
if [ $? != 0 ]; then
  echo 'Usage: build-tune.sh [--shim] [--altwheel_config <yaml>] <target arch> [optional pre-compiled triton wheel]' >&2
  exit 1
fi

eval set -- "$TEMP"

while true; do
  case "$1" in
    --shim)
      shim_only=true
      shift
      ;;
    --altwheel_config)
      altwheel_config="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Internal error!" >&2
      exit 1
      ;;
  esac
done

if [ "$#" -lt 1 ]; then
  echo 'Missing arguments. Usage: build-tune.sh [--shim] [--altwheel_config <yaml>] <target arch> [optional pre-compiled triton wheel]' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-build.sh"

target_arch="$1"
shift

# Prepare arguments for common_build
if [ "$shim_only" = true ]; then
  build_args=("${target_arch}" "tune_shim_only" "-DAOTRITON_BUILD_FOR_TUNING=ON" "-DAOTRITON_NOIMAGE_MODE=ON")
else
  build_args=("${target_arch}" "tune" "-DAOTRITON_BUILD_FOR_TUNING=ON")
fi

if [ -n "$altwheel_config" ]; then
  build_args+=("-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=$(realpath "$altwheel_config")")
fi

# Add optional triton wheel argument if provided -- not when an altwheel
# config is set, which supplies the default wheel via its own .venvs.default.
if [ "$#" -ge 1 ] && [ -z "$altwheel_config" ]; then
  wheel=$(realpath "$1")
  build_args+=("-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${wheel}")
fi

# Call common_build with collected arguments
common_build "${build_args[@]}"
