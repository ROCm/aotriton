#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

usage() {
  echo 'Usage: build-test.sh [--database_root <dir>] [--name_suffix <suffix>] [--no_mold] [--altwheel_config <yaml>] <target arch> [optional pre-compiled triton wheel]' >&2
  echo '<target arch> can be semicolon separated list of arches.' >&2
}

TEMP=$(getopt -o '' --long database_root:,name_suffix:,no_mold,altwheel_config: -n 'build-test.sh' -- "$@")
if [ $? != 0 ]; then
  usage
  exit 1
fi

eval set -- "$TEMP"

database_root=""
name_suffix=""
no_mold=false
altwheel_config=""
while true; do
  case "$1" in
    --database_root)
      database_root="$2"
      shift 2
      ;;
    --name_suffix)
      name_suffix="$2"
      shift 2
      ;;
    --no_mold)
      no_mold=true
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
  usage
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-build.sh"

target_arch="$1"
shift

build_args=("${target_arch}" "test" -DAOTRITON_GPU_BUILD_TIMEOUT=0)
if [ "$no_mold" = false ]; then
  build_args+=(-DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold")
fi

if [ -n "$database_root" ]; then
  build_args+=("-DAOTRITON_TUNING_DATABASE_ROOT=${database_root}")
fi

if [ -n "$name_suffix" ]; then
  export AOTRITON_NAME_SUFFIX_OVERRIDE="$name_suffix"
fi

if [ -n "$altwheel_config" ]; then
  build_args+=("-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=$(realpath "$altwheel_config")")
fi

# Not when an altwheel config is set, which supplies the default wheel via
# its own .venvs.default.
if [ "$#" -ge 1 ] && [ -z "$altwheel_config" ]; then
  wheel=$(realpath "$1")
  build_args+=("-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${wheel}")
fi

common_build "${build_args[@]}"
