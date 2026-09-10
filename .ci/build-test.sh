#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

usage() {
  echo 'Usage: build-test.sh [--database_root <dir>] [--flydsl_kernel_root <dir>] [--flydsl_wheel <whl>] [--name_suffix <suffix>] [--no_mold] [--altwheel_config <yaml>] <target arch> [optional pre-compiled triton wheel]' >&2
  echo '<target arch> can be semicolon separated list of arches.' >&2
  echo '' >&2
  echo '--flydsl_wheel is OPTIONAL and installs a flydsl wheel from local disk into' >&2
  echo 'the build venv instead of the version pinned by' >&2
  echo 'third_party/flydsl-compiler.txt. Use it for an unreleased build -- one' >&2
  echo 'carrying a compiler patch, say. Once that build is published, move the pin' >&2
  echo 'instead. The wheel must match the build venv Python: flydsl wheels are' >&2
  echo 'CPython-ABI specific.' >&2
  echo '' >&2
  echo 'The compiler wheel and the kernel source tree are independent pins, so a' >&2
  echo 'patched wheel may or may not want a matching --flydsl_kernel_root.' >&2
  echo '' >&2
  echo '--flydsl_kernel_root is OPTIONAL and points the flyc backend at an existing' >&2
  echo 'FlyDSL source tree. Left unset, CMake shallow-clones the ref named by' >&2
  echo 'third_party/flydsl-kernel.txt into <build dir>/flydsl and points' >&2
  echo 'AOTRITON_FLYDSL_KERNEL_ROOT at that, so no build needs this flag to' >&2
  echo 'configure. Pass it to build against local FlyDSL kernel changes.' >&2
  echo '' >&2
  echo 'No head-dim range needs this flag. The gfx950 kernels above head_dim 128' >&2
  echo 'require an LDS-transpose emitter the pinned FlyDSL release does not have,' >&2
  echo 'but those five emitters are vendored in' >&2
  echo 'modules/flash/flyc/fmha_dualwave_gfx950.py, so the default clone builds' >&2
  echo 'them correctly. See docs/FlyDSL.md.' >&2
}

TEMP=$(getopt -o '' --long database_root:,flydsl_kernel_root:,flydsl_wheel:,name_suffix:,no_mold,altwheel_config: -n 'build-test.sh' -- "$@")
if [ $? != 0 ]; then
  usage
  exit 1
fi

eval set -- "$TEMP"

database_root=""
flydsl_kernel_root=""
flydsl_wheel=""
name_suffix=""
no_mold=false
altwheel_config=""
while true; do
  case "$1" in
    --database_root)
      database_root="$2"
      shift 2
      ;;
    --flydsl_kernel_root)
      flydsl_kernel_root="$2"
      shift 2
      ;;
    --flydsl_wheel)
      flydsl_wheel="$2"
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

if [ -n "$flydsl_kernel_root" ]; then
  build_args+=("-DAOTRITON_FLYDSL_KERNEL_ROOT=$(realpath "$flydsl_kernel_root")")
fi

# realpath is what satisfies cmake's absolute-path requirement on this variable.
if [ -n "$flydsl_wheel" ]; then
  build_args+=("-DAOTRITON_USE_LOCAL_FLYDSL_WHEEL=$(realpath "$flydsl_wheel")")
fi

if [ -n "$name_suffix" ]; then
  export AOTRITON_NAME_SUFFIX_OVERRIDE="$name_suffix"
fi

if [ -n "$altwheel_config" ]; then
  build_args+=("-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=$(realpath "$altwheel_config")")
fi

# Not when an altwheel config is set. The two are mutually exclusive at the
# cmake level (fatal error if both given): the altwheel config's own
# .venvs.default supplies the main venv's wheel too, so no separate flag is
# needed here.
if [ "$#" -ge 1 ] && [ -z "$altwheel_config" ]; then
  wheel=$(realpath "$1")
  build_args+=("-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${wheel}")
fi

common_build "${build_args[@]}"
