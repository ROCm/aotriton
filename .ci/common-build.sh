#!/bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

function _common_build() {
  target_arch="$1"
  build_type="$2"
  cmake_option0="$3"
  bdir="build-${aotriton_major}.${aotriton_minor}-${build_type}-${target_arch}"
  mkdir -p ${SCRIPT_DIR}/../${bdir}
  (
    cd ${SCRIPT_DIR}/../${bdir};
    cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir \
      -DCMAKE_BUILD_TYPE=$4 \
      -DAOTRITON_TARGET_ARCH=${target_arch} \
      ${cmake_option0} \
      -DAOTRITON_NAME_SUFFIX=123 -G Ninja;
    ninja install/strip
  )
}

function common_build() {
  if [ "$#" -ne 3 ]; then
    echo 'common_build expects 3 arguments: <target arch> <build type> <cmake option>' >&2
    exit 1
  fi
  _common_build "$@" Release
}

function debug_build() {
  if [ "$#" -ne 3 ]; then
    echo 'common_build expects 3 arguments: <target arch> <build type> <cmake option>' >&2
    exit 1
  fi
  _common_build "$@" Debug
}
