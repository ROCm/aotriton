#!/bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

function _common_build() {
  build_type="$1"
  suffix="$2"
  target_arch="$3"
  build_for="$4"
  shift 4
  bdir="build-${aotriton_major}.${aotriton_minor}-${build_for}-${target_arch}"
  mkdir -p ${SCRIPT_DIR}/../${bdir}
  (
    cd ${SCRIPT_DIR}/../${bdir};
    cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DAOTRITON_TARGET_ARCH=${target_arch} \
      -DAOTRITON_NAME_SUFFIX=${suffix} \
      "$@" \
      -G Ninja;
    ninja install/strip
  )
}

function common_build() {
  if [ "$#" -lt 2 ]; then
    echo 'common_build expects at least 2 arguments: <target arch> <build for> [cmake options...]' >&2
    exit 1
  fi
  _common_build Release "123" "$@"
}

function debug_build() {
  if [ "$#" -lt 2 ]; then
    echo 'common_build expects at least 2 arguments: <target arch> <build for> [cmake options..]' >&2
    exit 1
  fi
  _common_build Debug "123" "$@"
}
