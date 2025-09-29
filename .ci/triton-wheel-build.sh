#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo 'Missing arguments. Usage: triton-wheel-build.sh <python version> <triton commit>' >&2
  echo '<python version> should be the name suffix of python package from AlmaLinux8' >&2
  exit 1
fi

set -ex

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-setup-volume.sh"

PYVER="$1"
TRITON_GIT_NAME="$2"

SOURCE_VOLUME="triton-src-shared"
GIT_HTTPS_ORIGIN='https://github.com/triton-lang/triton.git'
LOCAL_DIR=triton
# TODO: deduplicate with releasesuite-git-head.sh
BASE_DOCKER_IMAGE="aotriton:base"
if [ -z "$(docker images -q ${BASE_DOCKER_IMAGE} 2>/dev/null)" ]; then
  docker build --network=host -t ${BASE_DOCKER_IMAGE} -f base.Dockerfile ${SCRIPT_DIR}
fi
BASE_DOCKER_IMAGE="aotriton:buildenv-triton_tester-py${PYVER}"
if [ -z "$(docker images -q ${BASE_DOCKER_IMAGE} 2>/dev/null)" ]; then
  docker build --network=host -t ${BASE_DOCKER_IMAGE} --build-arg PYVER=${PYVER} -f buildenv-triton_tester.Dockerfile ${SCRIPT_DIR}
fi

setup_source_volume ${SOURCE_VOLUME} ${GIT_HTTPS_ORIGIN} ${LOCAL_DIR} ${TRITON_GIT_NAME}

INPUT_DIR=${SCRIPT_DIR}/triton-patch
OUTPUT_DIR=${SCRIPT_DIR}/../dockerfile/input

function build_wheel_inside() {
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
    --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
    --tmpfs "/root/build:exec" \
    -w / \
    ${BASE_DOCKER_IMAGE} \
    bash \
    /input/docker-script-build.sh ${TRITON_COMMIT}
}

build_wheel_inside
