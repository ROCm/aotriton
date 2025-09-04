#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  echo 'Missing arguments. Usage: triton-wheel-build.sh <triton commit>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

GIT_HTTPS_ORIGIN='https://github.com/triton-lang/triton.git'
TRITON_COMMIT="$1"

BASE_DOCKER_IMAGE="aotriton:base"
SOURCE_VOLUME="triton-src-shared"
docker volume create --name ${SOURCE_VOLUME}
NEED_CLONE=0
if docker volume ls -q -f name="${SOURCE_VOLUME}" | grep -q "${SOURCE_VOLUME}"; then
  set +e
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src \
    -w /src/triton \
    ${BASE_DOCKER_IMAGE} \
    bash -c "set -ex; git fetch && git checkout ${TRITON_COMMIT} --recurse-submodules"
  RET=$?
  set -e
  if [ $RET -ne 0 ]; then
    NEED_CLONE=1
  fi
fi

if [ ${NEED_CLONE} -ne 0 ]; then
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src \
    -w /src \
    ${BASE_DOCKER_IMAGE} \
    bash -c "set -ex; git clone --recursive ${GIT_HTTPS_ORIGIN} && cd triton && git checkout ${TRITON_COMMIT} && git submodule sync && git submodule update --init --recursive --force"
fi

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
