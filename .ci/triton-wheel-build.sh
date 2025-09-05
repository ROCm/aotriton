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
. "${SCRIPT_DIR}/common-setup-volume.sh"

TRITON_COMMIT="$1"

SOURCE_VOLUME="triton-src-shared"
GIT_HTTPS_ORIGIN='https://github.com/triton-lang/triton.git'
LOCAL_DIR=triton

setup_source_volume ${SOURCE_VOLUME} ${GIT_HTTPS_ORIGIN} ${LOCAL_DIR} ${TRITON_COMMIT}

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
