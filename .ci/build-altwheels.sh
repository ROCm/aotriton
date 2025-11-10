#!/bin/bash

set -ex

WHEEL_OUTPUT_DIR="$1"
shift
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
INPUT_DIR="${SCRIPT_DIR}/../dockerfile/input"

BASE_DOCKER_IMAGE="aotriton:base"
SOURCE_VOLUME="altwheel-src-shared"
GIT_HTTPS_ORIGIN='https://github.com/ROCm/triton.git'

docker volume create --name ${SOURCE_VOLUME}

function ensure_triton() {
  local git_hash="$1"
  local need_clone=0
  if docker volume ls -q -f name="${SOURCE_VOLUME}" | grep -q "${SOURCE_VOLUME}"; then
    set +e
    docker run --network=host -it --rm \
      -v ${SOURCE_VOLUME}:/src \
      -w /src/triton \
      ${BASE_DOCKER_IMAGE} \
      bash -c "set -ex; git fetch && git checkout ${git_hash} --recurse-submodules"
    if [ $? -ne 0 ]; then
      need_clone=1
    fi
    set -e
  fi
  if [ ${need_clone} -ne 0 ]; then
    docker run --network=host -it --rm \
      -v ${SOURCE_VOLUME}:/src \
      -w /src \
      ${BASE_DOCKER_IMAGE} \
      bash -c "set -ex; git clone --recursive ${GIT_HTTPS_ORIGIN} && cd triton && git checkout ${git_hash} && git submodule sync && git submodule update --init --recursive --force"
  fi
}

function build_wheel() {
  local input_dir="$1"
  local output_dir="$2"
  local git_hash="$3"
  local git_short=$(echo ${git_hash}|head -c 8)
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    --mount "type=bind,source=$(realpath ${input_dir}),target=/input" \
    --mount "type=bind,source=$(realpath ${output_dir}),target=/output" \
    --tmpfs "/root/build:exec" \
    -w / \
    ${BASE_DOCKER_IMAGE} \
    bash \
    /input/docker-script-build-altwheel.sh
}

for althash in "$@"; do
  ensure_triton ${althash}
  build_wheel "${INPUT_DIR}" "${WHEEL_OUTPUT_DIR}" "${althash}"
done
