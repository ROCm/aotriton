#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: releasesuite-git-head.sh <output dir>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

GIT_ORIGIN=$(git remote get-url origin)
if [[ ${GIT_ORIGIN} == "https://"* ]]; then
  GIT_HTTPS_ORIGIN=${GIT_ORIGIN}
else
  git_path=$(echo "${GIT_ORIGIN}"|cut -d ":" -f 2)
  GIT_HTTPS_ORIGIN="https://github.com/${git_path}"
fi

GIT_SHORT=$(git rev-parse --short=12 HEAD)

export NOIMAGE_MODE=ON

BASE_DOCKER_IMAGE="aotriton:base"

# build base docker image
if [ -z "$(docker images -q ${BASE_DOCKER_IMAGE} 2>/dev/null)" ]; then
  docker build --network=host -t ${BASE_DOCKER_IMAGE} -f base.Dockerfile .
fi

SOURCE_VOLUME="aotriton-src-${GIT_SHORT}"
# Download source code to volume
docker volume create --name ${SOURCE_VOLUME}
NEED_CLONE=0
if docker volume ls -q -f name="${SOURCE_VOLUME}" | grep -q "${SOURCE_VOLUME}"; then
  set +e
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src \
    -w /src/aotriton \
    ${BASE_DOCKER_IMAGE} \
    bash -c 'git fetch && git checkout ${GIT_SHORT} --recurse-submodules'
  if [ $? -ne 0 ]; then
    NEED_CLONE=1
  fi
  set -e
fi

if [ ${NEED_CLONE} -ne 0 ]; then
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src \
    -w /src \
    ${BASE_DOCKER_IMAGE} \
    bash -c "git clone --recursive ${GIT_HTTPS_ORIGIN} && cd aotriton && git checkout ${GIT_SHORT} && git submodule sync && git submodule update --init --recursive --force"
fi

INPUT_DIR=${SCRIPT_DIR}/../dockerfile/input
OUTPUT_DIR="$1"

function build_inside() {
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
    --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
    --tmpfs "/root/build:exec" \
    -w / \
    ${DOCKER_IMAGE} \
    bash \
    /input/docker-script-build.sh ${TRITON_LLVM_HASH} ON
}

# build ROCM runtime image
for rocmver in 6.2.4 6.3.4 6.4.3 7.0_rc1
do
  DOCKER_IMAGE="aotriton:buildenv-rocm${rocmver}"
  if [ -z "$(docker images -q ${DOCKER_IMAGE} 2>/dev/null)" ]; then
    docker build --network=host -t ${DOCKER_IMAGE} \
      --build-arg ROCM_VERSION_IN_URL=${rocmver} \
      -f rocm.Dockerfile .
  fi
  build_inside ${DOCKER_IMAGE} ON
done

# Automatically use last image
build_inside ${DOCKER_IMAGE} OFF
