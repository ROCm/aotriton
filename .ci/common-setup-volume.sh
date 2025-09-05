#!/bin/bash

function setup_source_volume() {
  SOURCE_VOLUME="$1"
  GIT_REMOTE="$2"
  LOCAL_DIR="$3"
  GIT_COMMIT="$4"
  if [ "$#" -ge 5 ]; then
    BASE_DOCKER_IMAGE="$5"
  else
    BASE_DOCKER_IMAGE="aotriton:base"
  fi
  # Download source code to volume
  docker volume create --name ${SOURCE_VOLUME}
  NEED_CLONE=0
  if docker volume ls -q -f name="${SOURCE_VOLUME}" | grep -q "${SOURCE_VOLUME}"; then
    set +e
    docker run --network=host -it --rm \
      -v ${SOURCE_VOLUME}:/src \
      -w /src/${LOCAL_DIR} \
      ${BASE_DOCKER_IMAGE} \
      bash -c "set -ex; git fetch && git checkout ${GIT_COMMIT} --recurse-submodules"
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
      bash -c "set -ex; git clone --recursive ${GIT_HTTPS_ORIGIN} && cd ${LOCAL_DIR} && git checkout ${GIT_COMMIT} && git submodule sync && git submodule update --init --recursive --force"
  fi
}
