#!/bin/bash

function setup_source_volume() {
  local source_volume="$1"
  local git_remote="$2"
  local local_dir="$3"
  local git_commit="$4"
  if [ "$#" -ge 5 ]; then
    local base_docker_image="$5"
  else
    local base_docker_image="aotriton:base"
  fi
  # Download source code to volume
  docker volume create --name ${source_volume}
  local need_clone=0
  if docker volume ls -q -f name="${source_volume}" | grep -q "${source_volume}"; then
    set +e
    docker run --network=host -it --rm \
      -v ${source_volume}:/src \
      -w /src/${local_dir} \
      ${base_docker_image} \
      bash -c "set -ex; git fetch && git checkout ${git_commit} --recurse-submodules"
    if [ $? -ne 0 ]; then
      need_clone=1
    fi
    set -e
  fi

  if [ ${need_clone} -ne 0 ]; then
    docker run --network=host -it --rm \
      -v ${source_volume}:/src \
      -w /src \
      ${base_docker_image} \
      bash -c "set -ex; git clone --recursive ${git_https_origin} && cd ${local_dir} && git checkout ${git_commit} && git submodule sync && git submodule update --init --recursive --force"
  fi
}
