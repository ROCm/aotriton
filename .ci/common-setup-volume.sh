#!/bin/bash

function setup_source_volume() {
  local source_volume="$1"
  local git_https_origin="$2"
  local local_dir="$3"
  local git_name="$4"
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
    docker run --network=host -i --rm \
      -v ${source_volume}:/src \
      -w /src/${local_dir} \
      ${base_docker_image} \
      bash -s "${git_https_origin}" "${git_name}" << 'EOF'
set -ex
git remote set-url origin "$1" || git remote add origin "$1"
git fetch --all
git checkout -f "$2"
git submodule sync
git submodule update --init --recursive --force
EOF
    if [ $? -ne 0 ]; then
      need_clone=1
    fi
    set -e
  fi

  if [ ${need_clone} -ne 0 ]; then
    docker run --network=host -i --rm \
      -v ${source_volume}:/src \
      -w /src \
      ${base_docker_image} \
      bash -s "${git_https_origin}" "${local_dir}" "${git_name}" << 'EOF'
set -ex
rm -rf "$2"
git clone --recursive "$1" "$2"
cd "$2"
git checkout "$3"
git submodule sync
git submodule update --init --recursive --force
EOF
  fi
}
