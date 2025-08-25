#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

function help() {
  cat <<EOF
Usage releasesuite-git-head.sh [-h] [--image] [--runtime] <output directory>.
          -h: show help and exit.
     --image: build GPU images.
   --runtime: build C++ runtimes.
By default both GPU images and runtimes are built.
If either --image or --runtime is specified, the missing one will not be built.
EOF
>&2
  exit $1
}

TEMP=$(getopt -o h --longoptions image,runtime -- "$@")

if [ $? -ne 0 ]; then
  echo "Error: Invalid option." >&2
  help 1
fi

eval set -- "$TEMP"

SUITE_SELECT_IMAGE=-1
SUITE_SELECT_RUNTIME=-1
SUITE_DEFAULT_SELECTION=1

while true; do
  case "$1" in
    -h)
      help 0
      ;;
    --image)
      SUITE_SELECT_IMAGE=1
      SUITE_DEFAULT_SELECTION=0
      ;;
    --runtime)
      SUITE_SELECT_RUNTIME=1
      SUITE_DEFAULT_SELECTION=0
      ;;
    '--')
      shift
      break
      ;;
  esac
  shift
done

if [ "$#" -ne 1 ]; then
  echo "$@"
  echo 'Missing argument <output directory>.' >&2
  help 1
fi

if [ ${SUITE_SELECT_IMAGE} -lt 0 ]; then
  SUITE_SELECT_IMAGE=${SUITE_DEFAULT_SELECTION}
fi

if [ ${SUITE_SELECT_RUNTIME} -lt 0 ]; then
  SUITE_SELECT_RUNTIME=${SUITE_DEFAULT_SELECTION}
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

BASE_DOCKER_IMAGE="aotriton:base"

# build base docker image
if [ -z "$(docker images -q ${BASE_DOCKER_IMAGE} 2>/dev/null)" ]; then
  docker build --network=host -t ${BASE_DOCKER_IMAGE} -f base.Dockerfile .
fi

SOURCE_VOLUME="aotriton-src-shared"
# Download source code to volume
docker volume create --name ${SOURCE_VOLUME}
NEED_CLONE=0
if docker volume ls -q -f name="${SOURCE_VOLUME}" | grep -q "${SOURCE_VOLUME}"; then
  set +e
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src \
    -w /src/aotriton \
    ${BASE_DOCKER_IMAGE} \
    bash -c "set -ex; git fetch && git checkout ${GIT_SHORT} --recurse-submodules"
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
    bash -c "set -ex; git clone --recursive ${GIT_HTTPS_ORIGIN} && cd aotriton && git checkout ${GIT_SHORT} && git submodule sync && git submodule update --init --recursive --force"
fi

INPUT_DIR=${SCRIPT_DIR}/../dockerfile/input
OUTPUT_DIR="$1"

function build_inside() {
  rocmver="$1"
  NOIMAGE_MODE="$2"
  DOCKER_IMAGE="aotriton:buildenv-rocm${rocmver}"
  if [ -z "$(docker images -q ${DOCKER_IMAGE} 2>/dev/null)" ]; then
    docker build --network=host -t ${DOCKER_IMAGE} \
      --build-arg ROCM_VERSION_IN_URL=${rocmver} \
      -f rocm.Dockerfile .
  fi
  docker run --network=host -it --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
    --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
    --tmpfs "/root/build:exec" \
    -w / \
    ${DOCKER_IMAGE} \
    bash \
    /input/docker-script-build.sh ${llvm_hash_url} ${NOIMAGE_MODE}
}

if [ ${SUITE_SELECT_RUNTIME} -gt 0 ]; then
  # build ROCM runtime image
  for rocmver in 6.2.4 6.3.4 6.4.3 7.0_rc1
  do
    build_inside ${rocmver} ON
  done
fi

if [ ${SUITE_SELECT_IMAGE} -gt 0 ]; then
  rocmver=7.0_rc1
  build_inside ${rocmver} OFF
fi
