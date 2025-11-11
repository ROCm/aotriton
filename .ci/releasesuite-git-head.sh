#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if ! command -v yq &> /dev/null; then
  cat <<EOF
Command 'yq' could not be found. Install it with
dnf install yq
or
snap install yq
EOF
>&2
  exit 1
fi

function help() {
  cat <<EOF
Usage releasesuite-git-head.sh [-h] [-r <ROCM ver>] [--image] [--runtime] [--yaml <yaml config file>] <output directory>.
                    -h: show help and exit.
         -r <ROCM ver>: build ROCM runtime image
               --image: build GPU images.
             --runtime: build all C++ runtimes.
         --yaml <.yml>: Use yml config file to build the release
By default both GPU images and runtimes are built.
If either --image or --runtime is specified, the missing one will not be built.

The YAML configuration file follows the format shown in docs/AltWheelExample.yaml.
However it accepts GIT SHA1 for Triton wheels instead.
The build process will
1. Build Triton wheels from the SHA1
2. Replace SHA1 with actual wheel path and use the replaced yaml file to build
   AOTriton
EOF
>&2
  exit $1
}

TEMP=$(getopt -o h,r: --longoptions image,runtime,yaml: -- "$@")

if [ $? -ne 0 ]; then
  echo "Error: Invalid option." >&2
  help 1
fi

eval set -- "$TEMP"

SUITE_SELECT_IMAGE=-1
SUITE_SELECT_RUNTIME=-1
SUITE_RUNTIME_LIST=(6.2.4 6.3.4 6.4.3 7.0.2 7.1)
CMDLIST=()
SUITE_DEFAULT_SELECTION=1
SUITE_YAML=""

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
    -r)
      SUITE_SELECT_RUNTIME=1
      SUITE_DEFAULT_SELECTION=0
      shift
      CMDLIST+=("$1")
      ;;
    --yaml)
      shift
      SUITE_YAML="$1"
      ;;
    '--')
      shift
      break
      ;;
  esac
  shift
done

if [[ ${#CMDLIST[@]} -ne 0 ]]; then
  SUITE_RUNTIME_LIST=("${CMDLIST[@]}")
fi

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

echo "SUITE_RUNTIME_LIST ${SUITE_RUNTIME_LIST[@]}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"
. "${SCRIPT_DIR}/include-altwheel.sh"

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

if [[ -n "${SUITE_YAML}" && ${SUITE_SELECT_IMAGE} -gt 0 ]]; then
  readarray -t TRITON_ALTHASH < <(yq -r '.venvs|.[]' "${SUITE_YAML}")
  mkdir -p "${INPUT_DIR}/altwheels"
  cp "${SUITE_YAML}" "${INPUT_DIR}/altwheels/tmpconfig.yaml"
  bash "${SCRIPT_DIR}/build-altwheels.sh" "${INPUT_DIR}/altwheels" "${TRITON_ALTHASH[@]}"
  replace_hash \
    "${INPUT_DIR}/altwheels/tmpconfig.yaml" \
    "${INPUT_DIR}/altwheels" \
    "/input/altwheels" \
    "${TRITON_ALTHASH[@]}"
  ALTWHEEL_CFG="/input/altwheels/tmpconfig.yaml"
else
  ALTWHEEL_CFG=""
fi

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
    /input/docker-script-build.sh ${llvm_hash_url} ${NOIMAGE_MODE} "${ALTWHEEL_CFG}"
}

if [ ${SUITE_SELECT_RUNTIME} -gt 0 ]; then
  # build ROCM runtime image
  for rocmver in "${SUITE_RUNTIME_LIST[@]}"
  do
    build_inside ${rocmver} ON
  done
fi

if [ ${SUITE_SELECT_IMAGE} -gt 0 ]; then
  rocmver=7.1
  build_inside ${rocmver} OFF
fi
