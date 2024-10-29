#!/bin/bash

if [ "$#" -ne 5 ]; then
  echo "Fatal. Must Use build.sh <Input directory> <Workspace directory> <Output Directory> <AOTriton's GIT Name> <AOTriton's Target GPU>"
  echo "Specify tmpfs as Workspace directory will mount type=tmpfs for AOTriton's build"
  echo ""
  echo "Example: bash build.sh input tmpfs output 0.7.1b \"MI300X;MI200\""
  exit 1
fi

INPUT_DIR="$1"
WORKSPACE="$2"
OUTPUT_DIR="$3"
AOTRITON_GIT_NAME="$4"
AOTRITON_TARGET_GPUS="$5"

if [ -z ${TRITON_HASH+x} ]; then
  echo "Guessing Triton's Hash from AOTriton's Git tag"
  case "${AOTRITON_GIT_NAME}" in
    0.7b)
      TRITON_HASH="657ec732"
      ;;
    0.7.*b)
      TRITON_HASH="657ec732"
      ;;
    0.8b)
      TRITON_HASH="b5cc222d"
      ;;
    *)
      echo "Cannot guess Triton's hash from tag. Set env var TRITON_HASH to proceed"
      echo 'The value can be get from `head -c 8 third_party/triton/cmake/llvm-hash.txt`'
      echo '(Must switch branch and update submodule first)'
      exit
      ;;
  esac
fi

DOCKER_IMAGE=aotriton:manylinux_2_28-buildenv-tiny  # TODO: FIXME

if [ -z "$(docker images -q ${DOCKER_IMAGE} 2> /dev/null)" ]; then
  docker build -t ${DOCKER_IMAGE} -f manylinux_2_28.Dockerfile .
fi

if [ "$WORKSPACE" == "tmpfs" ]; then
  workspace_option1="--tmpfs"
  workspace_option2="/root/build:exec"
else
  workspace_option1="--mount"
  workspace_option2="type=bind,target=/root/build,source=$(realpath $WORKSPACE)"
fi

echo /input/install.sh ${TRITON_HASH} ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_GPUS}"

docker run --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
  --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
  ${workspace_option1} ${workspace_option2} \
  -w / \
  -it ${DOCKER_IMAGE} \
  bash \
  /input/install.sh ${TRITON_HASH} ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_GPUS}"
