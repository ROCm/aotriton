#!/bin/bash

if [ "$#" -ne 5 ]; then
  echo "Fatal. Must Use build.sh <Input directory> <Workspace directory> <Output Directory> <AOTriton's GIT Name> <AOTriton's Target GPU>"
  echo "Specify tmpfs as Workspace directory will mount type=tmpfs for AOTriton's build"
  echo ""
  echo "Example: bash build.sh input tmpfs output 0.7.1b \"MI300X;MI200\""
  exit 1
fi

TRITON_HASH="657ec732"
INPUT_DIR="$1"
WORKSPACE="$2"
OUTPUT_DIR="$3"
AOTRITON_GIT_NAME="$4"
AOTRITON_TARGET_GPUS="$5"

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
