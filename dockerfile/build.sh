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

if [ -z ${TRITON_LLVM_HASH+x} ]; then
  echo "Guessing Triton's Hash from AOTriton's Git tag"
  case "${AOTRITON_GIT_NAME}" in
    0.7b)
      TRITON_LLVM_HASH="657ec732"
      ;;
    0.7.*b)
      TRITON_LLVM_HASH="657ec732"
      ;;
    0.8b)
      TRITON_LLVM_HASH="b5cc222d"
      ;;
    *)
      echo "Only 0.7b, 0.7.1b and 0.8b are supported right now"
      exit
      ;;
  esac
fi

DOCKER_IMAGE=aotriton:manylinux_2_28-buildenv-tiny

if [ -z "$(docker images -q ${DOCKER_IMAGE} 2> /dev/null)" ]; then
  if [ -z ${AMDGPU_INSTALLER+x} ]; then
    build_options="--build-arg amdgpu_installer=${AMDGPU_INSTALLER}"
  else
    build_options=""
  fi
  docker build ${build_options} -t ${DOCKER_IMAGE} -f manylinux_2_28.Dockerfile .
fi

if [ "$WORKSPACE" == "tmpfs" ]; then
  workspace_option1="--tmpfs"
  workspace_option2="/root/build:exec"
else
  workspace_option1="--mount"
  workspace_option2="type=bind,target=/root/build,source=$(realpath $WORKSPACE)"
fi

echo /input/install.sh ${TRITON_LLVM_HASH} ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_GPUS}"

docker run --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
  --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
  ${workspace_option1} "${workspace_option2}" \
  -w / \
  -it ${DOCKER_IMAGE} \
  bash \
  /input/install.sh ${TRITON_LLVM_HASH} ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_GPUS}"
