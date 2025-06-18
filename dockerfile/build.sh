#!/bin/bash

if [ "$#" -ne 5 ]; then
  echo 'Fatal. Must Use build.sh <Input directory> <Workspace directory> <Output Directory> <AOTriton GIT Name> <AOTriton Target GPUs>
Specify tmpfs as Workspace directory will mount type=tmpfs instead.

Example: bash build.sh input tmpfs output 0.7.1b "gfx942;gfx90a"

CAVEAT: tmpfs is only recommended for systems with > 100GiB physical memory.

Supported Environment Variables

  TRITON_LLVM_HASH: Override the LLVM hash. Useful when tags are not supported
                    or git commit sha1 is used.
  NOIMAGE_MODE=ON: Pass -DAOTRITON_NOIMAGE_MODE=ON to AOTriton build.

Example Usage of Environment Variables:

  NOIMAGE_MODE=ON TRITON_LLVM_HASH=bd9145c8 bash build.sh input tmpfs output 3c542918a0 "gfx90a;gfx942;gfx1201"
'
  exit 1
fi

INPUT_DIR="$1"
WORKSPACE="$2"
OUTPUT_DIR="$3"
AOTRITON_GIT_NAME="$4"
AOTRITON_TARGET_ARCH="$5"

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
      TRITON_LLVM_HASH="bd9145c8"
      ;;
    0.8.*b)
      TRITON_LLVM_HASH="bd9145c8"
      ;;
    0.9b)
      TRITON_LLVM_HASH="86b69c31"
      ;;
    0.10b)
      TRITON_LLVM_HASH="3c709802"
      ;;
    *)
      echo "Unknown AOTRITON_GIT_NAME ${AOTRITON_GIT_NAME}. Please set TRITON_LLVM_HASH explicitly."
      exit
      ;;
  esac
fi

if [ -z ${AOTRITON_GIT_URL+x} ]; then
  AOTRITON_GIT_URL="https://github.com/ROCm/aotriton.git"
fi

if [ -z ${NOIMAGE_MODE+x} ]; then
  # default value
  NOIMAGE_MODE="OFF"
fi

if [ -z ${DELETE_ONCE_COMPLETE_OPTION+x} ]; then
  # default value
  DELETE_ONCE_COMPLETE_OPTION=""
fi

if [ -z ${AOTRITON_TARBALL_SHARD+x} ]; then
  # default value
  AOTRITON_TARBALL_SHARD_OPTION=""
else
  AOTRITON_TARBALL_SHARD_OPTION="--env AOTRITON_TARBALL_SHARD=${AOTRITON_TARBALL_SHARD}"
fi

if [ -z ${DOCKER_IMAGE+x} ]; then
  DOCKER_IMAGE=aotriton:manylinux_2_28-buildenv-tiny
fi

if [ -z "$(docker images -q ${DOCKER_IMAGE} 2> /dev/null)" ]; then
  if [ -z ${AMDGPU_INSTALLER+x} ]; then
    build_options=""
  else
    build_options="--build-arg amdgpu_installer=${AMDGPU_INSTALLER}"
  fi
  if [ -z ${AMDGPU_INSTALLER_SELECT_VERSION+x} ]; then
    version_selection="true"
  else
    version_selection="${AMDGPU_INSTALLER_SELECT_VERSION}"
  fi
  echo "docker build ${build_options} ${version_selection} -t ${DOCKER_IMAGE} -f manylinux_2_28.Dockerfile ."
  docker build ${build_options} \
    --build-arg amdgpu_installer_select_version="${version_selection}" \
    -t ${DOCKER_IMAGE} \
    -f manylinux_2_28.Dockerfile .
fi

if [ "$WORKSPACE" == "tmpfs" ]; then
  workspace_option1="--tmpfs"
  workspace_option2="/root/build:exec"
else
  workspace_option1="--mount"
  workspace_option2="type=bind,target=/root/build,source=$(realpath $WORKSPACE)"
fi

echo /input/install.sh ${TRITON_LLVM_HASH} ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_ARCH}"

docker run --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
  --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
  ${workspace_option1} "${workspace_option2}" \
  -w / \
  --env AOTRITON_GIT_URL=${AOTRITON_GIT_URL} \
  ${AOTRITON_TARBALL_SHARD_OPTION} \
  ${DELETE_ONCE_COMPLETE_OPTION} \
  -it \
  ${DOCKER_IMAGE} \
  bash \
  /input/install.sh ${TRITON_LLVM_HASH} ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_ARCH}" ${NOIMAGE_MODE}
