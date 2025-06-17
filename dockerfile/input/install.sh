#!/bin/bash

TRITON_LLVM_HASH=$1
AOTRITON_GIT_NAME="$2"
AOTRITON_TARGET_ARCH="$3"
NOIMAGE_MODE="$4"

if [[ -z "${AOTRITON_GIT_NAME}" ]]; then
  echo 'Must define AOTRITON_GIT_NAME environment variable'
  exit 1
fi

if [ "$NOIMAGE_MODE" = "OFF" ]; then
  echo /input/install_triton.sh ${TRITON_LLVM_HASH}
  bash /input/install_triton.sh ${TRITON_LLVM_HASH}
fi
echo /input/install_aotriton.sh ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_ARCH}" "${NOIMAGE_MODE}"
bash /input/install_aotriton.sh ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_ARCH}" "${NOIMAGE_MODE}"
