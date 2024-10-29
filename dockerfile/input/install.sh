#!/bin/bash

TRITON_HASH=$1
AOTRITON_GIT_NAME="$2"
AOTRITON_TARGET_GPUS="$3"

echo /input/install_triton.sh ${TRITON_HASH}
bash /input/install_triton.sh ${TRITON_HASH}
echo /input/install_aotriton.sh ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_GPUS}"
bash /input/install_aotriton.sh ${AOTRITON_GIT_NAME} "${AOTRITON_TARGET_GPUS}"
