#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 4 ]; then
  echo 'Missing arguments. Usage: triton-tester-build.sh <baseimage> <output dir> <arch> <triton commit> <triton wheel>' >&2
  echo '<trion wheel> should be obtained through triton-wheel-build.sh <triton commit>' >&2
  exit 1
fi

set -ex

BASE_IMAGE="$1"
OUTPUT_DIR="$2"
TARGET_ARCH="$3"
TRITON_COMMIT="$4"
TRITON_WHEEL="$5"
TRITON_SHORT=$(echo ${TRITON_COMMIT} | head -c 8)

if [ ! -f "${TRITON_WHEEL}" ]; then
  echo "triton wheel file not exists" >&2
  exit 1
fi
GIT_COMMIT=$(git rev-parse HEAD)
TRITON_WHEEL_BASE=$(basename ${TRITON_WHEEL})

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
INPUT_DIR=${SCRIPT_DIR}/../dockerfile/input
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/common-setup-volume.sh"
. "${SCRIPT_DIR}/common-git-https-origin.sh"

SOURCE_VOLUME="aotriton-src-shared"
LOCAL_DIR="aotriton"
setup_source_volume ${SOURCE_VOLUME} ${GIT_HTTPS_ORIGIN} ${LOCAL_DIR} ${GIT_COMMIT}

docker run --network=host -it --rm \
  -v ${SOURCE_VOLUME}:/src:ro \
  --mount "type=bind,source=$(realpath ${INPUT_DIR}),target=/input" \
  --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
  --mount "type=bind,source=$(realpath ${TRITON_WHEEL}),target=/aotriton-compiler/${TRITON_WHEEL_BASE}" \
  --tmpfs "/root/build:exec" \
  -w / \
  ${BASE_IMAGE} \
  bash \
  /input/docker-script-triton-tester-build.sh "${TARGET_ARCH}" "${TRITON_COMMIT}"
