#!/bin/bash

set -ex

TARGET_ARCH="$1"
TRITON_COMMIT="$2"
TRITON_WHEEL=$(realpath /aotriton-compiler/*.whl)
TRITON_SHORT=$(echo "${TRITON_COMMIT}"|head -n 8)
TRITON_SHORT12=$(echo "${TRITON_COMMIT}"|head -n 12)

rsync -a --exclude='.git' /src/aotriton/ /root/build/aotriton/
cd /src/aotriton/
GIT_FULL=$(git rev-parse HEAD)
GIT_SHORT=$(git rev-parse --short=12 HEAD)

cd /root/build/aotriton
export AOTRITON_CI_SUPPLIED_SHA1=${GIT_FULL}
python -m pip install /*.whl
bash .ci/build-triton-tester.sh "${TARGET_ARCH}" "${TRITON_WHEEL}"

cd /root/build/aotriton/build-triton_tester/install_dir/
tar c aotriton | gzip --fast > /output/aotriton-triton_tester-${TRITON_SHORT12}.tar.gz
