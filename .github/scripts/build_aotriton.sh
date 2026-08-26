#!/bin/bash

set -Eeuo pipefail

# ENV VARS
export DEBIAN_FRONTEND=noninteractive

# Install dependencies
apt-get update && \
apt-get install -y --no-install-recommends \
mold \
ninja-build \
cmake && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
pip install -r requirements-dev.txt

# Build AOTriton
bash .ci/build-test.sh ${PYTORCH_ROCM_ARCH}