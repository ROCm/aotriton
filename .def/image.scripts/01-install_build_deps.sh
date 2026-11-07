#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install system packages required to build third_party/triton (e.g. LLVM's
# optional zstd-based compression support, required by Triton's CMake build).

set -ex

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libzstd-dev
rm -rf /var/lib/apt/lists/*
