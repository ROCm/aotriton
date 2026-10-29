#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim build of AOTriton 0.10b: <src_dir> <install_dir> <arch>. See
# lib/shim.sh for the contract.
#
# Like 0.9.2b, this release publishes no separate images package: its images
# are inside the one jumbo tarball.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib/shim.sh"
shim_init 0.10b "$@"
shim_require_lzma

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH}"
shim_install_images "aotriton-0.10b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz"
shim_finish
