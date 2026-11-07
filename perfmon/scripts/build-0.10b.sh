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

# The jumbo tarball has the images of every arch of the release; the pack names
# only serve shim_record_missing_images, which looks for amd-<name> directories.
image_pack_of() {
  case "$1" in
    gfx90a|gfx942|gfx950|gfx1250) echo "$1" ;;
    gfx1100|gfx1101|gfx1102|gfx1103) echo "gfx110x gfx11xx $1" ;;
    gfx1150|gfx1151|gfx1152|gfx1153) echo "gfx115x gfx11xx $1" ;;
    gfx1200|gfx1201) echo "gfx120x $1" ;;
    *) return 1 ;;
  esac
}

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH_LIST}"
shim_install_images "aotriton-0.10b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz"
shim_finish
