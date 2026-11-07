#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim build of the 0.11 line: <src_dir> <install_dir> <arch>. See lib/shim.sh
# for the contract. build-0.11b.sh is a symlink to this script and the tag is
# read from the name it was called by. The images are a package per arch group,
# with gfx11xx as one group.

set -euo pipefail
TAG="$(basename "${BASH_SOURCE[0]}" .sh)"
TAG="${TAG#build-}"
case "${TAG}" in
  0.11b|0.11.2b) ;;
  *) echo "[build-${TAG}] ERROR: this script is for the 0.11 line only." >&2; exit 1 ;;
esac
. "$(dirname "${BASH_SOURCE[0]}")/lib/shim.sh"
shim_init "${TAG}" "$@"
shim_require_lzma

image_pack_of() {
  case "$1" in
    gfx90a)  echo gfx90a ;;
    gfx942)  echo gfx942 ;;
    gfx950)  echo gfx950 ;;
    gfx1100|gfx1101|gfx1150|gfx1151) echo gfx11xx ;;
    gfx1200|gfx1201) echo gfx120x ;;
    *) return 1 ;;
  esac
}

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH_LIST}"
shim_install_packs "aotriton-${TAG}-images-amd-%s.tar.gz"
shim_finish
