#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim build of 0.12.1b and 0.13b: <src_dir> <install_dir> <arch>. See
# lib/shim.sh for the contract. build-0.12.1b.sh is a symlink to this script
# and the tag is read from the name it was called by. Both split gfx11xx into
# gfx110x and gfx115x and have gfx1250. (+asan assets also exist and are never
# selected.)

set -euo pipefail
TAG="$(basename "${BASH_SOURCE[0]}" .sh)"
TAG="${TAG#build-}"
case "${TAG}" in
  0.12.1b|0.13b) ;;
  *) echo "[build-${TAG}] ERROR: this script is for 0.12.1b and 0.13b only." >&2; exit 1 ;;
esac
. "$(dirname "${BASH_SOURCE[0]}")/lib/shim.sh"
shim_init "${TAG}" "$@"
shim_require_lzma

image_pack_of() {
  case "$1" in
    gfx90a)  echo gfx90a ;;
    gfx942)  echo gfx942 ;;
    gfx950)  echo gfx950 ;;
    gfx1250) echo gfx1250 ;;
    gfx1100|gfx1101) echo gfx110x ;;
    gfx1150|gfx1151) echo gfx115x ;;
    gfx1200|gfx1201) echo gfx120x ;;
    *) return 1 ;;
  esac
}

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH_LIST}"
shim_install_packs "aotriton-${TAG}-images-amd-%s.tar.gz"
shim_finish
