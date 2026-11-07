#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim build of AOTriton 0.14.2b: <src_dir> <install_dir> <arch>. See
# lib/shim.sh for the contract.
#
# The images are a package per arch pack (python/gpu_targets.py's
# AOTRITON_ARCH_TO_PACK at this tag), named
# aotriton-0.14.2b-images-amd-<pack>.tar.gz. The shim installs
# include/aotriton/flash/backends.h, which the adapter build turns into its
# backend table.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib/shim.sh"
shim_init 0.14.2b "$@"
shim_require_lzma

image_pack_of() {
  case "$1" in
    gfx90a)  echo gfx90a ;;
    gfx942)  echo gfx942 ;;
    gfx950)  echo gfx950 ;;
    gfx1250) echo gfx1250 ;;
    gfx1100|gfx1101|gfx1102|gfx1103) echo gfx110x ;;
    gfx1150|gfx1151|gfx1152|gfx1153) echo gfx115x ;;
    gfx1200|gfx1201) echo gfx120x ;;
    *) return 1 ;;
  esac
}

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH_LIST}"
shim_install_packs "aotriton-${TAG}-images-amd-%s.tar.gz"
if [ ! -f "${INSTALL_DIR}/include/aotriton/flash/backends.h" ]; then
  echo "[build-${TAG}] ERROR: the install has no include/aotriton/flash/backends.h." >&2
  exit 1
fi
shim_finish
