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

case "${ARCH%%:*}" in
  gfx90a)  PACK=gfx90a ;;
  gfx942)  PACK=gfx942 ;;
  gfx950)  PACK=gfx950 ;;
  gfx1250) PACK=gfx1250 ;;
  gfx1100|gfx1101|gfx1102|gfx1103) PACK=gfx110x ;;
  gfx1150|gfx1151|gfx1152|gfx1153) PACK=gfx115x ;;
  gfx1200|gfx1201) PACK=gfx120x ;;
  *) echo "[build-${TAG}] ERROR: ${TAG} publishes no kernel images for '${ARCH}'." >&2; exit 1 ;;
esac

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH}"
shim_install_images "aotriton-${TAG}-images-amd-${PACK}.tar.gz"
if [ ! -f "${INSTALL_DIR}/include/aotriton/flash/backends.h" ]; then
  echo "[build-${TAG}] ERROR: the install has no include/aotriton/flash/backends.h." >&2
  exit 1
fi
shim_finish
