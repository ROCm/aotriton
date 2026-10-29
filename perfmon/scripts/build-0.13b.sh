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

case "${ARCH%%:*}" in
  gfx90a)  GROUP=gfx90a ;;
  gfx942)  GROUP=gfx942 ;;
  gfx950)  GROUP=gfx950 ;;
  gfx1250) GROUP=gfx1250 ;;
  gfx1100|gfx1101) GROUP=gfx110x ;;
  gfx1150|gfx1151) GROUP=gfx115x ;;
  gfx1200|gfx1201) GROUP=gfx120x ;;
  *) echo "[build-${TAG}] ERROR: ${TAG} publishes no kernel images for '${ARCH}'." >&2; exit 1 ;;
esac

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH}"
shim_install_images "aotriton-${TAG}-images-amd-${GROUP}.tar.gz"
shim_finish
