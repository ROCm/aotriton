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

case "${ARCH%%:*}" in
  gfx90a)  GROUP=gfx90a ;;
  gfx942)  GROUP=gfx942 ;;
  gfx950)  GROUP=gfx950 ;;
  gfx1100|gfx1101|gfx1150|gfx1151) GROUP=gfx11xx ;;
  gfx1200|gfx1201) GROUP=gfx120x ;;
  *) echo "[build-${TAG}] ERROR: ${TAG} publishes no kernel images for '${ARCH}'." >&2; exit 1 ;;
esac

shim_build ON -DAOTRITON_TARGET_ARCH="${ARCH}"
shim_install_images "aotriton-${TAG}-images-amd-${GROUP}.tar.gz"
shim_finish
