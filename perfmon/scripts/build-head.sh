#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Full build of the current checkout: <src_dir> <install_dir> <arch>. See
# lib/shim.sh for the contract. HEAD has no published kernel images, so unlike
# the release scripts this compiles the kernels itself (the normal build, with
# AOTRITON_NAME_SUFFIX=_pmon) and installs lib/aotriton.images from it. That
# needs Triton: pass what the build needs through PERFMON_HEAD_CMAKE_ARGS,
# e.g. "-DAOTRITON_USE_LOCAL_TRITON_WHEEL=/path/to.whl
# -DAOTRITON_USE_LOCAL_FLYDSL_WHEEL=/path/to.whl". The install holds
# include/aotriton/flash/backends.h, which the adapter build turns into its
# backend table.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib/shim.sh"
shim_init head "$@"
shim_require_lzma

# Word splitting is intended: the variable holds a list of -D options.
# shellcheck disable=SC2086
shim_build OFF -DAOTRITON_TARGET_ARCH="${ARCH}" ${PERFMON_HEAD_CMAKE_ARGS:-}
if [ ! -f "${INSTALL_DIR}/include/aotriton/flash/backends.h" ]; then
  echo "[build-${TAG}] ERROR: the install has no include/aotriton/flash/backends.h." >&2
  exit 1
fi
shim_finish
