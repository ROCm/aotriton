#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Make torch usable at all on a gfx90a worker image built against ROCm 7.14.
#
# Without this a gfx90a worker cannot allocate a tensor: even
# torch.rand(device='cuda:0') raises hipErrorInvalidImage ("device kernel image
# is invalid"). Every task then dies in prepare_data before a single kernel is
# benchmarked, so the architecture is untunable rather than merely slow.
#
# The cause is which kpack archive torch resolves to. amd-torch-device-gfx90a
# ships torch_gfx90a.kpack, holding all 245 kernels, alongside
# torch_gfx90a:xnack{+,-}.kpack, which hold exactly one entry each --
# lib/libtorch_rocshmem.so#0. ROCm 7.14's loader picks the most specific
# arch-compatible archive and does not fall through when the kernel it wants is
# not in it. An MI250X reports gfx90a:sramecc+:xnack-, so the one-kernel
# archive matches more specifically and wins every lookup.
#
# Renaming the qualified archives out of the loader's glob leaves the generic
# one as the only match. The cost is rocSHMEM's single kernel, already dead
# weight here because rocSHMEM fails to initialise in this image for want of
# libnuma. ROCM_KPACK_PATH is the other available lever and is the wrong one:
# it is global, so it would also displace the rocBLAS and RCCL archives under
# _rocm_sdk_libraries/.kpack that nothing is wrong with.
#
# This is a defect in a specific ROCm release, so it is gated on both the
# architecture and the ROCm version and is expected to be deleted rather than
# maintained. When ROCM_VERSION in .tune/lib/create_dockerfile.sh moves past
# 7.14, re-test gfx90a and drop this file if the archives resolve correctly.

set -e

CONFIG_RC="${CONFIG_RC:-/config.rc}"

if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# shellcheck disable=SC1090
. "$CONFIG_RC"

if [ -z "${CELERY_WORKER_PYTHON:-}" ]; then
  echo "Error: CELERY_WORKER_PYTHON not set in $CONFIG_RC" >&2
  exit 1
fi

# Gate 1: the architecture this image was built for. Supplied by the Dockerfile
# ARG that build_image.sh fills from the worker registry, so it is the image's
# own target rather than anything probed from hardware that may not be present
# at build time.
if [ "${ROCM_GPU_ARCH:-}" != "gfx90a" ]; then
  echo "50-fix_gfx90a_torch: image targets ${ROCM_GPU_ARCH:-<unset>}, not gfx90a -- skipping"
  exit 0
fi

# Gate 2: the ROCm release. Read from torch's own version string, which encodes
# it (e.g. 2.12.0+rocm7.14.1) -- that string is what the install actually
# produced, unlike the generator-side ROCM_VERSION which never reaches the
# image and could drift from what is installed here.
TORCH_VERSION="$("$CELERY_WORKER_PYTHON" -c 'import torch; print(torch.__version__)')"
case "$TORCH_VERSION" in
  *+rocm7.14*) ;;
  *)
    echo "50-fix_gfx90a_torch: torch is $TORCH_VERSION, not a ROCm 7.14 build -- skipping"
    exit 0
    ;;
esac

echo "50-fix_gfx90a_torch: gfx90a on $TORCH_VERSION -- disabling target-id-qualified kpack archives"

VENV_DIR="$(dirname "$(dirname "$CELERY_WORKER_PYTHON")")"

for f in "$VENV_DIR"/lib/python*/site-packages/torch/.kpack/torch_gfx90a:xnack[+-].kpack; do
    [ -e "$f" ] && mv -v -- "$f" "$f.disabled" || :
done
