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
  echo "51-fix_gfx90a_torch: image targets ${ROCM_GPU_ARCH:-<unset>}, not gfx90a -- skipping"
  exit 0
fi

# Gate 2: the ROCm release, from `rocm-sdk version`.
#
# Asked of the SDK itself because the SDK is what the defect is in. torch's
# version string carries a rocm suffix and would answer the question too, but
# it answers a different one -- which ROCm that wheel was built against -- and
# it would keep this gate hostage to torch being importable and to the wheel's
# naming convention, neither of which has anything to do with the loader bug.
VENV_BIN="$(dirname "$CELERY_WORKER_PYTHON")"
if [ ! -x "$VENV_BIN/rocm-sdk" ]; then
  echo "51-fix_gfx90a_torch: no rocm-sdk in this image -- skipping"
  exit 0
fi

# Take the first dotted-numeric token out of the output rather than matching the
# whole string: `rocm-sdk version` is free to print a bare version or to wrap it
# in a sentence, and only the number is being tested.
ROCM_VERSION="$("$VENV_BIN/rocm-sdk" version 2>/dev/null \
                | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)*' | head -n 1)"

case "$ROCM_VERSION" in
  7.14|7.14.*) ;;
  '')
    echo "51-fix_gfx90a_torch: could not read a version from \`rocm-sdk version\` -- skipping"
    exit 0
    ;;
  *)
    echo "51-fix_gfx90a_torch: ROCm is $ROCM_VERSION, not 7.14 -- skipping"
    exit 0
    ;;
esac

echo "51-fix_gfx90a_torch: gfx90a on ROCm $ROCM_VERSION -- disabling target-id-qualified kpack archives"

VENV_DIR="$(dirname "$VENV_BIN")"

for f in "$VENV_DIR"/lib/python*/site-packages/torch/.kpack/torch_gfx90a:xnack[+-].kpack; do
    [ -e "$f" ] && mv -v -- "$f" "$f.disabled" || :
done
