#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim build of AOTriton 0.9.2b: <src_dir> <install_dir> <arch>. See
# lib/shim.sh for the contract.
#
# This release predates AOTRITON_TARGET_ARCH: it takes TARGET_GPUS, a name from
# v2python/gpu_targets.py. Its images are not a separate package: they are
# inside the one jumbo tarball, which also carries the runtime.

set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib/shim.sh"
shim_init 0.9.2b "$@"

# The TARGET_GPUS name of an arch, read from the release's own table.
target_gpu_of() {
  python3 - "${SRC_DIR}/v2python/gpu_targets.py" "${1%%:*}" <<'PYEOF'
import ast, pathlib, re, sys
table, gfx = sys.argv[1], sys.argv[2]
m = re.search(r"AOTRITON_GPU_ARCH_TUNING_STRING\s*=\s*(\{.*?\})", pathlib.Path(table).read_text(), re.S)
if not m:
    sys.exit(f"no AOTRITON_GPU_ARCH_TUNING_STRING in {table}")
by_gfx = {v: k for k, v in ast.literal_eval(m.group(1)).items()}
if gfx not in by_gfx:
    sys.exit(f"{gfx} is not supported by this release; it knows: " + " ".join(sorted(by_gfx)))
print(by_gfx[gfx])
PYEOF
}

TARGET_GPU="$(target_gpu_of "${ARCH}")"
echo "[build-${TAG}] arch ${ARCH} -> TARGET_GPUS=${TARGET_GPU}" >&2

shim_build ON -DTARGET_GPUS="${TARGET_GPU}"
shim_install_images "aotriton-0.9.2b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz"
shim_finish
