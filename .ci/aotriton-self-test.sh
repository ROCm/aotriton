#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# CPU-only unit tests: the ATI code generator (python/test) and the GPU-lease
# pytest plugin's own suite. No GPU, no ROCm, no built library required.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${AOTRITON_UNITTEST_VENV:-${ROOT}/build-unittest-venv}"

# cd BEFORE pip: requirements-dev.txt carries a CWD-relative path
# (./python/pytest-gpu-lease, added in Step 4).
cd "${ROOT}"

[ -d "${VENV}" ] || python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install -q -r requirements-dev.txt
"${VENV}/bin/python" -m pip install -q .   # the aotriton package; python/test imports aotriton.*

exec "${VENV}/bin/python" -m pytest python/test python/pytest-gpu-lease/tests -q "$@"
