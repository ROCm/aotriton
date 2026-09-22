#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Guard: verify the `aotriton.tune` package (python/tune/ -> aotriton.tune, a
# separate `aotriton-tune` distribution, see python/tune/setup.py) is
# importable before invoking `python3 -m aotriton.<tool>`. Every caller of
# this guard is a .tune/bin/* tuning CLI wrapper, so the real requirement is
# the tune subpackage, not just the `aotriton` namespace package.
# Source this (requires REPO_ROOT already set), do not execute it.

if ! python3 -c "import aotriton.tune" &>/dev/null; then
    echo "Error: the 'aotriton.tune' package is not importable by python3." >&2
    echo "  Install it with:" >&2
    echo "    pip install -e '${REPO_ROOT}'" >&2
    echo "    pip install -e '${REPO_ROOT}/python/tune'" >&2
    exit 1
fi
