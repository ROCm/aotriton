#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Guard: verify both the main `aotriton` distribution and `aotriton.tune`
# (its own `aotriton-tune` distribution, see python/tune/setup.py) are
# importable before `python3 -m aotriton.<tool>`. Checking `aotriton.tune`
# alone is not enough: `aotriton` is a namespace package, so installing only
# `aotriton-tune` already satisfies `import aotriton.tune` with the main
# distribution absent -- callers like `.tune/bin/sancheck`
# (`python3 -m aotriton.generate`) or `.tune/bin/decomposedb`
# (`python3 -m aotriton.database_decompose`) would pass this guard and only
# fail later.
# Source this (requires REPO_ROOT already set), do not execute it.

if ! python3 -c "import aotriton.generate, aotriton.tune" &>/dev/null; then
    echo "Error: 'aotriton' (main) and/or 'aotriton.tune' not importable by python3." >&2
    echo "  Install both with:" >&2
    echo "    pip install -e '${REPO_ROOT}'" >&2
    echo "    pip install -e '${REPO_ROOT}/python/tune'" >&2
    exit 1
fi
