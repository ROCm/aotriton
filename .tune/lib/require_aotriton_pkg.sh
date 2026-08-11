#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Guard: verify the `aotriton` package (python/ -> aotriton, see repo-root
# setup.py) is importable before invoking `python3 -m aotriton.<tool>`.
# Source this (requires REPO_ROOT already set), do not execute it.

if ! python3 -c "import aotriton" &>/dev/null; then
    echo "Error: the 'aotriton' package is not importable by python3." >&2
    echo "  Install it with: pip install -e '${REPO_ROOT}'" >&2
    exit 1
fi
