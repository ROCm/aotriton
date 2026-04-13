#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Load configuration from workdir

load_config() {
    local workdir="$1"
    if [ ! -f "$workdir/config.rc" ]; then
        echo "Error: config.rc not found in $workdir" >&2
        return 1
    fi
    . "$workdir/config.rc"
}
