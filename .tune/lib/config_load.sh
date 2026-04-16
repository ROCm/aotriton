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

    # Load default workdir from database
    DEFAULT_WORKDIR=$(sqlite3 "$workdir/workers.db" "SELECT value FROM config WHERE key = 'default_workdir';" 2>/dev/null)
    if [ -z "$DEFAULT_WORKDIR" ]; then
        echo "Error: Default working directory not set. Use manage-workers.py set-default-workdir" >&2
        return 1
    fi
}

# Simple config loader for scripts running inside containers
# Does not query database - workdir is always /wkdir in container
load_config_container() {
    local workdir="$1"
    if [ ! -f "$workdir/config.rc" ]; then
        echo "Error: config.rc not found in $workdir" >&2
        return 1
    fi
    . "$workdir/config.rc"
}
