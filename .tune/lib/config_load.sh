#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

_CONFIG_LOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$_CONFIG_LOAD_DIR/sqlite3_compat.sh"

# Load configuration from workdir

load_config() {
    local workdir="$1"
    if [ ! -f "$workdir/config.rc" ]; then
        echo "Error: config.rc not found in $workdir" >&2
        return 1
    fi
    . "$workdir/config.rc"

    # Load default workdir from database
    if [ ! -f "$workdir/workers.db" ]; then
        echo "Error: $workdir/workers.db not found." >&2
        echo "Start the webui once with '.tune/dashboard.py $workdir' (it creates workers.db)," >&2
        echo "then set the default working directory on its Workers page." >&2
        return 1
    fi
    # '|| true': a failing query must not abort callers running under 'set -e';
    # sqlite3's stderr stays visible, and the check below reports the empty result.
    DEFAULT_WORKDIR=$(sqlite3 "$workdir/workers.db" "SELECT value FROM config WHERE key = 'default_workdir';") || true
    if [ -z "$DEFAULT_WORKDIR" ]; then
        echo "Error: Default working directory not set." >&2
        echo "Set it on the Workers page of the webui ('.tune/dashboard.py $workdir')." >&2
        return 1
    fi
}

# Simple config loader for scripts running inside containers
# Does not query database - the workdir is the container's (AOTRITON_WORKDIR)
load_config_container() {
    local workdir="$1"
    if [ ! -f "$workdir/config.rc" ]; then
        echo "Error: config.rc not found in $workdir" >&2
        return 1
    fi
    . "$workdir/config.rc"
}
