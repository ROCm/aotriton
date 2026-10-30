#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Read an application's manifest (python/<app>/manifest.py):
#
#   app_manifest tune deploy.role
#   app_manifest tune install.subtree --arch gfx950 --cat_level1 op
#   require_app tune          # exit 1 unless the application is registered
#
# The manifest is the only place per-application shell knowledge lives. Works
# from a bare checkout: pkgroot/aotriton is a symlink to python/, so the
# `aotriton` namespace resolves without installing anything.

_MANIFEST_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

app_manifest() {
    local app="$1"
    shift
    case "$app" in
        ""|*[!a-z0-9_]*) echo "Error: invalid application name '$app'" >&2; return 1 ;;
    esac
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$_MANIFEST_LIB_DIR/pkgroot${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m "aotriton.$app.manifest" "$@"
}

# The one validation every script keeps: "is this application registered?"
require_app() {
    local app="$1"
    if [ "$(app_manifest "$app" app 2>/dev/null)" != "$app" ]; then
        echo "Error: application '$app' is not registered (no python/$app/manifest.py with app=$app)" >&2
        return 1
    fi
}
