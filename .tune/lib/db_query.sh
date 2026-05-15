#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

_DB_QUERY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$_DB_QUERY_DIR/sqlite3_compat.sh"

# Database query helpers

get_workers() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers ORDER BY hostname"
}

get_hostnames() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT DISTINCT hostname FROM workers ORDER BY hostname"
}

get_architectures() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT DISTINCT arch FROM workers ORDER BY arch"
}

get_slurm_batch() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT id, gres FROM slurm_batch"
}

get_worker_by_hostname() {
    local workdir="$1"
    local hostname="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT arch, COALESCE(workdir_override, '') FROM workers WHERE hostname = '$hostname' LIMIT 1"
}

get_default_workdir() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT value FROM config WHERE key = 'default_workdir'"
}

get_slurm_bad_nodes() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT hostname FROM slurm_bad_nodes"
}

get_buildnode_workdir() {
    local workdir="$1"
    local override
    override=$(sqlite3 "$workdir/workers.db" \
        "SELECT COALESCE(value,'') FROM config WHERE key = 'buildnode::workdir_override'" 2>/dev/null || true)
    if [ -n "$override" ]; then
        echo "$override"
    else
        get_default_workdir "$workdir"
    fi
}

get_worker_gpu_selection() {
    local workdir="$1"
    local hostname="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT value FROM config WHERE key = '$hostname::gpu_selection'"
}

get_tester_workers() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT w.hostname, w.arch, COALESCE(w.workdir_override, '') FROM workers w INNER JOIN worker_roles r ON w.hostname = r.hostname WHERE r.role_name = 'Tester' ORDER BY w.hostname"
}
