#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

get_worker_gpu_selection() {
    local workdir="$1"
    local hostname="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT value FROM config WHERE key = '$hostname::gpu_selection'"
}
