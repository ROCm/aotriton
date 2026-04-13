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

get_slurm_bad_nodes() {
    local workdir="$1"
    sqlite3 "$workdir/workers.db" \
        "SELECT hostname FROM slurm_bad_nodes"
}
