#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

_DB_QUERY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$_DB_QUERY_DIR/sqlite3_compat.sh"

# Database query helpers

get_workers() {
    local workdir="$1"
    local role="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT w.hostname, w.arch, COALESCE(w.workdir_override, '') FROM workers w INNER JOIN worker_roles r ON w.hostname = r.hostname WHERE r.role_name = '$role' ORDER BY w.hostname"
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

# The subpool (machine group inside the arch, e.g. MI355) of a worker, or empty.
# Empty on a workers.db made before the column existed.
get_worker_subpool() {
    local workdir="$1"
    local hostname="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT COALESCE(subpool, '') FROM workers WHERE hostname = '$hostname' LIMIT 1" 2>/dev/null || true
}

get_worker_gpu_selection() {
    local workdir="$1"
    local hostname="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT value FROM config WHERE key = '$hostname::gpu_selection'"
}


# The thermal gate of a host, "on" or "off". A host property shared by every
# DEF application (config row thermal_gate::<hostname>, 1 = on), not a perfmon
# setting: it describes the machine's cooling. Reaches a node as
# AOTRITON_THERMAL_GATE. "off" when the row is absent.
get_worker_thermal_gate() {
    local workdir="$1"
    local hostname="$2"
    if [ "$(get_config_value "$workdir" "thermal_gate::$hostname")" = "1" ]; then echo on; else echo off; fi
}

# Whether a host reaches the server's services through SSH tunnels: 1 or 0.
# A host property shared by every DEF application (config row
# firewalled::<hostname>, 1 = yes); 0 when the row is absent. It describes the
# host's network, not the remedy (perfmon-pgsql-ssh-tunnel.md).
get_worker_firewalled() {
    local workdir="$1"
    local hostname="$2"
    if [ "$(get_config_value "$workdir" "firewalled::$hostname")" = "1" ]; then echo 1; else echo 0; fi
}

# A value of the workers.db config table, empty when the key is absent.
get_config_value() {
    local workdir="$1"
    local key="$2"
    sqlite3 "$workdir/workers.db" \
        "SELECT COALESCE(value, '') FROM config WHERE key = '$key'" 2>/dev/null || true
}

# A config value that holds a JSON list of strings (e.g. perfmon::tags), one
# item per line; <default> (a JSON list) when the key is absent or empty.
get_config_list() {
    local workdir="$1"
    local key="$2"
    local default="$3"
    local v
    v="$(get_config_value "$workdir" "$key")"
    [ -n "$v" ] || v="$default"
    python3 -c 'import json, sys
for item in json.loads(sys.argv[1]):
    print(item)' "$v"
}
