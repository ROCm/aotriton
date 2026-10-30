#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Control-side helpers shared by the .perfmon/bin commands that drive a fleet
# peer over ssh (automation doc sections 1.1 and 4). Source it after
# .def/lib/db_query.sh.

PERFMON_DEFAULT_ROCM="10.1.0"
PERFMON_DEFAULT_TAGS='["0.9.2b","0.10b","0.11.2b","0.12.1b","0.13b","0.14.2b"]'

# The shared scratch: --scratch if given, else the parent of the workdir
# (claudepak-scratch/wkdir.<name> lives directly in it, automation doc 4).
perfmon_scratch_root() {
    local workdir="$1"
    local override="${2:-}"
    if [ -n "$override" ]; then
        echo "$override"
    else
        dirname "$(cd "$workdir" && pwd)"
    fi
}

# The ROCm versions to use: the explicit ones if the caller has any (the
# arguments), else workers.db's perfmon::rocm, else the default. One per line.
perfmon_rocm_versions() {
    local workdir="$1"
    shift
    if [ "$#" -gt 0 ]; then
        printf '%s\n' "$@"
    else
        get_config_list "$workdir" "perfmon::rocm" "[\"$PERFMON_DEFAULT_ROCM\"]"
    fi
}
