#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# SSH command helpers

tsp_wait_and_tail() {
    local jobid="$1"
    if [ "$(tsp -s "$jobid")" = "queued" ]; then
        echo "Waiting for tsp job $jobid to start..."
        while [ "$(tsp -s "$jobid")" = "queued" ]; do sleep 5; done
    fi
    tsp -t "$jobid"
}

ssh_exec() {
    local hostname="$1"
    shift
    ssh "$hostname" "$@"
}

ssh_script() {
    local hostname="$1"
    local script="$2"
    shift 2
    ssh "$hostname" bash -s "$@" < "$script"
}
