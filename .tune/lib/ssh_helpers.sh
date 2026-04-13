#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# SSH command helpers

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
