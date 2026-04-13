#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Host filtering logic

declare -a FILTER_HOSTS=()

parse_host_filter() {
    # Parse --host options from arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)
                shift
                while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                    FILTER_HOSTS+=("$1")
                    shift
                done
                ;;
            *)
                shift
                ;;
        esac
    done
}

should_process_host() {
    local hostname="$1"
    if [ ${#FILTER_HOSTS[@]} -eq 0 ]; then
        return 0  # No filter, process all
    fi
    for filter in "${FILTER_HOSTS[@]}"; do
        if [ "$hostname" = "$filter" ]; then
            return 0  # Match, process
        fi
    done
    return 1  # No match, skip
}
