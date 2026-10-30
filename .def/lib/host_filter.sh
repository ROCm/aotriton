#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Host filtering logic

declare -a FILTER_HOSTS=()

parse_host_filter() {
    # Parse --host options from arguments.
    #
    # ONE hostname per --host, repeatable: `--host a --host b`. It used to slurp
    # every following argument that did not start with `--`, which silently ate
    # the positional that came after the host list:
    #
    #     imgbld --host worker-01 ~/wkdir     # the documented argument order
    #
    # put BOTH `worker-01` and `~/wkdir` into FILTER_HOSTS, leaving no workdir,
    # so the command died on its own usage message. The slurp could not tell a
    # second hostname from a trailing positional, and nothing in the tree ever
    # relied on it: .tune/bin/detect-gpus (:17-26, :55-56) and .tune/bin/runtest
    # (:28) both already take one value per flag and document repetition, so
    # this makes the shared helper agree with them rather than inventing a rule.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)
                shift
                if [[ $# -eq 0 ]]; then
                    echo "Error: --host requires a hostname argument" >&2
                    return 1
                fi
                FILTER_HOSTS+=("$1")
                shift
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
