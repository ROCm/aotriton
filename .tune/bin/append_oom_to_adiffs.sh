#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Grep OutOfMemoryError failures from a pytest .out file and print them
# in adiffs.txt format: "<test_id> (call)\tOOM"
#
# Usage:
#   append_oom_to_adiffs.sh <out_file>

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <out_file>" >&2
    exit 1
fi

OUT_FILE="$1"

if [ ! -f "$OUT_FILE" ]; then
    echo "Error: out file not found: $OUT_FILE" >&2
    exit 1
fi

grep '^FAILED' "$OUT_FILE" | grep 'OutOfMemoryError' | \
    sed 's/^FAILED //' | sed 's/ - .*//' | \
    while IFS= read -r test_id; do
        printf '%s (call)\tOOM\n' "$test_id"
    done
