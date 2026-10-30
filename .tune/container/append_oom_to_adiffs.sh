#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Grep OutOfMemoryError failures from a pytest .out file and print them
# in adiffs.txt format: "<test_id> (call)\tOOM"
#
# Emits OOM only. The file's other values are NAN and a JSON adiff (both written
# by the RECORD_ADIFFS_TO path in modules/flash/tests/_core_test_backward.py),
# and CPUREF, which is hand-only: it asserts that a test's GPU reference is
# untrustworthy, which no pytest log can establish.
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
