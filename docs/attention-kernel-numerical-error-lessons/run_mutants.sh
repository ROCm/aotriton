#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Every mutant from mutants.py must FAIL test_common_mistakes, and the pristine
# Triton kernels must pass it. Needs a GPU and the Triton JIT environment.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KERNEL_DIR="$HERE/../../modules/flash/kernel"
export MUTANTS_DIR="${MUTANTS_DIR:-${TMPDIR:-/tmp}/aotriton-mutants}"
export TRITON_F32_DEFAULT=ieee

run() {
    (cd "$1" && timeout 1800 python -m pytest -q test_backward.py -k test_common_mistakes -p no:cacheprovider 2>&1)
}

out=$(run "$KERNEL_DIR")
echo "=== pristine: $(echo "$out" | grep -E 'passed|failed' | tail -1)"

python "$HERE/mutants.py" make all > /dev/null || exit 1
for m in $(python "$HERE/mutants.py" list); do
    out=$(run "$MUTANTS_DIR/$m")
    echo "=== $m: $(echo "$out" | grep -E 'passed|failed' | tail -1)"
    echo "$out" | grep -E '^E +AssertionError' | sed 's/^E  */    /' | cut -c1-230 | sort -u
done
