#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 3 ]; then
  echo 'Missing arguments. Usage: run-test.sh <pass#> <test_level> <split/fused/aiter/v3>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

pass=$1
test_level="$2"
backend="$3"
bdir="build-${aotriton_major}.${aotriton_minor}-test-${native_arch}"

small_vram=$(amd-smi static -g 0 -v --json| python -c 'import json, sys; j = json.load(sys.stdin); print(int(j["gpu_data"][0]["vram"]["size"]["value"] / 1024.0 < 60))')

(
  ulimit -c 0
  cd ${SCRIPT_DIR}/..;
  export SMALL_VRAM=${small_vram};
  export COLUMNS=400;
  export FOR_RELEASE=${test_level};
  if [[ "$backend" == "split" ]]; then
    export BWD_IMPL=0
    fnprefix="ut_pass"
  fi
  if [[ "$backend" == "fused" ]]; then
    export BWD_IMPL=1
    fnprefix="fused_pass"
  fi
  if [[ "$backend" == "aiter" ]]; then
    export BWD_IMPL=2
    fnprefix="aiter_pass"
  fi
  if [[ "$backend" == "v3" ]]; then
    export V3_API=1
    fnprefix="oput_pass"
  fi
  set -v
  export PYTHONPATH="${bdir}/install_dir/lib"
  pytest --tb=line -n ${ngpus} --max-worker-restart 48 -rfEsx \
    test/test_backward.py \
    -v \
    1>"${fnprefix}${pass}.out" \
    2>"${fnprefix}${pass}.err" || true
  grep '^FAILED' "${fnprefix}${pass}.out"|sed 's/^FAILED //' | sed 's/].*/]/' > "sel${pass}.txt"
  pytest --tb=line -n ${ngpus} --max-worker-restart 48 -rfEsx \
    test/test_varlen.py \
    -v \
    1>"${fnprefix}${pass}.varlen.out" \
    2>"${fnprefix}${pass}.varlen.err" || true
  grep '^FAILED' "${fnprefix}${pass}.varlen.out"|sed 's/^FAILED //' | sed 's/].*/]/' > "sel${pass}.varlen.txt"
)
