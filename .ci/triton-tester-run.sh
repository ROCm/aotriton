#!/bin/bash

#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: triton-tester-run.sh <pass#>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

pass=$1
bdir="aotriton-build-triton_tester-${native_arch}"

(
  ulimit -c 0
  cd ${SCRIPT_DIR}/..;
  export COLUMNS=400;
  export PYTHONPATH="${bdir}/lib"
  fnprefix='triton_tester_pass'
  pytest --tb=line -n ${ngpus} --max-worker-restart 48 -rfEsx \
    test/triton_tester.py \
    -v \
    1>"${fnprefix}${pass}.out" \
    2>"${fnprefix}${pass}.err" || true
)
