#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"

cd "${SCRIPT_DIR}/.."
export USE_ADIFFS_TXT="$(realpath test/adiffs/${native_arch}.txt)"
bash "${SCRIPT_DIR}/run-test.sh" "$@"
