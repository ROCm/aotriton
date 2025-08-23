#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: build-tune.sh <target arch>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-build.sh"

common_build "$1" "tune" "-DAOTRITON_BUILD_FOR_TUNING=ON"
