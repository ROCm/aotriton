#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: create-project-directory.sh <dir>' >&2
  exit 1
fi
dir="$1"
pidf="$dir/run/container.pids"

if [ -f "$pidf" ]; then
  docker stop $(cat "$pidf") && rm -f "$pidf"
else
  echo "$pidf does not exist. No recorded running containers"
fi
