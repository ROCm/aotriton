#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run SSH command on all registered GPU workers

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 2 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <command> [args...]

Run SSH command on all registered GPU workers.

Arguments:
  <workdir>  Project working directory
  <command>  Command and arguments to execute on each worker

Examples:
  $0 /path/to/workdir hostname
  $0 /path/to/workdir docker ps
  $0 /path/to/workdir cat /etc/os-release
EOF
  exit 1
fi

WORKDIR="$1"
shift

# Validate workdir
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Get unique hostnames
HOSTNAMES=($(sqlite3 "$WORKDIR/workers.db" "SELECT DISTINCT hostname FROM workers ORDER BY hostname;"))

if [ ${#HOSTNAMES[@]} -eq 0 ]; then
  echo "Error: No workers registered in database" >&2
  exit 1
fi

# Run command on each host
# Build properly escaped command string
CMD=$(printf '%q ' "$@")

for hostname in "${HOSTNAMES[@]}"; do
  echo "=== $hostname ==="
  ssh "$hostname" "$CMD"
  echo
done
