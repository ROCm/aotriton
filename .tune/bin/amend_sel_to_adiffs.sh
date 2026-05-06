#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Convert a sel<pass>.txt file into an adiffs.txt file.
# Each test id becomes: "<test_id> (call)\t<error_reason>"
#
# Usage:
#   amend_sel_to_adiffs.sh <sel_file> [--error_reason <reason>]
#
# Output is written to stdout.

set -euo pipefail

SEL_FILE=""
ERROR_REASON="OOM"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --error_reason) ERROR_REASON="$2"; shift 2 ;;
    -*) echo "Unknown argument: $1" >&2; exit 1 ;;
    *)
      if [ -z "$SEL_FILE" ]; then
        SEL_FILE="$1"; shift
      else
        echo "Unexpected argument: $1" >&2; exit 1
      fi
      ;;
  esac
done

if [ -z "$SEL_FILE" ]; then
  echo "Usage: $0 <sel_file> [--error_reason <reason>]" >&2
  exit 1
fi

if [ ! -f "$SEL_FILE" ]; then
  echo "Error: sel file not found: $SEL_FILE" >&2
  exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  printf '%s (call)\t%s\n' "$line" "$ERROR_REASON"
done < "$SEL_FILE"
