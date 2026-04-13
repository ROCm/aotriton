#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Restart worker on one host
# Usage: restart_worker.sh <hostname> <arch> <workdir> <image>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/stop_worker.sh" "$1" "$2" "$3"
"$SCRIPT_DIR/start_worker.sh" "$@"
