#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Remote script to get worker status
# Runs on worker node, called via SSH from get_worker_status.sh
# Usage: get_status.sh <worker_workdir>

WORKER_WORKDIR="$1"
CONTAINER_ID_FILE="$WORKER_WORKDIR/run/worker.containerid"

# Read container ID
if [ ! -f "$CONTAINER_ID_FILE" ]; then
  echo "Stopped"
  exit 0
fi

CONTAINER_ID=$(cat "$CONTAINER_ID_FILE" 2>/dev/null)
if [ -z "$CONTAINER_ID" ]; then
  echo "Stopped"
  exit 0
fi

# Truncate to 12 digits
CONTAINER_ID_SHORT="${CONTAINER_ID:0:12}"

# Get GPU process counts using amd-smi
PROCESS_JSON=$(docker exec "$CONTAINER_ID" amd-smi process --json 2>&1)
if [ $? -ne 0 ]; then
  echo "pod: $CONTAINER_ID_SHORT; ngproc: unknown"
  exit 0
fi

# Parse JSON with Python
NGPROC=$(python3 <<EOF
import json
import sys

try:
    data = json.loads('''$PROCESS_JSON''')

    # Format: list of {"gpu": N, "process_list": [...]}
    gpu_procs = {}

    if isinstance(data, list):
        for gpu_entry in data:
            gpu_id = gpu_entry.get('gpu', 0)
            process_list = gpu_entry.get('process_list', [])
            gpu_procs[gpu_id] = len(process_list)
    else:
        # Unexpected format
        print('0')
        sys.exit(0)

    # Get max GPU ID to know range
    if gpu_procs:
        max_gpu = max(gpu_procs.keys())
        counts = [str(gpu_procs.get(i, 0)) for i in range(max_gpu + 1)]
        print('/'.join(counts))
    else:
        print('0')

except json.JSONDecodeError as e:
    print('error', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print('error', file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
  echo "pod: $CONTAINER_ID_SHORT; ngproc: error"
  exit 0
fi

echo "pod: $CONTAINER_ID_SHORT; ngproc: $NGPROC"
