#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Remote script to get worker status
# Runs on worker node, called via SSH from get_worker_status.sh
# Usage: get_status.sh <worker_workdir>

WORKER_WORKDIR="$1"
CONTAINER_ID_FILE="$WORKER_WORKDIR/run/worker.containerid"

# Check container status
CONTAINER_STATUS="Stopped"
CONTAINER_ID_SHORT=""
if [ -f "$CONTAINER_ID_FILE" ]; then
  CONTAINER_ID=$(cat "$CONTAINER_ID_FILE" 2>/dev/null)
  if [ -n "$CONTAINER_ID" ]; then
    CONTAINER_ID_SHORT="${CONTAINER_ID:0:12}"
    CONTAINER_STATUS="pod: $CONTAINER_ID_SHORT"
  fi
fi

# Get GPU process counts (always, regardless of container state)
PROCESS_JSON=$(amd-smi process --json 2>/dev/null)
PROCESS_EXIT=$?

# Get GPU usage (always, regardless of container state)
MONITOR_JSON=$(amd-smi monitor --json 2>/dev/null)
MONITOR_EXIT=$?

# Parse with Python
python3 <<EOF
import json
import sys

# Parse process counts
ngproc = "unknown"
if $PROCESS_EXIT == 0:
    try:
        data = json.loads('''$PROCESS_JSON''')
        gpu_procs = {}

        if isinstance(data, list):
            for gpu_entry in data:
                gpu_id = gpu_entry.get('gpu', 0)
                process_list = gpu_entry.get('process_list', [])
                gpu_procs[gpu_id] = len(process_list)

            if gpu_procs:
                max_gpu = max(gpu_procs.keys())
                counts = [str(gpu_procs.get(i, 0)) for i in range(max_gpu + 1)]
                ngproc = '/'.join(counts)
            else:
                ngproc = '0'
    except:
        ngproc = "error"

# Parse GPU usage
gpu_usage = "unknown"
if $MONITOR_EXIT == 0:
    try:
        data = json.loads('''$MONITOR_JSON''')
        gpu_usages = {}

        if isinstance(data, list):
            for gpu_entry in data:
                gpu_id = gpu_entry.get('gpu', 0)
                gfx = gpu_entry.get('gfx', {})
                usage_value = gfx.get('value', 0)
                gpu_usages[gpu_id] = usage_value

            if gpu_usages:
                max_gpu = max(gpu_usages.keys())
                usages = [str(gpu_usages.get(i, 0)) for i in range(max_gpu + 1)]
                gpu_usage = '/'.join(usages)
            else:
                gpu_usage = '0'
    except:
        gpu_usage = "error"

# Output formatted status
print(f"$CONTAINER_STATUS; ngproc: {ngproc}; gpu: {gpu_usage}")
EOF
