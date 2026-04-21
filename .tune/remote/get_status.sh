#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Remote script to get worker status
# Runs on worker node, called via SSH from get_worker_status.sh
# Usage: get_status.sh <worker_workdir>

WORKER_WORKDIR="$1"
CONTAINER_ID_FILE="$WORKER_WORKDIR/run/worker.containerid"

# Get git status for aotriton.src
AOTRITON_SRC="$WORKER_WORKDIR/aotriton.src"
GIT_HASH=$(cd "$AOTRITON_SRC" && git rev-parse --short=12 HEAD 2>/dev/null || echo "unknown")

# Check if working tree is dirty (git diff returns 1 if dirty)
# Use subshell to avoid changing pwd
IS_DIRTY=$(cd "$AOTRITON_SRC" 2>/dev/null && { git diff --quiet HEAD 2>/dev/null && git diff --cached --quiet 2>/dev/null; echo $?; } || echo 0)

# Add -dirty suffix if needed
if [ "$IS_DIRTY" != "0" ]; then
  GIT_HASH="${GIT_HASH}-dirty"
fi

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

# Get GPU enumeration to build HIP ID mapping (always, regardless of container state)
ENUM_JSON=$(amd-smi list -e --json 2>/dev/null)
ENUM_EXIT=$?

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

# Build HIP ID to AMD-SMI GPU index mapping
hip_to_amdsmi = {}
if $ENUM_EXIT == 0:
    try:
        data = json.loads('''$ENUM_JSON''')
        if isinstance(data, list):
            for entry in data:
                hip_id = entry.get('hip_id')
                gpu_id = entry.get('gpu')
                if hip_id is not None and gpu_id is not None:
                    hip_to_amdsmi[hip_id] = gpu_id
    except:
        pass

# Parse process counts (indexed by AMD-SMI GPU)
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
                # Reorder by HIP ID
                if hip_to_amdsmi:
                    max_hip = max(hip_to_amdsmi.keys())
                    counts = []
                    for hip_id in range(max_hip + 1):
                        amdsmi_id = hip_to_amdsmi.get(hip_id)
                        if amdsmi_id is not None:
                            counts.append(str(gpu_procs.get(amdsmi_id, 0)))
                        else:
                            counts.append('?')
                    ngproc = '/'.join(counts)
                else:
                    # Fallback to AMD-SMI order if mapping unavailable
                    max_gpu = max(gpu_procs.keys())
                    counts = [str(gpu_procs.get(i, 0)) for i in range(max_gpu + 1)]
                    ngproc = '/'.join(counts)
            else:
                ngproc = '0'
    except:
        ngproc = "error"

# Parse GPU usage (indexed by AMD-SMI GPU)
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
                # Reorder by HIP ID
                if hip_to_amdsmi:
                    max_hip = max(hip_to_amdsmi.keys())
                    usages = []
                    for hip_id in range(max_hip + 1):
                        amdsmi_id = hip_to_amdsmi.get(hip_id)
                        if amdsmi_id is not None:
                            usages.append(str(gpu_usages.get(amdsmi_id, 0)))
                        else:
                            usages.append('?')
                    gpu_usage = '/'.join(usages)
                else:
                    # Fallback to AMD-SMI order if mapping unavailable
                    max_gpu = max(gpu_usages.keys())
                    usages = [str(gpu_usages.get(i, 0)) for i in range(max_gpu + 1)]
                    gpu_usage = '/'.join(usages)
            else:
                gpu_usage = '0'
    except:
        gpu_usage = "error"

# Output formatted status
print(f"aotriton.src: $GIT_HASH; $CONTAINER_STATUS; ngproc: {ngproc}; gpu: {gpu_usage} (GPUs ordered by HIP-ID NOT AMD-SMI)")
EOF
