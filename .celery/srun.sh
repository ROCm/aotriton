#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Launch SLURM jobs for all registered SLURM batch configurations

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir>

Launch SLURM jobs for all registered SLURM batch configurations.

Arguments:
  <workdir>  Project working directory

This script will:
  - SSH to SLURM_LOGIN_NODE (from config.rc)
  - Query slurm_batch table for registered architectures
  - Submit one sbatch job per architecture with gres constraints

Prerequisites:
  - SLURM_LOGIN_NODE and SLURM_WORKER_DIR set in config.rc
  - SLURM batch configurations registered via manage-workers.py slurm-add
  - Workdir deployed to SLURM via deploy-workdir.sh
  - SLURM venv built via build-slurm-venv.sh

Examples:
  $0 /path/to/workdir
EOF
  exit 1
fi

WORKDIR="$1"
CONFIG_RC="$WORKDIR/config.rc"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$CONFIG_RC" ]; then
  echo "Error: Invalid workdir or config.rc not found" >&2
  exit 1
fi

# Source config
. "$CONFIG_RC"

if [ -z "$SLURM_LOGIN_NODE" ]; then
  echo "Error: SLURM_LOGIN_NODE not set in config.rc. SLURM not enabled." >&2
  exit 1
fi

if [ -z "$SLURM_WORKER_DIR" ]; then
  echo "Error: SLURM_WORKER_DIR not set in config.rc" >&2
  exit 1
fi

echo "Submitting SLURM jobs via $SLURM_LOGIN_NODE"

# SSH to login node and submit jobs
ssh "$SLURM_LOGIN_NODE" bash -s "$SLURM_WORKER_DIR" <<'OUTER_EOF'
SLURM_WORKER_DIR="$1"

cd "$SLURM_WORKER_DIR"

# Check if slurm_batch table has any entries
COUNT=$(sqlite3 workers.db "SELECT COUNT(*) FROM slurm_batch;")
if [ "$COUNT" -eq 0 ]; then
  echo "Error: No SLURM batch configurations registered" >&2
  echo "Use manage-workers.py slurm-add to register architectures" >&2
  exit 1
fi

echo "Found $COUNT SLURM batch configuration(s)"

# Query slurm_batch table for architectures and constraints
sqlite3 workers.db "SELECT arch, gres FROM slurm_batch;" | while IFS='|' read -r arch gres; do
  echo "Submitting SLURM job for arch=$arch gres=$gres"

  JOB_ID=$(sbatch \
    --parsable \
    --job-name="aotriton-$arch" \
    --gres="$gres" \
    "$SLURM_WORKER_DIR/aotriton.src/.celery/slurm-worker.sh" \
    "$SLURM_WORKER_DIR")

  if [ -n "$JOB_ID" ]; then
    echo "  Job submitted: $JOB_ID"
  else
    echo "  Failed to submit job" >&2
  fi
done

echo "All jobs submitted"
OUTER_EOF

if [ $? -ne 0 ]; then
  echo "Error: Failed to submit SLURM jobs" >&2
  exit 1
fi

echo "SLURM job submission completed"
