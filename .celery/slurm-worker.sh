#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Example SLURM script for running AOTriton tuning workers
#
# Usage:
#   sbatch .celery/slurm-worker.sh <workdir>
#
# This script demonstrates how to run Celery workers in a SLURM environment
# with graceful shutdown handling.

#SBATCH --job-name=aotriton-worker
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Send SIGTERM 30 minutes (1800 seconds) before SLURM timeout
#SBATCH --signal=TERM@1800

if [ "$#" -ne 1 ]; then
  echo "Error: Missing workdir argument" >&2
  echo "Usage: sbatch $0 <workdir>" >&2
  exit 1
fi

WORKDIR="$1"
CONFIG_RC="$WORKDIR/config.rc"

# Validate workdir
if [ ! -d "$WORKDIR" ] || [ ! -f "$CONFIG_RC" ]; then
  echo "Error: Invalid workdir or config.rc not found" >&2
  exit 1
fi

# Source config
. "$CONFIG_RC"

# Find AOTriton source directory
# Assumes this script is in .celery/ subdirectory of AOTriton
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOTRITON_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set flag for worker-service.sh to use SLURM-specific paths
export CELERY_FROM_SLURM=1

# Setup Python environment
# Use SLURM venv if SLURM_WORKER_DIR is set, otherwise fall back to CELERY_WORKER_PYTHON
if [ -n "$SLURM_WORKER_DIR" ]; then
  VENV_ACTIVATE="$SLURM_WORKER_DIR/installed/venv/bin/activate"
else
  VENV_ACTIVATE="$(dirname "$CELERY_WORKER_PYTHON")/activate"
fi

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Error: Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

# Build PYTHONPATH with all installed architecture libraries and AOTriton source
PYTHONPATH="$AOTRITON_SRC"
for arch_dir in "$WORKDIR"/installed/*/lib; do
  if [ -d "$arch_dir" ]; then
    PYTHONPATH="$arch_dir:$PYTHONPATH"
  fi
done
export PYTHONPATH

# Set AOTRITON_CELERY_WORKDIR for tasks
export AOTRITON_CELERY_WORKDIR="$WORKDIR"

# Graceful shutdown handler
cleanup() {
  echo "$(date): SLURM timeout approaching, stopping workers gracefully..."
  bash "$AOTRITON_SRC/.celery/worker-service.sh" stopwait "$WORKDIR"
  echo "$(date): Workers stopped gracefully"

  # Clean up tmpfs
  rm -rf /dev/shm/aotriton-tuner

  exit 0
}

# Register signal handler
trap cleanup SIGTERM

echo "$(date): Starting AOTriton Celery workers on $(hostname)"
echo "Workdir: $WORKDIR"
echo "AOTriton source: $AOTRITON_SRC"
echo "Python: $CELERY_WORKER_PYTHON"

# Start workers
bash "$AOTRITON_SRC/.celery/worker-service.sh" start "$WORKDIR"

if [ $? -ne 0 ]; then
  echo "Error: Failed to start workers" >&2
  exit 1
fi

echo "$(date): Workers started successfully"
echo "Job will run until SLURM timeout or manual cancellation"
echo "Graceful shutdown will be triggered 5 minutes before timeout"

# Wait indefinitely (will be interrupted by SIGTERM)
sleep infinity &
wait $!
