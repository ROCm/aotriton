#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Launch SLURM jobs for all registered SLURM batch configurations

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -lt 1 ]; then
  cat >&2 <<EOF
Usage: $0 [--time <time>] [--gres <gres>...] <workdir>

Launch SLURM jobs for all registered SLURM batch configurations.

Arguments:
  <workdir>       Project working directory

Options:
  --time <time>   Set job time limit (e.g., "04:00:00", "2-12:00:00")
                  Default: 12:00:00
  --gres <gres>   Submit job with specific GRES constraint. Can be specified multiple times.
                  If provided, ignores database configurations and uses only these values.

This script will:
  - SSH to SLURM_LOGIN_NODE (from config.rc)
  - Query slurm_batch table for registered architectures (unless --gres specified)
  - Submit one sbatch job per gres configuration

Prerequisites:
  - SLURM_LOGIN_NODE and SLURM_WORKER_DIR set in config.rc
  - Workdir deployed to SLURM via deploy-workdir.sh
  - SLURM venv built via build-slurm-venv.sh
  - SLURM batch configurations registered via manage-workers.py slurm-add (unless using --gres)

Examples:
  # Use database configurations
  $0 /path/to/workdir
  $0 --time 08:00:00 /path/to/workdir

  # Override with specific GRES (bypasses database)
  $0 --gres "gpu:mi300x:8" /path/to/workdir
  $0 --gres "gpu:mi300x:8" --gres "gpu:gfx1100w:4" --time 12:00:00 /path/to/workdir
EOF
  exit 1
fi

# Parse arguments
TIME_LIMIT="12:00:00"
GRES_OVERRIDE=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --gres)
      GRES_OVERRIDE+=("$2")
      shift 2
      ;;
    -*)
      echo "Error: Unknown option: $1" >&2
      exit 1
      ;;
    *)
      WORKDIR="$1"
      shift
      ;;
  esac
done

if [ -z "$WORKDIR" ]; then
  echo "Error: Missing workdir argument" >&2
  exit 1
fi
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

# Determine GRES configurations
if [ ${#GRES_OVERRIDE[@]} -gt 0 ]; then
  GRES_ARRAY=("${GRES_OVERRIDE[@]}")
  echo "Using command-line GRES: ${GRES_ARRAY[*]}"
else
  # Query local database for GRES configurations
  mapfile -t GRES_ARRAY < <(sqlite3 "$WORKDIR/workers.db" "SELECT gres FROM slurm_batch;")
  if [ ${#GRES_ARRAY[@]} -eq 0 ]; then
    echo "Error: No SLURM batch configurations registered" >&2
    echo "Use manage-workers.py slurm-add to register gres configurations" >&2
    exit 1
  fi
  echo "Found ${#GRES_ARRAY[@]} SLURM batch configuration(s) from database"
fi

# Get list of bad nodes to exclude from local database
BAD_NODES=$(sqlite3 "$WORKDIR/workers.db" "SELECT GROUP_CONCAT(hostname, ',') FROM slurm_bad_nodes;" 2>/dev/null)
if [ -n "$BAD_NODES" ]; then
  echo "Excluding bad nodes: $BAD_NODES"
  EXCLUDE_OPT="--exclude=$BAD_NODES"
else
  EXCLUDE_OPT=""
fi

echo "Submitting SLURM jobs via $SLURM_LOGIN_NODE (time limit: $TIME_LIMIT)"

# Convert GRES array to space-separated string for passing to SSH
GRES_LIST="${GRES_ARRAY[*]}"

# SSH to login node and submit jobs
ssh "$SLURM_LOGIN_NODE" bash -l -s "$SLURM_WORKER_DIR" "$TIME_LIMIT" "$GRES_LIST" "$EXCLUDE_OPT" <<'OUTER_EOF'
SLURM_WORKER_DIR="$1"
TIME_LIMIT="$2"
GRES_LIST="$3"
EXCLUDE_OPT="$4"

cd "$SLURM_WORKER_DIR"

# Load config to get SLURM modules
source "$SLURM_WORKER_DIR/config.rc"

# Load SLURM modules
for module in "${SLURM_MODULES[@]}"; do
  module load "$module"
done

# Create directory for job tracking
mkdir -p "$SLURM_WORKER_DIR/run/slurm"
JOBID_FILE="$SLURM_WORKER_DIR/run/slurm/jobs-$(date +%Y%m%d-%H%M%S).txt"

# Convert space-separated GRES list back to array
read -ra GRES_ARRAY <<< "$GRES_LIST"

# Submit jobs for each gres configuration
for gres in "${GRES_ARRAY[@]}"; do
  echo "Submitting SLURM job for gres=$gres"

  JOB_ID=$(sbatch \
    --parsable \
    --job-name="aotriton-${gres//:/--}" \
    --time="$TIME_LIMIT" \
    --gres="$gres" \
    ${EXCLUDE_OPT:+$EXCLUDE_OPT} \
    "$SLURM_WORKER_DIR/aotriton.src/.celery/slurm-worker.sh" \
    "$SLURM_WORKER_DIR")

  if [ -n "$JOB_ID" ]; then
    echo "  Job submitted: $JOB_ID"
    echo "$JOB_ID|$gres" >> "$JOBID_FILE"
  else
    echo "  Failed to submit job" >&2
  fi
done

# Report job tracking file location
if [ -f "$JOBID_FILE" ]; then
  JOB_COUNT=$(wc -l < "$JOBID_FILE")
  echo "All jobs submitted: $JOB_COUNT job(s)"
  echo "Job IDs recorded in: $JOBID_FILE"
else
  echo "Warning: No jobs were successfully submitted" >&2
fi
OUTER_EOF

if [ $? -ne 0 ]; then
  echo "Error: Failed to submit SLURM jobs" >&2
  exit 1
fi

echo "SLURM job submission completed"
