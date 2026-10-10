#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Submit one SLURM job for a specific GRES configuration
# Usage: submit_slurm_job.sh <workdir> <gres>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
GRES="$2"

if [ -z "$WORKDIR" ] || [ -z "$GRES" ]; then
  echo "Usage: $0 <workdir> <gres>" >&2
  echo "" >&2
  echo "  Submit one SLURM job for a specific GRES (GPU resource) configuration." >&2
  echo "  Requires SLURM_LOGIN_NODE to be set in config.rc." >&2
  echo "  Bad nodes from workers.db are automatically excluded." >&2
  echo "  Time limit defaults to 24:00:00; override with SLURM_TIME_LIMIT env var." >&2
  echo "  Prints the SLURM job ID to stdout on success." >&2
  exit 1
fi

load_config "$WORKDIR"

if [ -z "$SLURM_LOGIN_NODE" ]; then
  echo "Error: SLURM not configured" >&2
  exit 1
fi

# Get bad nodes for exclusion
BAD_NODES_STR=$(get_slurm_bad_nodes "$WORKDIR" | tr '\n' ',' | sed 's/,$//')

EXCLUDE_OPT=""
if [ -n "$BAD_NODES_STR" ]; then
  EXCLUDE_OPT="--exclude=$BAD_NODES_STR"
fi

# Get time limit from environment or use default
TIME_LIMIT="${SLURM_TIME_LIMIT:-24:00:00}"

ssh "$SLURM_LOGIN_NODE" bash -s "$SLURM_WORKER_DIR" "$GRES" "$TIME_LIMIT" "$EXCLUDE_OPT" <<'EOF'
SLURM_WORKER_DIR="$1"
GRES="$2"
TIME_LIMIT="$3"
EXCLUDE_OPT="$4"

cd "$SLURM_WORKER_DIR"

JOB_ID=$(sbatch \
    --job-name="aotriton-${GRES//[^a-zA-Z0-9]/_}" \
    --gres="$GRES" \
    --time="$TIME_LIMIT" \
    $EXCLUDE_OPT \
    --parsable \
    "$SLURM_WORKER_DIR/aotriton.src/.tune/remote/slurm_worker_job.sh" \
    "$SLURM_WORKER_DIR")

echo "$JOB_ID"
EOF
