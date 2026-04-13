#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Submit one SLURM job
# Usage: submit_slurm_job.sh <login_node> <workdir> <gres> <time_limit> [exclude_nodes]

set -e

SLURM_LOGIN_NODE="$1"
SLURM_WORKER_DIR="$2"
GRES="$3"
TIME_LIMIT="$4"
EXCLUDE_NODES="${5:-}"

EXCLUDE_OPT=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_OPT="--exclude=$EXCLUDE_NODES"
fi

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
