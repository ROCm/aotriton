#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run .ci/run-test.sh on a single tester host via tsp (SSH-disconnect tolerant).
#
# Usage:
#   run-test.sh <workdir> <hostname> <pass#> <test_level> <split|fused|aiter|v3> [--follow]
#
#   --follow  Wait for the tsp job to complete (tsp -t <jobid>); default is fire-and-forget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

POSITIONAL=()
FOLLOW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow) FOLLOW=1; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [ "${#POSITIONAL[@]}" -lt 5 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <hostname> <pass#> <test_level> <split|fused|aiter|v3> [--follow]

  --follow  Wait for the remote tsp job to finish (tsp -t <jobid>).
            Without this flag the job is queued and the script returns immediately.
EOF
  exit 1
fi

WORKDIR="${POSITIONAL[0]}"
HOSTNAME="${POSITIONAL[1]}"
PASS_NUM="${POSITIONAL[2]}"
TEST_LEVEL="${POSITIONAL[3]}"
BACKEND="${POSITIONAL[4]}"

case "$BACKEND" in
  split|fused|aiter|v3) ;;
  *) echo "Error: backend must be one of split/fused/aiter/v3, got: $BACKEND" >&2; exit 1 ;;
esac

load_config "$WORKDIR"

WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r _arch workdir_override <<< "$WORKER_INFO"
REMOTE_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

LIBDIR="/wkdir/installed/test/lib"
REMOTE_SCRIPT="/wkdir/aotriton.src/.ci/run-test.sh"
OUTPUT_DIR="/wkdir/run/tests"

echo "[$HOSTNAME] Queuing run-test pass=$PASS_NUM level=$TEST_LEVEL backend=$BACKEND"
echo "[$HOSTNAME] output -> $REMOTE_WORKDIR/run/tests/"

# shellcheck disable=SC2029
JOBID=$(ssh "$HOSTNAME" bash -s "$REMOTE_WORKDIR" "$CELERY_WORKER_IMAGE" \
        "$LIBDIR" "$REMOTE_SCRIPT" "$OUTPUT_DIR" \
        "$PASS_NUM" "$TEST_LEVEL" "$BACKEND" <<'ENDSSH'
REMOTE_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"
LIBDIR="$3"
REMOTE_SCRIPT="$4"
OUTPUT_DIR="$5"
PASS_NUM="$6"
TEST_LEVEL="$7"
BACKEND="$8"

mkdir -p "$REMOTE_WORKDIR/run/tests"

set -x
jobid=$(tsp docker run --rm \
  --init \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -e AOTRITON_TEST_LIBDIR="$LIBDIR" \
  -e OUTPUT_DIR="$OUTPUT_DIR" \
  --mount type=bind,source="$(realpath "$REMOTE_WORKDIR")",target=/wkdir \
  "$CELERY_WORKER_IMAGE" \
  bash -l -c '
    source /wkdir/config.rc
    source "$(dirname "$CELERY_WORKER_PYTHON")/activate"
    cd /wkdir/aotriton.src
    exec bash "$0" "$@"
  ' "$REMOTE_SCRIPT" "$PASS_NUM" "$TEST_LEVEL" "$BACKEND")
echo "$jobid"
ENDSSH
)

echo "[$HOSTNAME] tsp job ID: $JOBID"

if [ "$FOLLOW" -eq 1 ]; then
  echo "[$HOSTNAME] Waiting for job $JOBID to complete..."
  ssh "$HOSTNAME" "tsp -t $JOBID"
fi
