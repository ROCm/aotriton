#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run .ci/run-test.sh on a single tester host via tsp (SSH-disconnect tolerant).
#
# Usage:
#   run-test.sh <workdir> <hostname> <arch> <workdir_override> <pass#> <test_level> <split|fused|aiter|v3> [partial] [--follow]
#
#   workdir_override  Remote workdir override (empty string = use DEFAULT_WORKDIR from config.rc)
#   partial           Optional 8th positional arg; sets PARTIAL_INFO_DIR and routes output to partial/
#   --follow          Wait for the tsp job to complete (tsp -t <jobid>); default is fire-and-forget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"

POSITIONAL=()
FOLLOW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow) FOLLOW=1; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [ "${#POSITIONAL[@]}" -lt 7 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <hostname> <arch> <workdir_override> <pass#> <test_level> <split|fused|aiter|v3> [partial] [--follow]

  workdir_override  Remote workdir override (empty string = use DEFAULT_WORKDIR).
  partial           Run partial tests using sel*.txt from the previous run as selectors.
  --follow          Wait for the remote tsp job to finish (tsp -t <jobid>).
                    Without this flag the job is queued and the script returns immediately.
EOF
  exit 1
fi

WORKDIR="${POSITIONAL[0]}"
HOSTNAME="${POSITIONAL[1]}"
ARCH="${POSITIONAL[2]}"
WORKDIR_OVERRIDE="${POSITIONAL[3]}"
PASS_NUM="${POSITIONAL[4]}"
TEST_LEVEL="${POSITIONAL[5]}"
BACKEND="${POSITIONAL[6]}"
VARIANT="${POSITIONAL[7]:-}"

case "$BACKEND" in
  split|fused|aiter|v3) ;;
  *) echo "Error: backend must be one of split/fused/aiter/v3, got: $BACKEND" >&2; exit 1 ;;
esac

case "${VARIANT:-}" in
  partial|"") ;;
  *) echo "Error: variant must be 'partial' or empty, got: $VARIANT" >&2; exit 1 ;;
esac

load_config "$WORKDIR"

REMOTE_WORKDIR="${WORKDIR_OVERRIDE:-$DEFAULT_WORKDIR}"

# Per-arch test install: installed/test/<arch>/lib
LIBDIR="/wkdir/installed/test/$ARCH/lib"
REMOTE_SCRIPT="/wkdir/aotriton.src/.ci/run-test.sh"
BASE_OUTPUT_DIR="/wkdir/run/tests"
if [ "${VARIANT:-}" = "partial" ]; then
  OUTPUT_DIR="$BASE_OUTPUT_DIR/partial"
  PARTIAL_INFO_DIR="$BASE_OUTPUT_DIR"
else
  OUTPUT_DIR="$BASE_OUTPUT_DIR"
  PARTIAL_INFO_DIR=""
fi

echo "[$HOSTNAME] Queuing run-test pass=$PASS_NUM level=$TEST_LEVEL backend=$BACKEND arch=$ARCH variant=${VARIANT:-normal}"
echo "[$HOSTNAME] output -> $REMOTE_WORKDIR/${OUTPUT_DIR#/wkdir/}/"

# shellcheck disable=SC2029
JOBID=$(ssh "$HOSTNAME" bash -s "$REMOTE_WORKDIR" "$CELERY_WORKER_IMAGE" \
        "$LIBDIR" "$REMOTE_SCRIPT" "$OUTPUT_DIR" "$PARTIAL_INFO_DIR" \
        "$PASS_NUM" "$TEST_LEVEL" "$BACKEND" <<'ENDSSH'
REMOTE_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"
LIBDIR="$3"
REMOTE_SCRIPT="$4"
OUTPUT_DIR="$5"
PARTIAL_INFO_DIR="$6"
PASS_NUM="$7"
TEST_LEVEL="$8"
BACKEND="$9"

mkdir -p "$REMOTE_WORKDIR/run/tests"
[ -n "$PARTIAL_INFO_DIR" ] && mkdir -p "$REMOTE_WORKDIR/${OUTPUT_DIR#/wkdir/}"

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
  ${PARTIAL_INFO_DIR:+-e PARTIAL_INFO_DIR="$PARTIAL_INFO_DIR"} \
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
  # shellcheck disable=SC2029
  ssh "$HOSTNAME" bash -s "$JOBID" <<'EOF'
JOBID="$1"
if [ "$(tsp -s "$JOBID")" = "queued" ]; then
  echo "Waiting for tsp job $JOBID to start..."
  while [ "$(tsp -s "$JOBID")" = "queued" ]; do sleep 5; done
fi
tsp -t "$JOBID"
EOF
fi
