#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run .ci/run-test.sh on a single tester host via tsp (SSH-disconnect tolerant).
#
# Usage:
#   run-test.sh --workdir <workdir> --hostname <host> --arch <arch>
#               --pass <pass#> --test_level <level> --backend <split|fused|aiter|v3>
#               [--workdir_override <path>] [--variant partial] [--follow]
#
#   --workdir_override  Remote workdir override (empty = use DEFAULT_WORKDIR from config.rc)
#   --variant partial   Sets PARTIAL_INFO_DIR and routes output to partial/
#   --variant partial_adiffs  As partial, plus RECORD_ADIFFS_TO
#   --ref_device_policy <p>   cpu|cuda|default -> AOTRITON_REF_DEVICE_OPTION.
#                             Orthogonal to --variant: it selects the device the
#                             REFERENCE is computed on, not what is run.
#   --follow            Wait for the tsp job to complete; default is fire-and-forget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"

WORKDIR=""
HOSTNAME=""
ARCH=""
WORKDIR_OVERRIDE=""
PASS_NUM=""
TEST_LEVEL=""
BACKEND=""
VARIANT=""
REF_DEVICE_POLICY=""
ADIFF=0
FOLLOW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir)          WORKDIR="$2";          shift 2 ;;
    --hostname)         HOSTNAME="$2";         shift 2 ;;
    --arch)             ARCH="$2";             shift 2 ;;
    --workdir_override) WORKDIR_OVERRIDE="$2"; shift 2 ;;
    --pass)             PASS_NUM="$2";         shift 2 ;;
    --test_level)       TEST_LEVEL="$2";       shift 2 ;;
    --backend)          BACKEND="$2";          shift 2 ;;
    --variant)          VARIANT="$2";          shift 2 ;;
    --ref_device_policy) REF_DEVICE_POLICY="$2"; shift 2 ;;
    --adiff)            ADIFF=1;               shift ;;
    --follow)           FOLLOW=1;              shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

_missing=()
[ -z "$WORKDIR"    ] && _missing+=(--workdir)
[ -z "$HOSTNAME"   ] && _missing+=(--hostname)
[ -z "$ARCH"       ] && _missing+=(--arch)
[ -z "$PASS_NUM"   ] && _missing+=(--pass)
[ -z "$TEST_LEVEL" ] && _missing+=(--test_level)
[ -z "$BACKEND"    ] && _missing+=(--backend)
if [ "${#_missing[@]}" -gt 0 ]; then
  echo "Error: missing required arguments: ${_missing[*]}" >&2
  cat >&2 <<EOF
Usage: $0 --workdir <workdir> --hostname <host> --arch <arch>
          --pass <pass#> --test_level <level> --backend <split|fused|aiter|v3>
          [--workdir_override <path>] [--variant partial] [--follow]
EOF
  exit 1
fi

case "$BACKEND" in
  split|fused|aiter|v3) ;;
  *) echo "Error: backend must be one of split/fused/aiter/v3, got: $BACKEND" >&2; exit 1 ;;
esac

case "${VARIANT:-}" in
  partial|partial_adiffs|"") ;;
  *) echo "Error: variant must be 'partial', 'partial_adiffs', or empty, got: $VARIANT" >&2; exit 1 ;;
esac

load_config "$WORKDIR"

REMOTE_WORKDIR="${WORKDIR_OVERRIDE:-$DEFAULT_WORKDIR}"

# Per-arch test install: installed/test/<arch>/lib
LIBDIR="/wkdir/installed/test/$ARCH/lib"
if [ "$ADIFF" -eq 1 ]; then
  REMOTE_SCRIPT="/wkdir/aotriton.src/.ci/run-ci-test.sh"
else
  REMOTE_SCRIPT="/wkdir/aotriton.src/.ci/run-test.sh"
fi
BASE_OUTPUT_DIR="/wkdir/run/tests"
case "${VARIANT:-}" in
  partial|partial_adiffs) OUTPUT_DIR="$BASE_OUTPUT_DIR/partial" ;;
  *)                      OUTPUT_DIR="$BASE_OUTPUT_DIR" ;;
esac
if [ "$OUTPUT_DIR" = "$BASE_OUTPUT_DIR" ]; then
  PARTIAL_INFO_DIR=""
else
  PARTIAL_INFO_DIR="$BASE_OUTPUT_DIR"
fi
if [ "${VARIANT:-}" = "partial_adiffs" ]; then
  RECORD_ADIFFS_TO="$OUTPUT_DIR/adiffs.txt"
else
  RECORD_ADIFFS_TO=""
fi
# Forwarded as AOTRITON_REF_DEVICE_OPTION, which tops the device-policy
# precedence in SdpaContext.create_ref_inputs. 'cpu' is the way round a GPU-side
# reference that cannot be trusted -- gfx1201's torch math_sdp segfaults on some
# shapes and hands the test a bad oracle, so the failure says nothing about the
# kernel under test.
case "${REF_DEVICE_POLICY:-}" in
  cpu|cuda|default|"") REF_DEVICE_OPTION="$REF_DEVICE_POLICY" ;;
  *) echo "Error: --ref_device_policy must be 'cpu', 'cuda', 'default', or empty, got: $REF_DEVICE_POLICY" >&2; exit 1 ;;
esac

echo "[$HOSTNAME] Queuing run-test pass=$PASS_NUM level=$TEST_LEVEL backend=$BACKEND arch=$ARCH variant=${VARIANT:-normal} ref_device_policy=${REF_DEVICE_POLICY:-default}"
echo "[$HOSTNAME] output -> $REMOTE_WORKDIR/${OUTPUT_DIR#/wkdir/}/"

# shellcheck disable=SC2029
JOBID=$(ssh "$HOSTNAME" bash -s "$REMOTE_WORKDIR" "$CELERY_WORKER_IMAGE" \
        "$LIBDIR" "$REMOTE_SCRIPT" "$OUTPUT_DIR" \
        "$PASS_NUM" "$TEST_LEVEL" "$BACKEND" "$PARTIAL_INFO_DIR" "$RECORD_ADIFFS_TO" \
        "$REF_DEVICE_OPTION" <<'ENDSSH'
REMOTE_WORKDIR="$1"
CELERY_WORKER_IMAGE="$2"
LIBDIR="$3"
REMOTE_SCRIPT="$4"
OUTPUT_DIR="$5"
PASS_NUM="$6"
TEST_LEVEL="$7"
BACKEND="$8"
PARTIAL_INFO_DIR="$9"
RECORD_ADIFFS_TO="${10}"
REF_DEVICE_OPTION="${11}"

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
  -e PYTHONPYCACHEPREFIX=/wkdir/run/pycache \
  -e AOTRITON_TEST_LIBDIR="$LIBDIR" \
  -e OUTPUT_DIR="$OUTPUT_DIR" \
  ${PARTIAL_INFO_DIR:+-e PARTIAL_INFO_DIR="$PARTIAL_INFO_DIR"} \
  ${RECORD_ADIFFS_TO:+-e RECORD_ADIFFS_TO="$RECORD_ADIFFS_TO"} \
  ${REF_DEVICE_OPTION:+-e AOTRITON_REF_DEVICE_OPTION="$REF_DEVICE_OPTION"} \
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
