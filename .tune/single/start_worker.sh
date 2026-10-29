#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start worker on one host
# Usage: start_worker.sh <workdir> <hostname> [-- <extra args>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"
. "$TUNE_ROOT/lib/paths.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2

# Collect extra args. The '--' separator is OPTIONAL and may appear more than
# once, because callers disagree about it and always have:
#
#   wkctl:90,111          emits TWO   (MULTI_GPU_ARGS already starts with one)
#   stopstart_worker.sh   emits NONE  (it strips the one it was given)
#   webui tasks.py:341    emits one only when it also has --multi_gpu
#
# The old rule -- take the extras only if $1 is exactly '--' -- had no else
# branch, so a caller that omitted it had every extra SILENTLY DISCARDED. That
# is how `-- --multi_gpu -1 --tuning_mode op` reached the remote as plain
# defaults and a worker logged tuning_mode=kernel with no error anywhere. The
# doubled form failed differently: the surviving '--' travelled all the way to
# worker_service.sh's arg loop, which rejects it as an unknown option.
#
# So separators are accepted anywhere and dropped; no extra arg is ever a bare
# '--' (they are --multi_gpu/--category and their values).
EXTRA_ARGS=()
for _arg in "$@"; do
  [ "$_arg" = "--" ] || EXTRA_ARGS+=("$_arg")
done

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [-- <extra args>]" >&2
  echo "" >&2
  echo "  Start a worker container on <hostname> via SSH." >&2
  echo "" >&2
  echo "  WARNING: This script does NOT read GPU selection from the workers DB." >&2
  echo "  GPU assignment must be passed explicitly via extra args (e.g. -- --multi_gpu 0 1)." >&2
  echo "  Use wkctl start to apply GPU selection automatically." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# Add --hostname to extra args, and the worker's subpool (machine group inside
# its arch, host inventory in workers.db) when it has one
SUBPOOL="$(get_worker_subpool "$WORKDIR" "$HOSTNAME")"
EXTRA_ARGS=(--hostname "$HOSTNAME" "${EXTRA_ARGS[@]}" ${SUBPOOL:+--subpool "$SUBPOOL"})

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" "$CELERY_WORKER_IMAGE" "$AOTRITON_IMAGE_WORKDIR" "$AOTRITON_IMAGE_TMPFS" "$AOTRITON_IMAGE_SRC" "$AOTRITON_IMAGE_RUNDIR" "${EXTRA_ARGS[@]}" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
CELERY_WORKER_IMAGE="$3"
# The image's layout (.tune/lib/paths.sh), passed in: this runs remotely. The
# container gets it as AOTRITON_* variables, so nothing inside depends on it.
IMG_WORKDIR="$4"
IMG_TMPFS="$5"
IMG_SRC="$6"
IMG_RUNDIR="$7"
shift 7
EXTRA_ARGS=("$@")

# Read --category from EXTRA_ARGS. It is forwarded to worker_service.sh as it
# is, which validates it. Here it only selects the tune build on PYTHONPATH, as
# --tuning_mode did. (Tune-specific, kept minimal; generalising the container
# setup per application is left to the next application's change.)
CATEGORY=""
_prev=""
for _arg in "${EXTRA_ARGS[@]}"; do
  [ "$_prev" = "--category" ] && CATEGORY="$_arg"
  _prev="$_arg"
done

if [ "$CATEGORY" = "tune,op" ]; then
  WORKER_PYTHONPATH="$IMG_WORKDIR/installed/test/$ARCH/lib"
else
  WORKER_PYTHONPATH="$IMG_WORKDIR/installed/$ARCH/lib"
fi

RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

mkdir -p "$WORKER_WORKDIR/run"

if [ -f "$RUNFILE" ]; then
  echo "Worker already running or stale run file exists. Run stop first." >&2
  exit 1
fi

# `bash -lc`, never plain `bash -c`. Every shell that enters this image must be
# a LOGIN shell, because /etc/profile.d/aotriton.sh -- which activates the venv
# and exports ROCM_PATH along with ROCm's bin/ and lib/ -- is read by
# /etc/profile and by nothing else. A non-login shell skips it entirely and
# gets no ROCM_PATH, so anything it launches has to find HIP by luck.
#
# This is exactly how the worker and testrun_direct diverged: the same image,
# the same payload, but testrun_direct ran `bash -l -c` and worked while the
# worker ran `bash -c` and its testrun children died on arrival, reporting
# nothing more than a broken pipe back to the task handler.
#
# `source .../activate` below is not a substitute. It brings the venv and
# nothing else -- ROCM_PATH is not the venv's business -- which is why the
# worker looked correctly set up right until something needed ROCm.
set -x
WORKER_CONTAINER_ID=$(docker run -d \
  --init \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -e PYTHONPATH=$WORKER_PYTHONPATH \
  -e AOTRITON_WORKDIR=$IMG_WORKDIR \
  -e AOTRITON_SRC=$IMG_SRC \
  -e AOTRITON_RUNDIR=$IMG_RUNDIR \
  -e AOTRITON_TMPFS=$IMG_TMPFS \
  -e PYTHONPYCACHEPREFIX=$IMG_RUNDIR/pycache \
  --mount type=bind,source=$(realpath $WORKER_WORKDIR),target=$IMG_WORKDIR \
  "$CELERY_WORKER_IMAGE" \
  bash -lc "source \$AOTRITON_WORKDIR/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd \$AOTRITON_SRC && bash .tune/remote/install_aotriton_pkg.sh && bash .tune/remote/worker_service.sh start \$AOTRITON_WORKDIR $ARCH ${EXTRA_ARGS[*]} && exec sleep infinity")

if [ -z "$WORKER_CONTAINER_ID" ]; then
  echo "Failed to start container" >&2
  exit 1
fi

echo "$WORKER_CONTAINER_ID" > "$RUNFILE"
echo "Started container: $WORKER_CONTAINER_ID"
EOF
