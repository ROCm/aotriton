#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs on a fleet peer (the host dir: where it executes). Source it, with
#   PM_WORKDIR    the perfmon workdir (the scratch's wkdir.<name>)
#   PM_SCRATCH    the shared scratch
#   PM_ROCM       the ROCm version of the shared venv
#   PM_THERMAL_GATE  optional, on|off (workers.db thermal_gate of this host)
# set. It defines the environment a perfmon node needs (the path variables of
# .def/lib/paths.sh, automation doc 2.2) and activates the shared venv. The
# checkout is the one this file lives in: a peer mounts it at the same path as
# the control host.
#
#   perfmon_env_setup      export the variables; needs no venv
#   perfmon_venv_activate  activate the venv and put ROCm's bin on PATH
#   perfmon_env_print      the exports a node needs, for a human or an `eval`

PM_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

perfmon_venv_dir() {
    echo "$PM_SCRATCH/perfmon/venv/rocm$PM_ROCM"
}

perfmon_env_setup() {
    export AOTRITON_WORKDIR="$PM_WORKDIR"
    export AOTRITON_SRC="$PM_CHECKOUT"
    # Node-local: the broker socket and pid files cannot live on the scratch.
    export AOTRITON_RUNDIR="${XDG_RUNTIME_DIR:-/tmp}/aotriton-perfmon"
    mkdir -p "$AOTRITON_RUNDIR"
    # Memory-backed, node-local; /dev/shm may not be writable in a container.
    if [ -d /dev/shm ] && [ -w /dev/shm ]; then
        export AOTRITON_TMPFS=/dev/shm
    else
        export AOTRITON_TMPFS="$AOTRITON_RUNDIR/tmpfs"
        mkdir -p "$AOTRITON_TMPFS"
    fi
    export PYTHONPYCACHEPREFIX="$AOTRITON_RUNDIR/pycache"
    export PYTHONDONTWRITEBYTECODE=1
    if [ -n "${PM_THERMAL_GATE:-}" ]; then
        export AOTRITON_THERMAL_GATE="$PM_THERMAL_GATE"
    fi
}

perfmon_venv_activate() {
    local venv
    venv="$(perfmon_venv_dir)"
    if [ ! -f "$venv/bin/activate" ]; then
        echo "Error: shared venv $venv does not exist; run .perfmon/bin/setup_session first." >&2
        return 1
    fi
    . "$venv/bin/activate"
    # The ROCm wheels keep rocminfo and rocm_agent_enumerator out of the venv's
    # bin/; rocm-sdk knows where they are.
    if command -v rocm-sdk >/dev/null 2>&1; then
        export ROCM_PATH="$(rocm-sdk path --root)"
        export PATH="$(rocm-sdk path --bin):$PATH"
    fi
    # worker_service.sh starts every process with $CELERY_WORKER_PYTHON, which a
    # config.rc written for the tuning image points into that image. It
    # honours AOTRITON_PYTHON over it.
    export AOTRITON_PYTHON="$venv/bin/python"
}

perfmon_env_print() {
    cat <<PRINT
export AOTRITON_WORKDIR=$AOTRITON_WORKDIR
export AOTRITON_SRC=$AOTRITON_SRC
export AOTRITON_RUNDIR=$AOTRITON_RUNDIR
export AOTRITON_TMPFS=$AOTRITON_TMPFS
export PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX
export AOTRITON_THERMAL_GATE=${AOTRITON_THERMAL_GATE:-off}   # per host: workers.db thermal_gate
. $(perfmon_venv_dir)/bin/activate
PRINT
}
