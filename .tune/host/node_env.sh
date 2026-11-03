#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs on a fleet peer (the host dir: where it executes). Source it, with
#   TN_WORKDIR    the host's workdir: config.rc and installed/<arch>/ (the tuning
#                 build) live there
#   TN_ARCH       the arch of the build to put on PYTHONPATH (installed/<arch>/lib)
# set. It defines the environment a tune node needs outside the tuning image
# (the path variables of .def/lib/paths.sh), and uses the venv that is already
# active (a claudepak-managed one). The checkout is the one this file lives in,
# which need not be the project checkout of the host: a worktree under the
# scratch works.
#
# The rundir is node-local and tune's own (aotriton-tune), so that a tune node
# and a node of another application on the same peer (perfmon's is
# aotriton-perfmon) never share pid files or the broker socket.
#
#   tune_env_setup   export the variables
#   tune_env_print   the exports a node needs, for a human or an `eval`

TN_CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

tune_env_setup() {
    export AOTRITON_WORKDIR="$TN_WORKDIR"
    export AOTRITON_SRC="$TN_CHECKOUT"
    export AOTRITON_RUNDIR="${XDG_RUNTIME_DIR:-/tmp}/aotriton-tune"
    mkdir -p "$AOTRITON_RUNDIR"
    # Memory-backed, node-local, and not the other application's.
    local default_tmpfs
    default_tmpfs="$(. "$TN_CHECKOUT/.def/lib/paths.sh"; echo "$AOTRITON_TMPFS")"
    if [ -d "$default_tmpfs" ] && [ -w "$default_tmpfs" ]; then
        export AOTRITON_TMPFS="$default_tmpfs/aotriton-tune"
    else
        export AOTRITON_TMPFS="$AOTRITON_RUNDIR/tmpfs"
    fi
    mkdir -p "$AOTRITON_TMPFS"
    # aotriton is used from the checkout, not installed; pyaotriton (the
    # tuning build) comes from the workdir, as it does in the tuning image.
    export PYTHONPATH="$TN_CHECKOUT/.def/lib/pkgroot:$TN_WORKDIR/installed/$TN_ARCH/lib${PYTHONPATH:+:$PYTHONPATH}"
    export AOTRITON_MODULES_DIR="$TN_CHECKOUT/modules"
    export PYTHONPYCACHEPREFIX="$AOTRITON_RUNDIR/pycache"
    export PYTHONDONTWRITEBYTECODE=1
    if [ -n "${TN_THERMAL_GATE:-}" ]; then
        export AOTRITON_THERMAL_GATE="$TN_THERMAL_GATE"
    fi
    # The ROCm wheels keep rocminfo and rocm_agent_enumerator out of the
    # venv's bin/; rocm-sdk knows where they are.
    if command -v rocm-sdk >/dev/null 2>&1; then
        export ROCM_PATH="$(rocm-sdk path --root)"
        export PATH="$(rocm-sdk path --bin):$PATH"
    fi
    # worker_service.sh starts every process with $CELERY_WORKER_PYTHON, which
    # config.rc points into the tuning image. It honours AOTRITON_PYTHON over it.
    export AOTRITON_PYTHON="$(command -v python)"
}

tune_env_print() {
    cat <<PRINT
export AOTRITON_WORKDIR=$AOTRITON_WORKDIR
export AOTRITON_SRC=$AOTRITON_SRC
export AOTRITON_RUNDIR=$AOTRITON_RUNDIR
export AOTRITON_TMPFS=$AOTRITON_TMPFS
export PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX
export PYTHONPATH=$PYTHONPATH
export AOTRITON_MODULES_DIR=$AOTRITON_MODULES_DIR
PRINT
}
