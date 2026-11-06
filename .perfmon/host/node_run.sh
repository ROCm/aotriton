#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs on a remote GPU host, over ssh, from .perfmon/bin/node. Sets up the node
# environment (node_env.sh: the venv, the path variables, the thermal
# gate) and runs the generic node starter in it, rootless and with no docker.
#
# Usage: node_run.sh <workdir> <scratch> <rocm> <thermal_gate on|off|""> \
#                    <pghost|""> <pgport|""> <start|stop|status> <arch> \
#                    [worker_service.sh options...]
#
# pghost and pgport (read on the server, empty for a host that reaches
# PostgreSQL directly) become PGHOST and PGPORT for worker_service.sh.

set -eo pipefail

PM_WORKDIR="$1"
PM_SCRATCH="$2"
PM_ROCM="$3"
PM_THERMAL_GATE="$4"
PM_PGHOST="$5"
PM_PGPORT="$6"
COMMAND="$7"
ARCH="$8"
shift 8
export PM_WORKDIR PM_SCRATCH PM_ROCM PM_THERMAL_GATE

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
perfmon_env_setup
perfmon_venv_activate

# worker_service.sh keeps a preset PGHOST/PGPORT over config.rc's.
if [ -n "$PM_PGHOST" ]; then
    export PGHOST="$PM_PGHOST"
    export PGPORT="$PM_PGPORT"
fi

# worker_service.sh reads config.rc itself (database host and credentials).
# Not exec: its daemons are detached by it, and its exit status is the answer.
bash "$PM_CHECKOUT/.def/container/worker_service.sh" "$COMMAND" "$PM_WORKDIR" "$ARCH" "$@"
