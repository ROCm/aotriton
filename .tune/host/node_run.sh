#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs on a fleet peer, over ssh, from .tune/bin/node. Sets up the node
# environment (node_env.sh) and runs the generic node starter in it, rootless
# and with no docker.
#
# Usage: node_run.sh <host workdir> <thermal_gate on|off|""> <pghost|""> <pgport|""> \
#                    <start|stop|status> <arch> [worker_service.sh options...]
#
# pghost and pgport (read on the server, empty for a host that reaches
# PostgreSQL directly) become PGHOST and PGPORT for worker_service.sh.

set -eo pipefail

TN_WORKDIR="$1"
TN_THERMAL_GATE="$2"
TN_PGHOST="$3"
TN_PGPORT="$4"
COMMAND="$5"
TN_ARCH="$6"
shift 6
export TN_WORKDIR TN_THERMAL_GATE TN_ARCH

# The op level runs on the plain testing library (installed/test/<arch>), the
# kernel level on the tuning library (installed/<arch>); the level is the
# second part of --category.
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]} - 1; i++)); do
    if [ "${ARGS[$i]}" = --category ] && [[ "${ARGS[$((i + 1))]}" == *,op ]]; then
        export TN_INSTALL_SUBDIR="installed/test/$TN_ARCH"
    fi
done

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
tune_env_setup

# worker_service.sh keeps a preset PGHOST/PGPORT over config.rc's.
if [ -n "$TN_PGHOST" ]; then
    export PGHOST="$TN_PGHOST"
    export PGPORT="$TN_PGPORT"
fi

# worker_service.sh reads config.rc itself (database host and credentials).
# Not exec: its daemons are detached by it, and its exit status is the answer.
bash "$TN_CHECKOUT/.def/container/worker_service.sh" "$COMMAND" "$TN_WORKDIR" "$TN_ARCH" "$@"
