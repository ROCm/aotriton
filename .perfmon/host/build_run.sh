#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs on a build host, over ssh, from .perfmon/bin/build_subjects --host.
# Activates the shared venv of one ROCm (its compiler and libraries; node_env.sh)
# and runs build_subjects there.
#
# Usage: build_run.sh <workdir> <scratch> <rocm> [build_subjects options...]

set -eo pipefail

PM_WORKDIR="$1"
PM_SCRATCH="$2"
PM_ROCM="$3"
shift 3
export PM_WORKDIR PM_SCRATCH PM_ROCM

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
perfmon_venv_activate
exec bash "$PM_CHECKOUT/.perfmon/bin/build_subjects" "$@"
