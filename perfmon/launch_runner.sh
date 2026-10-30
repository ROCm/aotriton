#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Starts the runner of one subject on one GPU. The Python side spawns this, not
# the binary, and talks to it over stdin and stdout.
#
# Usage: launch_runner.sh --subject <name> --module <module> --gpu <id> [--arch <arch>]
#   --subject   a directory name under installed/perfmon/<arch>/, e.g.
#               aotriton-0.13b+rocm10.1.0
#   --module    the family; only "flash" has a runner
#   --gpu       the GPU ordinal on this host
#   --arch      the arch directory; if absent it is the one directory under
#               installed/perfmon/ that holds <name>, and it is an error when
#               there is none or more than one
#
# installed/perfmon/ is under $AOTRITON_WORKDIR (.def/lib/paths.sh).
#
# `exec` is load-bearing: it makes the runner the direct child of the caller,
# so crash detection, pids and EOF shutdown work as if the caller had started
# the binary itself.
#
# No LD_LIBRARY_PATH is set. The runner finds every library through the RUNPATHs
# it was linked with (the adapter and the release's runtime in the subject, the
# core in installed/perfmon/core, the ROCm runtime packages), and a variable
# would let a stray copy win over the subject's own.

set -euo pipefail

SUBJECT=""
MODULE=""
GPU=""
ARCH=""

while [ $# -gt 0 ]; do
  case "$1" in
    --subject) SUBJECT="$2"; shift 2 ;;
    --module)  MODULE="$2"; shift 2 ;;
    --gpu)     GPU="$2"; shift 2 ;;
    --arch)    ARCH="$2"; shift 2 ;;
    *) echo "launch_runner.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${SUBJECT}" ] || [ -z "${MODULE}" ] || [ -z "${GPU}" ]; then
  echo "Usage: $0 --subject <name> --module <module> --gpu <id> [--arch <arch>]" >&2
  exit 2
fi
if [ "${MODULE}" != "flash" ]; then
  echo "launch_runner.sh: no runner for module '${MODULE}' (only flash)." >&2
  exit 2
fi
case "${SUBJECT}" in
  */*|.*|"") echo "launch_runner.sh: '${SUBJECT}' is not a subject name." >&2; exit 2 ;;
esac

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.def/lib/paths.sh"
ROOT="$AOTRITON_WORKDIR/installed/perfmon"

if [ -z "${ARCH}" ]; then
  MATCHES=()
  for dir in "${ROOT}"/*/"${SUBJECT}"; do
    [ -d "${dir}" ] && MATCHES+=("${dir}")
  done
  if [ "${#MATCHES[@]}" -eq 0 ]; then
    echo "launch_runner.sh: no subject '${SUBJECT}' under ${ROOT}/*/." >&2
    exit 1
  elif [ "${#MATCHES[@]}" -gt 1 ]; then
    echo "launch_runner.sh: subject '${SUBJECT}' is under more than one arch;" \
         "pass --arch. Found: ${MATCHES[*]}" >&2
    exit 1
  fi
  SUBJECT_DIR="${MATCHES[0]}"
else
  SUBJECT_DIR="${ROOT}/${ARCH}/${SUBJECT}"
fi

RUNNER="${SUBJECT_DIR}/bin/runner"
if [ ! -x "${RUNNER}" ]; then
  echo "launch_runner.sh: no runner at ${RUNNER}." >&2
  exit 1
fi

# HIP_VISIBLE_DEVICES alone. ROCR_VISIBLE_DEVICES filters first, at the ROCr
# layer, and HIP then indexes into what is left, so setting both to N asks for
# device N of a one-device list: only GPU 0 would work. An inherited one is
# cleared too, since it would silently change which GPU --gpu means. The runner
# uses device 0 of what it sees.
export HIP_VISIBLE_DEVICES="${GPU}"
unset ROCR_VISIBLE_DEVICES

exec "${RUNNER}"
