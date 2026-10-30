#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Every filesystem location the tooling assumes, in one place. Source it; it
# defines and exports the variables below, keeping any value already set.
# python/pgdef/paths.py is the Python side: the same variables, the same
# defaults. Keep the two in sync.
#
#   AOTRITON_WORKDIR        the workdir (config.rc, installed/, run/)
#   AOTRITON_SRC            the checkout the node runs from
#   AOTRITON_RUNDIR         node-local runtime files: sockets, pid files,
#                           pycache. Must be a local filesystem.
#   AOTRITON_TMPFS          fast (memory-backed) scratch for prepared data
#   AOTRITON_BROKER_SOCKET  the localq broker's Unix socket. worker_service.sh
#                           always puts it under its rundir; perfmon points
#                           AOTRITON_RUNDIR at a node-local path for that.
#
# No other script spells these locations.
#
# The tuning image's layout. Scripts that start a container from
# CELERY_WORKER_IMAGE mount the host workdir at AOTRITON_IMAGE_WORKDIR and pass
# the AOTRITON_* variables into the container explicitly, so nothing inside
# depends on where the mount is. These describe the image, not this host, and
# are therefore independent of this host's AOTRITON_WORKDIR. Not exported.
#
#   AOTRITON_IMAGE_WORKDIR  where the workdir is mounted inside the image
#   AOTRITON_IMAGE_SRC      the checkout inside the image, under its workdir
#   AOTRITON_IMAGE_RUNDIR   the image's rundir, under its workdir
#   AOTRITON_IMAGE_TMPFS    the image's tmpfs (shared with the host: --ipc=host)

: "${AOTRITON_IMAGE_WORKDIR:=/wkdir}"
: "${AOTRITON_IMAGE_SRC:=$AOTRITON_IMAGE_WORKDIR/aotriton.src}"
: "${AOTRITON_IMAGE_RUNDIR:=$AOTRITON_IMAGE_WORKDIR/run}"
: "${AOTRITON_IMAGE_TMPFS:=/dev/shm}"

: "${AOTRITON_WORKDIR:=$AOTRITON_IMAGE_WORKDIR}"
: "${AOTRITON_SRC:=$AOTRITON_WORKDIR/aotriton.src}"
: "${AOTRITON_RUNDIR:=$AOTRITON_WORKDIR/run}"
: "${AOTRITON_TMPFS:=$AOTRITON_IMAGE_TMPFS}"
: "${AOTRITON_BROKER_SOCKET:=/tmp/aotriton-broker.sock}"

export AOTRITON_WORKDIR AOTRITON_SRC AOTRITON_RUNDIR AOTRITON_TMPFS AOTRITON_BROKER_SOCKET
