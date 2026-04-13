#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Sync architecture-specific files to one host
# Usage: sync_arch.sh <hostname> <arch> <src_workdir> <dest_workdir>

set -e

HOSTNAME="$1"
ARCH="$2"
SRC_WORKDIR="$3"
DEST_WORKDIR="$4"

if [ "$ARCH" = "ALL" ]; then
    SUBDIR=""
else
    SUBDIR="/$ARCH"
fi

if [ -d "$SRC_WORKDIR/installed$SUBDIR" ]; then
    rsync -azR --info=progress2 \
        "$SRC_WORKDIR/./installed$SUBDIR" \
        "$HOSTNAME:$DEST_WORKDIR/./"
fi
