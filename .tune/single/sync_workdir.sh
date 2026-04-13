#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Sync main workdir to one host
# Usage: sync_workdir.sh <hostname> <src_workdir> <dest_workdir>

set -e

HOSTNAME="$1"
SRC_WORKDIR="$2"
DEST_WORKDIR="$3"

# Create directory structure
ssh "$HOSTNAME" mkdir -p "$DEST_WORKDIR"

# Sync main directories (exclude build, installed, run, scratch)
rsync -az --info=progress2 \
  --exclude '/build/' \
  --exclude '/installed/' \
  --exclude '/run/' \
  --exclude '/scratch/' \
  "$SRC_WORKDIR/" "$HOSTNAME:$DEST_WORKDIR/"
