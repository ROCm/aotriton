#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir>

Prepare working directory on dev node.

Arguments:
  <workdir>  Project working directory (created by create-project-directory.sh)

This script will:
  1. Clone or update AOTriton source to <workdir>/aotriton.src
  2. Copy image.scripts from .celery to <workdir>
  3. Create <workdir>/image.build/Dockerfile
EOF
  exit 1
fi

set -x

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
AOTRITON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR=$(realpath "$1")

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Step 1: Clone or update AOTriton source
AOTRITON_SRC="$WORKDIR/aotriton.src"

cd "$AOTRITON_ROOT"

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ -z "$CURRENT_BRANCH" ]; then
  echo "Error: Cannot determine current branch" >&2
  exit 1
fi

# Check if current branch is upstream/main
IS_UPSTREAM_MAIN=false
if [ "$CURRENT_BRANCH" = "main" ] && git rev-parse --verify upstream/main >/dev/null 2>&1 && [ "$(git rev-parse HEAD)" = "$(git rev-parse upstream/main)" ]; then
  IS_UPSTREAM_MAIN=true
  CLONE_DEPTH=3
fi

# If not upstream/main, verify upstream/main is in history
if [ "$IS_UPSTREAM_MAIN" = false ]; then
  if ! git merge-base --is-ancestor upstream/main HEAD 2>/dev/null; then
    echo "Error: upstream/main is not in current branch's history" >&2
    echo "Current branch: $CURRENT_BRANCH" >&2
    exit 1
  fi
  # Calculate depth from upstream/main to HEAD
  CLONE_DEPTH=$(git rev-list --count upstream/main..HEAD)
  CLONE_DEPTH=$((CLONE_DEPTH + 1))
fi

# Try to update existing clone or create new one
if [ -d "$AOTRITON_SRC/.git" ]; then
  echo "Updating existing AOTriton source..."
  cd "$AOTRITON_SRC"
  if ! git pull --rebase origin "$CURRENT_BRANCH" 2>/dev/null; then
    echo "Pull failed (possibly due to rebase), removing and re-cloning..."
    cd "$AOTRITON_ROOT"
    rm -rf "$AOTRITON_SRC"
    NEED_CLONE=true
  else
    NEED_CLONE=false
  fi
else
  NEED_CLONE=true
fi

if [ "$NEED_CLONE" = true ]; then
  echo "Cloning from local repository (branch: $CURRENT_BRANCH, depth: $CLONE_DEPTH)..."
  cd "$AOTRITON_ROOT"
  git clone --depth "$CLONE_DEPTH" --branch "$CURRENT_BRANCH" "file://$(pwd)" "$AOTRITON_SRC"

  if [ $? -ne 0 ]; then
    echo "Error: Failed to clone AOTriton source" >&2
    exit 1
  fi
fi

# Step 2: Sync image.scripts directory
echo "Syncing image.scripts..."
rsync -a "$SCRIPT_DIR/image.scripts/" "$WORKDIR/image.scripts/"

# Step 3: Create Dockerfile
echo "Creating Dockerfile..."
bash "$SCRIPT_DIR/create-Dockerfile.sh" "$WORKDIR"

echo "Working directory preparation completed"
