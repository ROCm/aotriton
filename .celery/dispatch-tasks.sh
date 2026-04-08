#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Wrapper script for v3python/tune/dispatch_tasks.py
# This script sets up the Python environment and calls the actual dispatch script

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

# Get the directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOTRITON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get workdir from first argument
if [ "$#" -lt 2 ]; then
  echo "Error: Missing workdir argument" >&2
  echo "Usage: $0 <module> <workdir> [module-options]" >&2
  exit 1
fi

WORKDIR="$2"

# Validate workdir
if [ ! -d "$WORKDIR" ]; then
  echo "Error: Working directory does not exist: $WORKDIR" >&2
  exit 1
fi

# Convert to absolute path
WORKDIR=$(cd "$WORKDIR" && pwd)

# Set up venv path
VENV_DIR="$WORKDIR/scratch/venv.devnode"
VENV_PYTHON="$VENV_DIR/bin/python"

# Create venv if it doesn't exist
if [ ! -f "$VENV_PYTHON" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  mkdir -p "$WORKDIR/scratch"
  python3 -m venv "$VENV_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment" >&2
    exit 1
  fi
fi

# Install requirements if needed
REQUIREMENTS_FILE="$AOTRITON_ROOT/requirements-tuning.txt"
INSTALLED_MARKER="$VENV_DIR/.requirements_installed"

if [ ! -f "$INSTALLED_MARKER" ] || [ "$REQUIREMENTS_FILE" -nt "$INSTALLED_MARKER" ]; then
  echo "Installing requirements from $REQUIREMENTS_FILE..."
  "$VENV_PYTHON" -m pip install -q --upgrade pip
  "$VENV_PYTHON" -m pip install -q -r "$REQUIREMENTS_FILE"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements" >&2
    exit 1
  fi
  touch "$INSTALLED_MARKER"
fi

# Apply celery patch if needed
PATCH_MARKER="$VENV_DIR/.celery_patched"

if [ ! -f "$PATCH_MARKER" ]; then
  echo "Applying celery patch..."

  # Set CELERY_TO_PATCH_PYTHON to patch the venv's celery installation
  CONFIG_RC="$WORKDIR/config.rc" CELERY_TO_PATCH_PYTHON="$VENV_PYTHON" bash "$SCRIPT_DIR/image.scripts/01-patch_celery.sh"
  PATCH_STATUS=$?

  if [ $PATCH_STATUS -eq 0 ]; then
    touch "$PATCH_MARKER"
  else
    echo "Error: Failed to apply celery patch" >&2
    exit 1
  fi
fi

# Add aotriton root to PYTHONPATH so v3python can be imported
export PYTHONPATH="${AOTRITON_ROOT}:${PYTHONPATH}"

# Execute the actual dispatch script as a module with the venv python
cd "$AOTRITON_ROOT"
exec "$VENV_PYTHON" -m v3python.tune.dispatch_tasks "$@"
