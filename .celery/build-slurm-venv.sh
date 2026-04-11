#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Python virtual environment for SLURM workers

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir>

Build Python virtual environment for SLURM workers.

Arguments:
  <workdir>  Project working directory

This script will:
  - SSH to SLURM_LOGIN_NODE (from config.rc)
  - Create venv at SLURM_WORKER_DIR/installed/venv
  - Install PyTorch from official ROCm repository
  - Install Triton wheel from <workdir>/scratch/triton/
  - Install requirements-tuning.txt
  - Apply Celery patches and install amdsmi

Prerequisites:
  - SLURM_LOGIN_NODE and SLURM_WORKER_DIR set in config.rc
  - Triton wheel built via build-for-tuning.sh
EOF
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
AOTRITON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="$1"
CONFIG_RC="$WORKDIR/config.rc"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$CONFIG_RC" ]; then
  echo "Error: Invalid workdir or config.rc not found" >&2
  exit 1
fi

# Source config
. "$CONFIG_RC"

if [ -z "$SLURM_LOGIN_NODE" ]; then
  echo "Error: SLURM_LOGIN_NODE not set in config.rc. SLURM not enabled." >&2
  exit 1
fi

if [ -z "$SLURM_WORKER_DIR" ]; then
  echo "Error: SLURM_WORKER_DIR not set in config.rc" >&2
  exit 1
fi

# Build venv on SLURM login node
echo "Building venv on $SLURM_LOGIN_NODE:$SLURM_WORKER_DIR/installed/venv ..."
ssh "$SLURM_LOGIN_NODE" bash -s "$SLURM_WORKER_DIR" <<'EOF'
SLURM_WORKER_DIR="$1"
VENV_DIR="$SLURM_WORKER_DIR/installed/venv"

export CONFIG_RC="$SLURM_WORKER_DIR/config.rc"
source "$CONFIG_RC"

set -e

# Load SLURM modules
for module in "${SLURM_MODULES[@]}"; do
  module load "$module"
done

# Create venv
echo "Creating venv at $VENV_DIR"
mkdir -p "$SLURM_WORKER_DIR/installed"
python3 -m venv "$VENV_DIR"

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch from official ROCm repository
# TODO: Make ROCm version configurable
echo "Installing PyTorch from official ROCm repository..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

# Install requirements
echo "Installing requirements-tuning.txt..."
pip install -r "$SLURM_WORKER_DIR/aotriton.src/requirements-tuning.txt"

echo "Venv built successfully at $VENV_DIR"

# Set CELERY_WORKER_PYTHON for patch scripts
export CELERY_TO_PATCH_PYTHON="$VENV_DIR/bin/python"
export CONFIG_RC="$SLURM_WORKER_DIR/config.rc"

# Run patch scripts
echo "Patching Celery..."
bash "$SLURM_WORKER_DIR/image.scripts/01-patch_celery.sh"

echo "Uninstalling old amdsmi..."
pip uninstall amdsmi || true

echo "Installing amdsmi..."
bash "$SLURM_WORKER_DIR/image.scripts/02-install_amdsmi.sh"

echo "Patches applied successfully"
EOF

if [ $? -ne 0 ]; then
  echo "Error: Failed to build venv" >&2
  exit 1
fi

echo "SLURM venv build completed successfully"
