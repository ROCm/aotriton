#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install amdsmi to Python virtual environment
#
# This script handles both classical ROCm and theRock (ROCm >= 7.10) installations.
# - Classical ROCm: Install from /opt/rocm/share/amd_smi
# - theRock: Install from _rocm_sdk_core/share/amd_smi (located via rocm_sdk_core package)
#
# Note: theRock images have /opt/venv with ROCm SDK installed, but amdsmi is not
# pre-installed in the venv and must be installed separately.

CONFIG_RC="${CONFIG_RC:-/config.rc}"

if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# Source config to get CELERY_WORKER_PYTHON
. "$CONFIG_RC"

# CELERY_TO_PATCH_PYTHON can override CELERY_WORKER_PYTHON
CELERY_TO_PATCH_PYTHON="${CELERY_TO_PATCH_PYTHON:-$CELERY_WORKER_PYTHON}"

if [ -z "$CELERY_TO_PATCH_PYTHON" ]; then
  echo "Error: CELERY_TO_PATCH_PYTHON not set" >&2
  exit 1
fi

# Activate venv
VENV_ACTIVATE="$(dirname "$CELERY_TO_PATCH_PYTHON")/activate"
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Error: Virtual environment activate script not found at $VENV_ACTIVATE" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

# Detect ROCm type and install amdsmi
if [ -d "/opt/rocm" ]; then
  # Classical ROCm installation
  echo "Detected classical ROCm installation"
  AMDSMI_DIR=$(hipconfig --rocmpath)/share/amd_smi

  if [ ! -d "$AMDSMI_DIR" ]; then
    echo "Error: amdsmi directory not found at $AMDSMI_DIR" >&2
    exit 1
  fi

  echo "Installing amdsmi from $AMDSMI_DIR"
  # FIXME: This does not work on certain SLURM configurations where
  #        amdsmi.egg-info already exists under $AMDSMI_DIR.
  #        Can be fixed with
  #           TGT=$SLURM_WORKER_DIR/scratch/amd_smi
  #           cp -r $AMDSMI_DIR $TGT
  #           (cd $TGT; # pip install .)
  (cd "$AMDSMI_DIR" && pip install .)
else
  # theRock installation (ROCm >= 7.10)
  echo "Detected theRock installation (ROCm >= 7.10)"

  # Locate amdsmi via rocm_sdk_core package
  # FIXME: AMDSMI_DIR=$(hipconfig --rocmpath)/share/amd_smi also works
  AMDSMI_DIR=$(python -c "
import rocm_sdk_core
from pathlib import Path
sdk_path = Path(rocm_sdk_core.__file__).parent.parent / '_rocm_sdk_core' / 'share' / 'amd_smi'
print(sdk_path.as_posix())
" 2>/dev/null)

  if [ -z "$AMDSMI_DIR" ] || [ ! -d "$AMDSMI_DIR" ]; then
    echo "Error: Could not locate amdsmi directory via rocm_sdk_core package" >&2
    echo "Expected path pattern: .../site-packages/_rocm_sdk_core/share/amd_smi" >&2
    exit 1
  fi

  echo "Installing amdsmi from $AMDSMI_DIR"
  (cd "$AMDSMI_DIR" && pip install .)
fi

echo "amdsmi installation completed successfully"
