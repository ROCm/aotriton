#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Detect GPU information from a worker host
# Usage: detect_gpu.sh <workdir> <hostname>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"

WORKDIR="$1"
HOSTNAME="$2"

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname>" >&2
  exit 1
fi

load_config "$WORKDIR"

echo "Detecting GPU info from $HOSTNAME..."

# Run amd-smi static --json inside Docker container
# Disable set -e temporarily to capture exit code
set +e
JSON_OUTPUT=$(ssh "$HOSTNAME" docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  "$CELERY_WORKER_IMAGE" \
  amd-smi static --json 2>&1)

EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
  echo "  Error: Failed to run amd-smi in Docker container (exit code: $EXIT_CODE)" >&2
  echo "  Output: $JSON_OUTPUT" >&2
  exit 1
fi

if [ -z "$JSON_OUTPUT" ]; then
  echo "  Error: No output from amd-smi" >&2
  exit 1
fi

# Parse JSON using Python
PARSED=$(python3 <<EOF
import json
import sys

try:
    data = json.loads('''$JSON_OUTPUT''')

    # Count GPUs
    if 'gpu_data' not in data or not isinstance(data['gpu_data'], list):
        print('error=Invalid JSON structure: missing gpu_data array', file=sys.stderr)
        sys.exit(1)

    gpu_count = len(data['gpu_data'])

    if gpu_count == 0:
        print('error=No GPUs detected', file=sys.stderr)
        sys.exit(1)

    # Get first GPU info (assume all GPUs are same model)
    first_gpu = data['gpu_data'][0]

    # Extract architecture from target_graphics_version
    if 'asic' not in first_gpu or 'target_graphics_version' not in first_gpu['asic']:
        print('error=Missing asic.target_graphics_version field', file=sys.stderr)
        sys.exit(1)

    arch = first_gpu['asic']['target_graphics_version']

    # Extract PCIe vendor:device ID
    vendor_id = first_gpu['asic'].get('vendor_id', '1002')
    device_id = first_gpu['asic'].get('device_id', 'unknown')

    # Clean up hex format (remove 0x prefix if present)
    vendor_id = str(vendor_id).replace('0x', '').lower()
    device_id = str(device_id).replace('0x', '').lower()

    pciid = f"{vendor_id}:{device_id}"

    # Output key-value pairs for shell parsing
    print(f"arch={arch}")
    print(f"pciid={pciid}")
    print(f"count={gpu_count}")

except json.JSONDecodeError as e:
    print(f'error=Invalid JSON: {str(e)}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'error={str(e)}', file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
  echo "  Error: $PARSED" >&2
  exit 1
fi

# Parse the output into shell variables
eval "$PARSED"

echo "  Architecture: $arch"
echo "  PCIe ID: $pciid"
echo "  GPU Count: $count"

# Store in workers.db config table
# Use double colon as separator: <hostname>::gpu::arch
DB_PATH="$WORKDIR/workers.db"

sqlite3 "$DB_PATH" <<SQL
INSERT OR REPLACE INTO config (key, value) VALUES ('${HOSTNAME}::gpu::arch', '$arch');
INSERT OR REPLACE INTO config (key, value) VALUES ('${HOSTNAME}::gpu::pciid', '$pciid');
INSERT OR REPLACE INTO config (key, value) VALUES ('${HOSTNAME}::gpu::number', '$count');
SQL

echo "  ✓ GPU info stored in workers.db"
