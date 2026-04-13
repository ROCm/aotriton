#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate all certificates for mTLS
set -e

WORKDIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$WORKDIR" ]; then
  echo "Usage: $0 <workdir>" >&2
  exit 1
fi

# Load config to get CELERY_SERVICE_HOST
CONFIG_RC="$WORKDIR/config.rc"
if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

. "$CONFIG_RC"

if [ -z "$CELERY_SERVICE_HOST" ]; then
  echo "Error: CELERY_SERVICE_HOST not set in config.rc" >&2
  exit 1
fi

# Create secrets directory
SECRETS_DIR="$WORKDIR/secrets"
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

echo "Generating certificates for $CELERY_SERVICE_HOST..."
echo ""

# Step 1: Generate CA
"$SCRIPT_DIR/01-generate-ca.sh" "$WORKDIR"

# Step 2: Generate Server Certificate
"$SCRIPT_DIR/02-generate-server.sh" "$WORKDIR" "$CELERY_SERVICE_HOST"

# Step 3: Generate Client Certificate
"$SCRIPT_DIR/03-generate-client.sh" "$WORKDIR"

# Step 4: Generate PKCS12 bundle
"$SCRIPT_DIR/04-generate-p12.sh" "$WORKDIR"

echo ""
echo "All certificates generated in $SECRETS_DIR"
echo ""
echo "To use the dashboard:"
echo "1. Import $SECRETS_DIR/client.p12 into your browser"
echo "   - Chrome: Settings → Privacy → Manage certificates → Import"
echo "   - Firefox: Preferences → Privacy → Certificates → View Certificates → Import"
echo "2. Access https://$CELERY_SERVICE_HOST:8888 (or configured port)"
echo "3. Select the 'admin' certificate when prompted"
