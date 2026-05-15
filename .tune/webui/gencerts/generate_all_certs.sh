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

# Check if running in interactive mode
if [ -t 0 ]; then
  # Interactive mode: loop until user provides valid workdir or confirms generation
  while true; do
    echo "Do you have an existing workdir with certificates to reuse?"
    echo -n "Enter path (or press Enter to generate new): "
    read -r SOURCE_WORKDIR

    if [ -n "$SOURCE_WORKDIR" ]; then
      SOURCE_SECRETS="$SOURCE_WORKDIR/secrets"

      # Check if source has complete certificate set (including client.p12 for browsers)
      if [ -f "$SOURCE_SECRETS/ca.crt" ] && \
         [ -f "$SOURCE_SECRETS/ca.key" ] && \
         [ -f "$SOURCE_SECRETS/server.crt" ] && \
         [ -f "$SOURCE_SECRETS/server.key" ] && \
         [ -f "$SOURCE_SECRETS/client.crt" ] && \
         [ -f "$SOURCE_SECRETS/client.key" ] && \
         [ -f "$SOURCE_SECRETS/client.p12" ]; then

        echo "Found complete certificate set in $SOURCE_WORKDIR"
        echo "Copying all certificates..."

        cp "$SOURCE_SECRETS/ca.crt" "$SECRETS_DIR/"
        cp "$SOURCE_SECRETS/ca.key" "$SECRETS_DIR/"
        cp "$SOURCE_SECRETS/server.crt" "$SECRETS_DIR/"
        cp "$SOURCE_SECRETS/server.key" "$SECRETS_DIR/"
        cp "$SOURCE_SECRETS/client.crt" "$SECRETS_DIR/"
        cp "$SOURCE_SECRETS/client.key" "$SECRETS_DIR/"
        cp "$SOURCE_SECRETS/client.p12" "$SECRETS_DIR/"

        chmod 600 "$SECRETS_DIR"/*

        echo "✓ All certificates copied successfully"
        echo ""
        echo "Certificates are ready in $SECRETS_DIR"
        echo "Import client.p12 into browsers to access the WebUI"
        exit 0
      else
        echo "Warning: Incomplete certificate set in $SOURCE_WORKDIR"
        echo ""
        # Loop back to ask again
        continue
      fi
    fi

    # No source workdir or user pressed Enter - confirm generation
    echo ""
    echo "WARNING: Generating new CA and certificates will require:"
    echo "  1. Re-distributing client.p12 to all users"
    echo "  2. Re-importing certificates in browsers"
    echo "  3. Potentially overriding previous installations"
    echo ""
    echo -n "Are you sure you want to generate new certificates? (yes/NO): "
    read -r CONFIRM

    if [[ "$CONFIRM" =~ ^[Yy][Ee][Ss]$ ]]; then
      # User confirmed, break out of loop and generate
      break
    else
      # User declined, loop back to ask for existing workdir
      echo ""
      continue
    fi
  done
fi

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
