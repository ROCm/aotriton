#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate PKCS12 bundle for browser import
set -e

WORKDIR="$1"
SECRETS_DIR="$WORKDIR/secrets"

CLIENT_P12="$SECRETS_DIR/client.p12"
CLIENT_CRT="$SECRETS_DIR/client.crt"
CLIENT_KEY="$SECRETS_DIR/client.key"
CA_CRT="$SECRETS_DIR/ca.crt"

if [ -f "$CLIENT_P12" ]; then
  echo "[SKIP] PKCS12 bundle already exists: $CLIENT_P12"
  exit 0
fi

if [ ! -f "$CLIENT_CRT" ] || [ ! -f "$CLIENT_KEY" ] || [ ! -f "$CA_CRT" ]; then
  echo "Error: Client certificate not found. Run 03-generate-client.sh first" >&2
  exit 1
fi

echo "[4/4] Generating PKCS12 bundle (no password)..."
openssl pkcs12 -export -out "$CLIENT_P12" \
  -inkey "$CLIENT_KEY" \
  -in "$CLIENT_CRT" \
  -certfile "$CA_CRT" \
  -passout pass:

chmod 600 "$CLIENT_P12"
echo "       ✓ PKCS12 bundle generated"
