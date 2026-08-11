#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate Client Certificate with Ed25519
set -e

WORKDIR="$1"
SECRETS_DIR="$WORKDIR/secrets"

CLIENT_CRT="$SECRETS_DIR/client.crt"
CLIENT_KEY="$SECRETS_DIR/client.key"
CA_CRT="$SECRETS_DIR/ca.crt"
CA_KEY="$SECRETS_DIR/ca.key"

if [ -f "$CLIENT_CRT" ] && [ -f "$CLIENT_KEY" ]; then
  echo "[SKIP] Client certificate already exists: $CLIENT_CRT"
  exit 0
fi

if [ ! -f "$CA_CRT" ] || [ ! -f "$CA_KEY" ]; then
  echo "Error: CA certificate not found. Run 01-generate-ca.sh first" >&2
  exit 1
fi

echo "[3/4] Generating client certificate with RSA 2048 (valid 1 year)..."
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$CLIENT_KEY"
openssl req -new -key "$CLIENT_KEY" \
  -out "$SECRETS_DIR/client.csr" \
  -subj "/CN=admin/O=AOTriton-Tuning"

openssl x509 -req -days 365 -in "$SECRETS_DIR/client.csr" \
  -CA "$CA_CRT" -CAkey "$CA_KEY" \
  -set_serial "0x$(openssl rand -hex 16)" -out "$CLIENT_CRT"

rm "$SECRETS_DIR/client.csr"
chmod 600 "$CLIENT_KEY" "$CLIENT_CRT"
echo "       ✓ Client certificate generated (RSA 2048)"
