#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate Server Certificate with Ed25519
set -e

WORKDIR="$1"
HOSTNAME="$2"
SECRETS_DIR="$WORKDIR/secrets"

SERVER_CRT="$SECRETS_DIR/server.crt"
SERVER_KEY="$SECRETS_DIR/server.key"
CA_CRT="$SECRETS_DIR/ca.crt"
CA_KEY="$SECRETS_DIR/ca.key"

if [ -f "$SERVER_CRT" ] && [ -f "$SERVER_KEY" ]; then
  echo "[SKIP] Server certificate already exists: $SERVER_CRT"
  exit 0
fi

if [ ! -f "$CA_CRT" ] || [ ! -f "$CA_KEY" ]; then
  echo "Error: CA certificate not found. Run 01-generate-ca.sh first" >&2
  exit 1
fi

echo "[2/4] Generating server certificate for $HOSTNAME with RSA 2048 (valid 2 years)..."
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$SERVER_KEY"
openssl req -new -key "$SERVER_KEY" \
  -out "$SECRETS_DIR/server.csr" \
  -subj "/CN=$HOSTNAME/O=AOTriton-Tuning"

openssl x509 -req -days 730 -in "$SECRETS_DIR/server.csr" \
  -CA "$CA_CRT" -CAkey "$CA_KEY" \
  -set_serial "0x$(openssl rand -hex 16)" -out "$SERVER_CRT" \
  -extensions v3_req -extfile <(cat <<EOF
[v3_req]
subjectAltName = DNS:$HOSTNAME,DNS:localhost,IP:127.0.0.1
EOF
)

rm "$SECRETS_DIR/server.csr"
chmod 600 "$SERVER_KEY" "$SERVER_CRT"
echo "       ✓ Server certificate generated (RSA 2048)"
