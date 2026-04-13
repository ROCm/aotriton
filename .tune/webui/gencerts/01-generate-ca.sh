#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate Certificate Authority (CA) with Ed25519
set -e

WORKDIR="$1"
SECRETS_DIR="$WORKDIR/secrets"

CA_CRT="$SECRETS_DIR/ca.crt"
CA_KEY="$SECRETS_DIR/ca.key"

if [ -f "$CA_CRT" ] && [ -f "$CA_KEY" ]; then
  echo "[SKIP] CA certificate already exists: $CA_CRT"
  exit 0
fi

echo "[1/4] Generating CA certificate with Ed25519 (valid 10 years)..."
openssl genpkey -algorithm ED25519 -out "$CA_KEY"
openssl req -new -x509 -days 3650 -key "$CA_KEY" \
  -out "$CA_CRT" \
  -subj "/CN=AOTriton-CA/O=AOTriton-Tuning"

chmod 600 "$CA_KEY" "$CA_CRT"
echo "       ✓ CA certificate generated (Ed25519)"
