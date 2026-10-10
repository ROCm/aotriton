#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate Demo Client Certificate and PKCS12 bundle (CN=demo)
set -e

WORKDIR="$1"
SECRETS_DIR="$WORKDIR/secrets"

DEMO_CRT="$SECRETS_DIR/demo.crt"
DEMO_KEY="$SECRETS_DIR/demo.key"
DEMO_P12="$SECRETS_DIR/demo.p12"
CA_CRT="$SECRETS_DIR/ca.crt"
CA_KEY="$SECRETS_DIR/ca.key"

if [ ! -f "$CA_CRT" ] || [ ! -f "$CA_KEY" ]; then
  echo "Error: CA certificate not found. Run 01-generate-ca.sh first" >&2
  exit 1
fi

if [ -f "$DEMO_CRT" ] && [ -f "$DEMO_KEY" ]; then
  echo "[SKIP] Demo client certificate already exists: $DEMO_CRT"
else
  echo "[5/5] Generating demo client certificate with RSA 2048 (valid 1 year)..."
  openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$DEMO_KEY"
  openssl req -new -key "$DEMO_KEY" \
    -out "$SECRETS_DIR/demo.csr" \
    -subj "/CN=demo/O=AOTriton-Tuning"

  openssl x509 -req -days 365 -in "$SECRETS_DIR/demo.csr" \
    -CA "$CA_CRT" -CAkey "$CA_KEY" \
    -set_serial "0x$(openssl rand -hex 16)" -out "$DEMO_CRT"

  rm "$SECRETS_DIR/demo.csr"
  chmod 600 "$DEMO_KEY" "$DEMO_CRT"
  echo "       Demo client certificate generated (RSA 2048)"
fi

if [ -f "$DEMO_P12" ]; then
  echo "[SKIP] Demo PKCS12 bundle already exists: $DEMO_P12"
  exit 0
fi

echo "       Generating demo PKCS12 bundle (no password)..."
openssl pkcs12 -export -out "$DEMO_P12" \
  -inkey "$DEMO_KEY" \
  -in "$DEMO_CRT" \
  -certfile "$CA_CRT" \
  -name "AOTriton Demo Certificate" \
  -passout pass:""

chmod 600 "$DEMO_P12"
echo "       Demo PKCS12 bundle generated"
