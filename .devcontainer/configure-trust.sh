#!/usr/bin/env bash
set -euo pipefail

# Coder agent startup happens after postCreateCommand. Install the optional
# organization CA here so bootstrap Git/pip commands already trust it.
if [[ -z "${CUSTOM_CA_CERT_B64:-}" ]]; then
    exit 0
fi
if [[ "$(id -u)" != "0" ]]; then
    echo "Installing the workspace CA requires the configured root user." >&2
    exit 1
fi

trust_tmp="$(mktemp -d)"
trap 'rm -rf -- "$trust_tmp"' EXIT
printf '%s' "$CUSTOM_CA_CERT_B64" | base64 --decode > "$trust_tmp/bundle.pem"
csplit --quiet --elide-empty-files --prefix="$trust_tmp/cert-" \
    --suffix-format='%02d.crt' "$trust_tmp/bundle.pem" \
    '/-----BEGIN CERTIFICATE-----/' '{*}'
for cert in "$trust_tmp"/cert-*.crt; do
    # Permit a descriptive preamble, but install only actual certificates.
    if grep -q '^-----BEGIN CERTIFICATE-----' "$cert"; then
        openssl x509 -in "$cert" -noout
        install -m 644 "$cert" "/usr/local/share/ca-certificates/aotriton-$(basename "$cert")"
    fi
done
update-ca-certificates
