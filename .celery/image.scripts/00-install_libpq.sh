#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS_ID="$ID"
  OS_ID_LIKE="$ID_LIKE"
else
  echo "Cannot detect OS type" >&2
  exit 1
fi

# Detect if RHEL-based or Debian-based
if [[ "$OS_ID" == "rhel" ]] || [[ "$OS_ID" == "centos" ]] || [[ "$OS_ID" == "fedora" ]] || [[ "$OS_ID" == "almalinux" ]] || [[ "$OS_ID" == "rocky" ]] || [[ "$OS_ID_LIKE" == *"rhel"* ]] || [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
  echo "Detected RHEL-based system, installing libpq with dnf"
  dnf install -y libpq
elif [[ "$OS_ID" == "debian" ]] || [[ "$OS_ID" == "ubuntu" ]] || [[ "$OS_ID_LIKE" == *"debian"* ]]; then
  echo "Detected Debian-based system, installing libpq5 with apt"
  DEBIAN_FRONTEND=noninteractive apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y libpq5
else
  echo "Unsupported OS: $OS_ID" >&2
  exit 1
fi

echo "libpq installation completed"
