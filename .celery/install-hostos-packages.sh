#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir>

Install required host OS packages on all registered GPU worker nodes.

Arguments:
  <workdir>  Project working directory
EOF
  exit 1
fi

WORKDIR="$1"

# Validate
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

# Define packages to install
RHEL_PACKAGES=(task-spooler)
DEBIAN_PACKAGES=(task-spooler)

# Load unique hostnames into array
mapfile -t HOSTNAMES < <(sqlite3 "$WORKDIR/workers.db" "SELECT DISTINCT hostname FROM workers ORDER BY hostname;")

# Install packages on each worker
for hostname in "${HOSTNAMES[@]}"; do
  echo "Installing packages on $hostname"

  ssh "$hostname" bash -s "${RHEL_PACKAGES[@]}" -- "${DEBIAN_PACKAGES[@]}" <<'EOF'
# Split arguments into RHEL and Debian package arrays
RHEL_PACKAGES=()
DEBIAN_PACKAGES=()
reading_rhel=true

for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    reading_rhel=false
    continue
  fi
  if $reading_rhel; then
    RHEL_PACKAGES+=("$arg")
  else
    DEBIAN_PACKAGES+=("$arg")
  fi
done

if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS_ID="$ID"
  OS_ID_LIKE="$ID_LIKE"
else
  echo "Cannot detect OS type" >&2
  exit 1
fi

# Detect if RHEL-based or Debian-based and install packages
if [[ "$OS_ID" == "rhel" ]] || [[ "$OS_ID" == "centos" ]] || [[ "$OS_ID" == "fedora" ]] || [[ "$OS_ID" == "almalinux" ]] || [[ "$OS_ID" == "rocky" ]] || [[ "$OS_ID_LIKE" == *"rhel"* ]] || [[ "$OS_ID_LIKE" == *"fedora"* ]]; then
  echo "Detected RHEL-based system, installing: ${RHEL_PACKAGES[*]}"
  sudo dnf install -y "${RHEL_PACKAGES[@]}"
elif [[ "$OS_ID" == "debian" ]] || [[ "$OS_ID" == "ubuntu" ]] || [[ "$OS_ID_LIKE" == *"debian"* ]]; then
  echo "Detected Debian-based system, installing: ${DEBIAN_PACKAGES[*]}"
  DEBIAN_FRONTEND=noninteractive sudo apt-get update
  DEBIAN_FRONTEND=noninteractive sudo apt-get install -y "${DEBIAN_PACKAGES[@]}"
else
  echo "Unsupported OS: $OS_ID" >&2
  exit 1
fi
EOF
done

echo "Package installation completed on all workers"
