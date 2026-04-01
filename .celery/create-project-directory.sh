#!/usr/bin/env
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

if [ ! -t 0 ]; then
  echo "This script requires TTY inputs for interactive operations" >&2
  exit 1
fi

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: create-project-directory.sh <dir>' >&2
  exit 1
fi
dir="$1"

mkdir "$dir"

if [ ! -d "$dir" ]; then
  echo 'Cannot create directory $dir' >&2
  exit 1
fi
rcfile="$dir/config.rc"

echo "Enter secret for services."
echo "This secret will be the common passwords for all services like rabbitmq/postgresql."
echo "random passwords will be used if empty"
read -p "Secret: " secret
if [[ -z "$secret" ]]; then
  secret=$(head -c 16 /dev/urandom | xxd -p)
  echo "Using secret $secret from /dev/urandom"
fi
read -p "Docker Image to Run Worker: " image
read -e -p "Docker Volume Name for PostgreSQL Database: " -i "aotriton_pgdata"  pgvolume
while true; do
	read -r -p "Docker Container Suffix (only allows a-zA-Z0-9_-): " suffix
  # Check if the input contains ONLY a-z, A-Z, and 0-9
  if [[ "$suffix" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Valid input received: $suffix"
    break # Exit the loop if validation passes
  else
    echo "Invalid input. Only a-zA-Z0-9_- characters are allowed."
  fi
done

cat << EOF > "$rcfile"
RABBITMQ_DEFAULT_USER=aotriton
RABBITMQ_DEFAULT_PASS=$secret
RABBITMQ_NODE_PORT=5672
POSTGRES_USER=aotriton
POSTGRES_PASSWORD=$secret
POSTGRES_PORT=5432
POSTGRES_DOCKER_IMAGE=postgres:17.6
POSTGRES_DOCKER_VOLUME=$pgvolume
CONTAINER_SUFFIX=$suffix
CELERY_SERVICE_HOST=$(hostname -f)
CELERY_WORKER_IMAGE=$image
EOF
