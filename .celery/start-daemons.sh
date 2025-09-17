#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -ne 1 ]; then
  echo 'Missing arguments. Usage: create-project-directory.sh <dir>' >&2
  exit 1
fi
dir="$1"

if [ ! -d "$dir" ]; then
  echo 'Not a directory: $dir' >&2
  exit 1
fi
rcfile="$dir/config.rc"

. "$rcfile"

mkdir -p "$dir/run"

docker run --ipc=host \
    --network=host \
    -d \
    --rm \
    -e RABBITMQ_DEFAULT_USER=${RABBITMQ_DEFAULT_USER} \
    -e RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS} \
    --name aotriton_rabbitmq \
    rabbitmq:4-management >> "$dir/run/container.pids"

# FIXME: /var/lib/postgresql/18/docker is used for PostgreSQL 18 and higher

docker run --ipc=host \
    --network=host \
    -d \
    --rm \
    -e POSTGRES_USER=${POSTGRES_USER} \
    -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
    -v ${POSTGRES_DOCKER_VOLUME}:/var/lib/postgresql/data \
    --name aotriton_pgsql \
    ${POSTGRES_DOCKER_IMAGE} >> "$dir/run/container.pids"

