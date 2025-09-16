#!/usr/bin/env

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

cat << EOF > "$rcfile"
RABBITMQ_DEFAULT_USER=aotriton
RABBITMQ_DEFAULT_PASS=$secret
RABBITMQ_NODE_PORT=5672
POSTGRES_USER=aotriton
POSTGRES_PASSWORD=$secret
POSTGRES_PORT=5432
POSTGRES_DOCKER_IMAGE=postgres:17.6
POSTGRES_DOCKER_VOLUME=$pgvolume
CELERY_SERVICE_HOST=$(hostname -f)
CELERY_WORKER_IMAGE=$image
EOF
