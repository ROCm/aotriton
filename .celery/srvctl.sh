#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Control server services (RabbitMQ and PostgreSQL)

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 2 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <action>

Control server services (RabbitMQ and PostgreSQL).

Arguments:
  <workdir>  Project working directory
  <action>   start|stop|restart

This script will:
  - For start: Launch RabbitMQ and PostgreSQL containers
  - For stop: Stop and remove the containers
  - For restart: Stop and start the containers
EOF
  exit 1
fi

WORKDIR="$1"
ACTION="$2"

# Validate action
if [[ "$ACTION" != "start" && "$ACTION" != "stop" && "$ACTION" != "restart" ]]; then
  echo "Error: Invalid action '$ACTION'. Must be start|stop|restart" >&2
  exit 1
fi

# Validate workdir
if [ ! -d "$WORKDIR" ]; then
  echo "Error: Working directory '$WORKDIR' does not exist" >&2
  exit 1
fi

CONFIG_RC="$WORKDIR/config.rc"
if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# Source config
. "$CONFIG_RC"

# Validate required config variables
if [ -z "$RABBITMQ_DEFAULT_USER" ] || [ -z "$RABBITMQ_DEFAULT_PASS" ]; then
  echo "Error: RABBITMQ_DEFAULT_USER or RABBITMQ_DEFAULT_PASS not set in config.rc" >&2
  exit 1
fi

if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DOCKER_IMAGE" ] || [ -z "$POSTGRES_DOCKER_VOLUME" ]; then
  echo "Error: PostgreSQL configuration incomplete in config.rc" >&2
  exit 1
fi

if [ -z "$CONTAINER_SUFFIX" ]; then
  echo "Error: CONTAINER_SUFFIX not set in config.rc" >&2
  exit 1
fi

# Container names
RABBITMQ_CONTAINER="aotriton_rabbitmq.${CONTAINER_SUFFIX}"
POSTGRES_CONTAINER="aotriton_pgsql.${CONTAINER_SUFFIX}"

# PID file
mkdir -p "$WORKDIR/run"
PIDF="$WORKDIR/run/container.pids"

start_services() {
  echo "Starting server services..."

  # Start RabbitMQ
  echo "Starting RabbitMQ..."
  RABBITMQ_ID=$(docker run --ipc=host \
    --network=host \
    -d \
    --rm \
    -e RABBITMQ_DEFAULT_USER="${RABBITMQ_DEFAULT_USER}" \
    -e RABBITMQ_DEFAULT_PASS="${RABBITMQ_DEFAULT_PASS}" \
    --name "${RABBITMQ_CONTAINER}" \
    rabbitmq:4-management)

  if [ -z "$RABBITMQ_ID" ]; then
    echo "Error: Failed to start RabbitMQ" >&2
    exit 1
  fi
  echo "$RABBITMQ_ID" >> "$PIDF"
  echo "Started RabbitMQ: $RABBITMQ_ID"

  # Start PostgreSQL
  echo "Starting PostgreSQL..."
  POSTGRES_ID=$(docker run --ipc=host \
    --network=host \
    -d \
    --rm \
    -e POSTGRES_USER="${POSTGRES_USER}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
    -v "${POSTGRES_DOCKER_VOLUME}:/var/lib/postgresql/data" \
    --name "${POSTGRES_CONTAINER}" \
    "${POSTGRES_DOCKER_IMAGE}")

  if [ -z "$POSTGRES_ID" ]; then
    echo "Error: Failed to start PostgreSQL" >&2
    echo "Cleaning up RabbitMQ..."
    docker stop "$RABBITMQ_ID"
    exit 1
  fi
  echo "$POSTGRES_ID" >> "$PIDF"
  echo "Started PostgreSQL: $POSTGRES_ID"

  echo "Server services started successfully"
}

stop_services() {
  echo "Stopping server services..."

  if [ -f "$PIDF" ]; then
    CONTAINER_IDS=$(cat "$PIDF")
    if [ -n "$CONTAINER_IDS" ]; then
      echo "Stopping containers: $CONTAINER_IDS"
      docker stop $CONTAINER_IDS
      rm -f "$PIDF"
      echo "Server services stopped and containers removed"
    else
      echo "PID file is empty"
      rm -f "$PIDF"
    fi
  else
    # Try to stop by container name
    echo "No PID file found. Attempting to stop by container name..."
    docker stop "${RABBITMQ_CONTAINER}" "${POSTGRES_CONTAINER}" 2>/dev/null || true
    docker rm "${RABBITMQ_CONTAINER}" "${POSTGRES_CONTAINER}" 2>/dev/null || true
    echo "Server services stopped (if they were running)"
  fi
}

# Execute action
case "$ACTION" in
  start)
    if [ -f "$PIDF" ]; then
      echo "Error: Services already running or stale PID file exists. Run stop first." >&2
      exit 1
    fi
    start_services
    ;;

  stop)
    stop_services
    ;;

  restart)
    stop_services
    sleep 2
    start_services
    ;;
esac
