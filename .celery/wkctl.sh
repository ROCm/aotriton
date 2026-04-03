#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Control all remote GPU workers

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 2 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <action>

Control all remote GPU workers.

Arguments:
  <workdir>  Project working directory
  <action>   start|stop|restart

This script will:
  - SSH to each registered GPU worker
  - For start: Launch docker container and record container ID
  - For stop: Stop worker service, then stop/remove container
  - For restart: Execute restart inside existing container

Prerequisites:
  - Working directory deployed via deploy-workdir.sh
  - Worker images built via build-worker-image.sh
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
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/workers.db" ]; then
  echo "Error: Invalid workdir or workers.db not found" >&2
  exit 1
fi

CONFIG_RC="$WORKDIR/config.rc"
if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# Source config
. "$CONFIG_RC"

if [ -z "$CELERY_WORKER_IMAGE" ]; then
  echo "Error: CELERY_WORKER_IMAGE not set in config.rc" >&2
  exit 1
fi

# Get default working directory for workers
REMOTE_WORKDIR=$(sqlite3 "$WORKDIR/workers.db" "SELECT value FROM config WHERE key = 'default_workdir';" 2>/dev/null)
if [ -z "$REMOTE_WORKDIR" ]; then
  echo "Error: Default working directory not set. Use manage-workers.py set-default-workdir" >&2
  exit 1
fi

# Control worker on each GPU node
control_worker() {
  local hostname="$1"
  local arch="$2"
  local workdir_override="$3"

  # Determine remote workdir for this worker
  if [ -n "$workdir_override" ]; then
    local WORKER_WORKDIR="$workdir_override"
  else
    local WORKER_WORKDIR="$REMOTE_WORKDIR"
  fi

  echo "[$hostname] ${ACTION}ing worker (arch: $arch, workdir: $WORKER_WORKDIR)"

  if [ "$ACTION" = "start" ]; then
    # Start: Launch container and record container ID
    ssh "$hostname" bash -s "$WORKER_WORKDIR" "$arch" "$CELERY_WORKER_IMAGE" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
CELERY_WORKER_IMAGE="$3"
RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

mkdir -p "$WORKER_WORKDIR/run"

if [ -f "$RUNFILE" ]; then
  echo "Worker already running or stale run file exists. Run stop first." >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(docker run -d \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -e PYTHONPATH=/wkdir/build/$ARCH/ \
  -v "$WORKER_WORKDIR:/wkdir" \
  "$CELERY_WORKER_IMAGE" \
  bash /wkdir/aotriton.src/.celery/worker-service.sh start /wkdir)

if [ -z "$WORKER_CONTAINER_ID" ]; then
  echo "Failed to start container" >&2
  exit 1
fi

echo "$WORKER_CONTAINER_ID" > "$RUNFILE"
echo "Started container: $WORKER_CONTAINER_ID"
EOF

  elif [ "$ACTION" = "stop" ]; then
    # Stop: Stop service, then stop/remove container
    ssh "$hostname" bash -s "$WORKER_WORKDIR" <<'EOF'
WORKER_WORKDIR="$1"
RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

if [ ! -f "$RUNFILE" ]; then
  echo "Worker not running or run file missing" >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(cat "$RUNFILE")

echo "Stopping worker service in container: $WORKER_CONTAINER_ID"
docker exec "$WORKER_CONTAINER_ID" bash /wkdir/aotriton.src/.celery/worker-service.sh stop /wkdir

echo "Stopping and removing container: $WORKER_CONTAINER_ID"
docker stop "$WORKER_CONTAINER_ID"
docker rm "$WORKER_CONTAINER_ID"

rm "$RUNFILE"
echo "Worker stopped and container removed"
EOF

  elif [ "$ACTION" = "restart" ]; then
    # Restart: Execute restart inside existing container
    ssh "$hostname" bash -s "$WORKER_WORKDIR" <<'EOF'
WORKER_WORKDIR="$1"
RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

if [ ! -f "$RUNFILE" ]; then
  echo "Worker not running or run file missing" >&2
  exit 1
fi

WORKER_CONTAINER_ID=$(cat "$RUNFILE")

echo "Restarting worker service in container: $WORKER_CONTAINER_ID"
docker exec "$WORKER_CONTAINER_ID" bash /wkdir/aotriton.src/.celery/worker-service.sh restart /wkdir
echo "Worker restarted"
EOF

  fi

  if [ $? -eq 0 ]; then
    echo "[$hostname] Worker ${ACTION}ed successfully"
  else
    echo "[$hostname] Failed to ${ACTION} worker" >&2
  fi
}

# Process each worker
sqlite3 "$WORKDIR/workers.db" "SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers ORDER BY hostname;" | while IFS='|' read -r hostname arch workdir_override; do
  control_worker "$hostname" "$arch" "$workdir_override"
done

echo "Worker control completed: $ACTION"
