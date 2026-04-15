#!/usr/bin/env bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Tuner v3.5 Worker Service Management Script
# SysV-style init script for managing multiple worker processes

set -euo pipefail

# Get script directory and aotriton root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOTRITON_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKER_MAIN="$AOTRITON_ROOT/v3python/tune/worker_main.py"

usage() {
    cat <<EOF
Usage: $0 <command> <workdir> <arch> [num_workers]

Manage Tuner v3.5 worker processes (SysV-style init script)

Commands:
    start       Start workers
    stop        Stop workers
    restart     Restart workers (stop then start)
    status      Show worker status
    force-stop  Force kill workers (SIGKILL)

Arguments:
    workdir       Path to workdir containing config.rc
    arch          GPU architecture (e.g., gfx942, gfx90a)
    num_workers   Number of worker processes (default: 4)

Examples:
    $0 start /path/to/workdir gfx942 8
    $0 stop /path/to/workdir gfx942
    $0 restart /path/to/workdir gfx942 8
    $0 status /path/to/workdir gfx942

PID files location: <workdir>/pids/worker-<arch>-<id>.pid
Log files location: <workdir>/logs/worker-<arch>-<id>.log
EOF
    exit 1
}

# Parse arguments
COMMAND="${1:-}"
WORKDIR="${2:-}"
ARCH="${3:-}"
NUM_WORKERS="${4:-4}"

if [ -z "$COMMAND" ] || [ -z "$WORKDIR" ] || [ -z "$ARCH" ]; then
    usage
fi

# Validate workdir
if [ ! -d "$WORKDIR" ]; then
    echo "Error: Workdir does not exist: $WORKDIR"
    exit 1
fi

if [ ! -f "$WORKDIR/config.rc" ]; then
    echo "Error: config.rc not found in workdir: $WORKDIR"
    exit 1
fi

# Validate worker_main.py exists
if [ ! -f "$WORKER_MAIN" ]; then
    echo "Error: worker_main.py not found: $WORKER_MAIN"
    exit 1
fi

# Directories
PID_DIR="$WORKDIR/pids"
LOG_DIR="$WORKDIR/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

# Helper functions

get_pidfile() {
    local worker_id="$1"
    echo "$PID_DIR/worker-$ARCH-$worker_id.pid"
}

get_logfile() {
    local worker_id="$1"
    echo "$LOG_DIR/worker-$ARCH-$worker_id.log"
}

is_running() {
    local pidfile="$1"

    if [ ! -f "$pidfile" ]; then
        return 1
    fi

    local pid
    pid=$(cat "$pidfile")

    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        # Stale pidfile
        rm -f "$pidfile"
        return 1
    fi
}

start_worker() {
    local worker_id="$1"
    local pidfile
    local logfile

    pidfile=$(get_pidfile "$worker_id")
    logfile=$(get_logfile "$worker_id")

    if is_running "$pidfile"; then
        echo "Worker $ARCH-$worker_id already running (PID: $(cat "$pidfile"))"
        return 0
    fi

    echo "Starting worker $ARCH-$worker_id..."

    python3 "$WORKER_MAIN" \
        "$WORKDIR" \
        "$ARCH" \
        --worker-id "$worker_id" \
        --daemonize \
        --pidfile "$pidfile" \
        --logfile "$logfile" \
        --log-level INFO

    # Wait a moment and verify it started
    sleep 1

    if is_running "$pidfile"; then
        echo "Worker $ARCH-$worker_id started (PID: $(cat "$pidfile"))"
    else
        echo "Failed to start worker $ARCH-$worker_id (check logs: $logfile)"
        return 1
    fi
}

stop_worker() {
    local worker_id="$1"
    local pidfile
    local pid

    pidfile=$(get_pidfile "$worker_id")

    if ! is_running "$pidfile"; then
        echo "Worker $ARCH-$worker_id not running"
        return 0
    fi

    pid=$(cat "$pidfile")
    echo "Stopping worker $ARCH-$worker_id (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid"

    # Wait up to 30 seconds for graceful shutdown
    local timeout=30
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Worker $ARCH-$worker_id stopped"
            rm -f "$pidfile"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # Force kill if still running
    echo "Worker $ARCH-$worker_id did not stop gracefully, force killing..."
    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$pidfile"
    echo "Worker $ARCH-$worker_id force stopped"
}

force_stop_worker() {
    local worker_id="$1"
    local pidfile
    local pid

    pidfile=$(get_pidfile "$worker_id")

    if ! is_running "$pidfile"; then
        echo "Worker $ARCH-$worker_id not running"
        return 0
    fi

    pid=$(cat "$pidfile")
    echo "Force stopping worker $ARCH-$worker_id (PID: $pid)..."

    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$pidfile"
    echo "Worker $ARCH-$worker_id force stopped"
}

worker_status() {
    local worker_id="$1"
    local pidfile
    local pid

    pidfile=$(get_pidfile "$worker_id")

    if is_running "$pidfile"; then
        pid=$(cat "$pidfile")
        echo "Worker $ARCH-$worker_id: RUNNING (PID: $pid)"
        return 0
    else
        echo "Worker $ARCH-$worker_id: STOPPED"
        return 1
    fi
}

# Main command handlers

cmd_start() {
    # Start Ray cluster first (shared by all workers)
    echo "Starting Ray cluster..."
    "$SCRIPT_DIR/rayctl" "$WORKDIR" start || {
        echo "Error: Failed to start Ray cluster"
        exit 1
    }

    echo "Starting $NUM_WORKERS workers for $ARCH..."

    local failed=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        if ! start_worker "$i"; then
            failed=$((failed + 1))
        fi
    done

    if [ $failed -eq 0 ]; then
        echo "All workers started successfully"
    else
        echo "Warning: $failed workers failed to start"
        exit 1
    fi
}

cmd_stop() {
    echo "Stopping all workers for $ARCH..."

    # Find all pidfiles for this arch
    local pidfiles
    pidfiles=$(find "$PID_DIR" -name "worker-$ARCH-*.pid" 2>/dev/null || true)

    if [ -z "$pidfiles" ]; then
        echo "No workers running for $ARCH"
    else
        local count=0
        for pidfile in $pidfiles; do
            local worker_id
            worker_id=$(basename "$pidfile" | sed "s/worker-$ARCH-//;s/\.pid//")
            stop_worker "$worker_id"
            count=$((count + 1))
        done
        echo "Stopped $count workers"
    fi

    # Stop Ray cluster (shared by all workers)
    echo "Stopping Ray cluster..."
    "$SCRIPT_DIR/rayctl" "$WORKDIR" stop || true
}

cmd_force_stop() {
    echo "Force stopping all workers for $ARCH..."

    # Find all pidfiles for this arch
    local pidfiles
    pidfiles=$(find "$PID_DIR" -name "worker-$ARCH-*.pid" 2>/dev/null || true)

    if [ -z "$pidfiles" ]; then
        echo "No workers running for $ARCH"
        return 0
    fi

    local count=0
    for pidfile in $pidfiles; do
        local worker_id
        worker_id=$(basename "$pidfile" | sed "s/worker-$ARCH-//;s/\.pid//")
        force_stop_worker "$worker_id"
        count=$((count + 1))
    done

    echo "Force stopped $count workers"
}

cmd_restart() {
    cmd_stop
    echo "Waiting 2 seconds before restart..."
    sleep 2
    cmd_start
}

cmd_status() {
    echo "Worker status for $ARCH:"

    # Find all pidfiles for this arch
    local pidfiles
    pidfiles=$(find "$PID_DIR" -name "worker-$ARCH-*.pid" 2>/dev/null | sort || true)

    if [ -z "$pidfiles" ]; then
        echo "No workers configured for $ARCH"
        return 0
    fi

    local running=0
    local stopped=0

    for pidfile in $pidfiles; do
        local worker_id
        worker_id=$(basename "$pidfile" | sed "s/worker-$ARCH-//;s/\.pid//")

        if worker_status "$worker_id"; then
            running=$((running + 1))
        else
            stopped=$((stopped + 1))
        fi
    done

    echo ""
    echo "Summary: $running running, $stopped stopped"

    if [ $running -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Execute command
case "$COMMAND" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    status)
        cmd_status
        ;;
    force-stop)
        cmd_force_stop
        ;;
    *)
        echo "Error: Unknown command: $COMMAND"
        usage
        ;;
esac
