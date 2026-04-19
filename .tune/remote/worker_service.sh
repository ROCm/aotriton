#!/usr/bin/env bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Tuner v3.5 Worker Service Management Script
# SysV-style init script for managing multiple worker processes

set -euo pipefail

# Get script directory and aotriton root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOTRITON_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to aotriton root so Python can find v3python module
cd "$AOTRITON_ROOT"

BROKER_MODULE="v3python.tune.localq.broker_main"
GPU_WORKER_MODULE="v3python.tune.localq.gpu_worker_socket"
PG_READER_MODULE="v3python.tune.localq.pg_reader_worker"
CPU_WORKER_MODULE="v3python.tune.localq.cpu_worker"

# Load config utilities
. "$TUNE_ROOT/lib/config_load.sh"

usage() {
    cat <<EOF
Usage: $0 <command> <workdir> <arch> [--multi_gpu <ids...>]

Manage Tuner v3.5 local queue processes (SysV-style init script)

Commands:
    start       Start broker and all workers
    stop        Stop all processes
    restart     Restart all processes (stop then start)
    status      Show process status
    force-stop  Force kill all processes (SIGKILL)

Arguments:
    command       Command to execute
    workdir       Path to workdir containing config.rc
    arch          GPU architecture (e.g., gfx942, gfx90a)

Options:
    --multi_gpu <id> [<id>...]    GPU IDs to use (space-separated)
                                  Use -1 for all GPUs (default: all auto-detected)

Components started:
    - 1 broker (message router)
    - GPU workers (auto-detected or selected via --multi_gpu)
    - 4 PG reader workers (fetch tasks from PostgreSQL)
    - 4 CPU workers (postprocess, write results)

Examples:
    # Start all GPU workers (auto-detected)
    $0 start /path/to/workdir gfx942

    # Start specific GPU workers
    $0 start /path/to/workdir gfx942 --multi_gpu 0 1
    $0 start /path/to/workdir gfx942 --multi_gpu 1 2 3 4 5 6

    # Start all GPUs explicitly
    $0 start /path/to/workdir gfx942 --multi_gpu -1

    # Stop always stops all processes regardless of which were started
    $0 stop /path/to/workdir gfx942

    # Restart with selected GPUs
    $0 restart /path/to/workdir gfx942 --multi_gpu 2 3

PID files location: <workdir>/run/pids/
Log files location: <workdir>/run/logs/
EOF
    exit 1
}

# Extract positional arguments first
COMMAND="${1:-}"
WORKDIR="${2:-}"
ARCH="${3:-}"
shift 3

# Parse optional --multi_gpu (must come after positionals)
GPU_IDS=()
USE_ALL_GPUS=false

while [ $# -gt 0 ]; do
    case "$1" in
        --multi_gpu)
            shift
            # Collect all following numeric arguments (or -1)
            while [ $# -gt 0 ] && [[ "$1" =~ ^-?[0-9]+$ ]]; do
                if [ "$1" = "-1" ]; then
                    USE_ALL_GPUS=true
                    shift
                    break
                else
                    GPU_IDS+=("$1")
                    shift
                fi
            done
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ -z "$COMMAND" ] || [ -z "$WORKDIR" ] || [ -z "$ARCH" ]; then
    usage
fi

# Validate workdir
if [ ! -d "$WORKDIR" ]; then
    echo "Error: Workdir does not exist: $WORKDIR"
    exit 1
fi

load_config_container "$WORKDIR" || exit 1

# Determine final GPU IDs list
if [ "$USE_ALL_GPUS" = true ] || [ ${#GPU_IDS[@]} -eq 0 ]; then
    # Auto-detect all GPUs
    NUM_GPUS=$(rocm_agent_enumerator | grep -v gfx000 | wc -l)
    GPU_IDS=($(seq 0 $((NUM_GPUS - 1))))
fi

echo "Using GPU IDs: ${GPU_IDS[*]}"

# Validate localq modules exist
if [ ! -f "v3python/tune/localq/broker_main.py" ]; then
    echo "Error: localq modules not found in v3python/tune/localq/"
    exit 1
fi

# Directories
PID_DIR="$WORKDIR/run/pids"
LOG_DIR="$WORKDIR/run/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

# Set broker socket path (used by all localq components)
# All entry points (broker_main, gpu_worker_socket, pg_reader_worker, cpu_worker)
# default --broker_socket to this env var, so we don't need to pass it explicitly
export AOTRITON_TUNER_BROKER_SOCKET="$WORKDIR/run/broker.sock"

# Helper functions

get_pidfile() {
    echo "$PID_DIR/$1.pid"
}

get_logfile() {
    echo "$LOG_DIR/$1.log"
}

daemonize() {
    local logfile="$1"
    local pidfile="$2"
    shift 2
    local cmd=("$@")

    # Fork and run in background, redirect I/O
    # Do NOT chdir to / - stay in current directory
    (
        # Close stdin
        exec 0</dev/null

        # Redirect stdout and stderr to logfile
        exec 1>>"$logfile"
        exec 2>&1

        # Run command
        "${cmd[@]}" &
        local child_pid=$!

        # Save PID
        echo $child_pid > "$pidfile"

        # Wait for child to exit and reap it (prevents zombies)
        wait $child_pid
        exit_code=$?

        # Clean up PID file when process exits
        rm -f "$pidfile"

        exit $exit_code
    ) &

    # Wait for PID file to be written
    local timeout=5
    local elapsed=0
    while [ ! -f "$pidfile" ] && [ $elapsed -lt $timeout ]; do
        sleep 0.1
        elapsed=$((elapsed + 1))
    done
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

start_single_process() {
    local display_name="$1"
    local name="$2"  # for pidfile/logfile
    shift 2
    local cmd_args=("$@")  # rest are command args

    local pidfile=$(get_pidfile "$name")
    local logfile=$(get_logfile "$name")

    if is_running "$pidfile"; then
        echo "$display_name already running (PID: $(cat "$pidfile"))"
        return 0
    fi

    echo "Starting $display_name..."
    daemonize "$logfile" "$pidfile" "${cmd_args[@]}"

    sleep 0.5

    if is_running "$pidfile"; then
        echo "$display_name started (PID: $(cat "$pidfile"))"
    else
        echo "Failed to start $display_name (check logs: $logfile)"
        return 1
    fi
}

start_process_group() {
    local group_name="$1"
    local name_prefix="$2"
    local module="$3"
    local arg_type="$4"  # "gpu" | "pg" | "cpu"
    shift 4
    local ids=("$@")     # sequence of IDs to start

    local failed=0

    for id in "${ids[@]}"; do
        local name="$name_prefix-$id"
        local display="$group_name $id"
        local module_args=()

        case "$arg_type" in
            gpu)
                module_args=("--gpu_id" "$id")
                ;;
            pg)
                module_args=("--worker_id" "$name" "--arch" "$ARCH" "--workdir" "$WORKDIR")
                ;;
            cpu)
                module_args=("--worker_id" "$name" "--workdir" "$WORKDIR")
                ;;
        esac

        if ! start_single_process "$display" "$name" \
            "$CELERY_WORKER_PYTHON" -m "$module" "${module_args[@]}"; then
            failed=$((failed + 1))
        fi
    done

    return $failed
}

stop_process_group() {
    local group_name="$1"
    local name_prefix="$2"
    local count="${3:-0}"  # number of processes (default 0 for single process like broker)

    # Build pidfile list
    local pidfiles=()
    if [ "$count" -eq 0 ]; then
        # Single process (e.g., broker)
        pidfiles+=("$(get_pidfile "$name_prefix")")
    else
        # Multiple processes with ID suffix (0 to count-1)
        for id in $(seq 0 $((count - 1))); do
            local pidfile="$(get_pidfile "$name_prefix-$id")"
            # Only add if file exists (tolerate non-existing processes)
            if [ -f "$pidfile" ]; then
                pidfiles+=("$pidfile")
            fi
        done
    fi

    # Collect running PIDs and map PID → pidfile
    local pids=()
    declare -A pid_to_file  # Associative array for PID → pidfile mapping

    for pidfile in "${pidfiles[@]}"; do
        if is_running "$pidfile"; then
            local pid
            pid=$(cat "$pidfile")
            pids+=("$pid")
            pid_to_file["$pid"]="$pidfile"
        fi
    done

    if [ ${#pids[@]} -eq 0 ]; then
        echo "No $group_name processes running"
        return 0
    fi

    echo "Stopping ${#pids[@]} $group_name processes (PIDs: ${pids[*]})..."

    # 1. Send SIGTERM to all
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    # 2. Wait up to 30 seconds for graceful shutdown
    local timeout=30
    local elapsed=0
    local remaining=("${pids[@]}")

    while [ $elapsed -lt $timeout ] && [ ${#remaining[@]} -gt 0 ]; do
        local new_remaining=()
        for pid in "${remaining[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_remaining+=("$pid")
            else
                # Process stopped, clean up pidfile
                local pidfile="${pid_to_file[$pid]}"
                rm -f "$pidfile"
            fi
        done
        remaining=("${new_remaining[@]}")

        if [ ${#remaining[@]} -gt 0 ]; then
            sleep 1
            elapsed=$((elapsed + 1))
        fi
    done

    # 3. Force kill any remaining processes
    if [ ${#remaining[@]} -gt 0 ]; then
        echo "$group_name: ${#remaining[@]} processes did not stop gracefully, force killing (PIDs: ${remaining[*]})..."
        for pid in "${remaining[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
            local pidfile="${pid_to_file[$pid]}"
            rm -f "$pidfile"
        done
        echo "$group_name: force stopped ${#remaining[@]} processes"
    else
        echo "$group_name: all processes stopped gracefully"
    fi
}

force_stop_process() {
    local name="$1"
    local pidfile="$2"
    local pid

    if ! is_running "$pidfile"; then
        echo "$name not running"
        return 0
    fi

    pid=$(cat "$pidfile")
    echo "Force stopping $name (PID: $pid)..."

    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$pidfile"
    echo "$name force stopped"
}

process_status() {
    local name="$1"
    local pidfile="$2"
    local pid

    if is_running "$pidfile"; then
        pid=$(cat "$pidfile")
        echo "$name: RUNNING (PID: $pid)"
        return 0
    else
        echo "$name: STOPPED"
        return 1
    fi
}

# Main command handlers

cmd_start() {
    echo "Starting local queue system for $ARCH..."
    echo ""

    local failed=0

    # 1. Start broker
    if ! start_single_process "Broker" "broker" \
        "$CELERY_WORKER_PYTHON" -m "$BROKER_MODULE"; then
        echo "Error: Failed to start broker"
        exit 1
    fi
    echo ""

    # Wait for broker socket to be ready
    sleep 1

    # 2. Start GPU workers
    echo "Starting ${#GPU_IDS[@]} GPU workers for GPUs: ${GPU_IDS[*]}..."
    start_process_group "GPU worker" "gpu-worker" "$GPU_WORKER_MODULE" "gpu" "${GPU_IDS[@]}"
    failed=$?
    echo ""

    # 3. Start PG readers (4 workers)
    echo "Starting 4 PG reader workers for arch=$ARCH..."
    start_process_group "PG reader" "pg-reader-$ARCH" "$PG_READER_MODULE" "pg" 0 1 2 3
    failed=$((failed + $?))
    echo ""

    # 4. Start CPU workers (4 workers)
    echo "Starting 4 CPU workers..."
    start_process_group "CPU worker" "cpu-worker" "$CPU_WORKER_MODULE" "cpu" 0 1 2 3
    failed=$((failed + $?))
    echo ""

    if [ $failed -eq 0 ]; then
        echo "All processes started successfully"
    else
        echo "Warning: $failed processes failed to start"
        exit 1
    fi
}

cmd_stop() {
    echo "Stopping all processes..."
    echo ""

    # Auto-detect NUM_GPUS for stop (may differ from start if GPUs were selected)
    local NUM_GPUS_ALL=$(rocm_agent_enumerator | grep -v gfx000 | wc -l)

    # Stop in order: PG readers → GPU workers → CPU workers → broker

    # 1. Stop PG readers first (halt incoming tasks)
    stop_process_group "PG readers" "pg-reader-$ARCH" 4
    echo ""

    # 2. Stop GPU workers (no more outgoing hsaco results)
    # Try to stop all possible GPU workers (0 to NUM_GPUS_ALL-1)
    stop_process_group "GPU workers" "gpu-worker" $NUM_GPUS_ALL
    echo ""

    # 3. Stop CPU workers (finish processing remaining results)
    stop_process_group "CPU workers" "cpu-worker" 4
    echo ""

    # 4. Stop broker last (stop message router)
    stop_process_group "Broker" "broker"
    echo ""

    # Clean up socket file
    rm -f "$AOTRITON_TUNER_BROKER_SOCKET"

    echo "All processes stopped"
}

force_stop_process_group() {
    local group_name="$1"
    local name_prefix="$2"
    local count="${3:-0}"  # number of processes (default 0 for single process like broker)

    # Build pidfile list
    local pidfiles=()
    if [ "$count" -eq 0 ]; then
        # Single process (e.g., broker)
        pidfiles+=("$(get_pidfile "$name_prefix")")
    else
        # Multiple processes with ID suffix (0 to count-1)
        for id in $(seq 0 $((count - 1))); do
            local pidfile="$(get_pidfile "$name_prefix-$id")"
            # Only add if file exists (tolerate non-existing processes)
            if [ -f "$pidfile" ]; then
                pidfiles+=("$pidfile")
            fi
        done
    fi

    # Collect running PIDs
    local pids=()

    for pidfile in "${pidfiles[@]}"; do
        if is_running "$pidfile"; then
            local pid
            pid=$(cat "$pidfile")
            pids+=("$pid")
        fi
    done

    if [ ${#pids[@]} -eq 0 ]; then
        echo "No $group_name processes running"
        return 0
    fi

    echo "Force stopping ${#pids[@]} $group_name processes (PIDs: ${pids[*]})..."

    # Send SIGKILL to all
    for pid in "${pids[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done

    # Clean up pidfiles
    for pidfile in "${pidfiles[@]}"; do
        rm -f "$pidfile"
    done

    echo "$group_name: force stopped ${#pids[@]} processes"
}

cmd_force_stop() {
    echo "Force stopping all processes..."
    echo ""

    # Auto-detect NUM_GPUS for force-stop
    local NUM_GPUS_ALL=$(rocm_agent_enumerator | grep -v gfx000 | wc -l)

    # Force stop in order: PG readers → GPU workers → CPU workers → broker

    # 1. PG readers
    force_stop_process_group "PG readers" "pg-reader-$ARCH" 4

    # 2. GPU workers
    force_stop_process_group "GPU workers" "gpu-worker" $NUM_GPUS_ALL

    # 3. CPU workers
    force_stop_process_group "CPU workers" "cpu-worker" 4

    # 4. Broker
    force_stop_process_group "Broker" "broker"

    # Clean up socket file
    rm -f "$AOTRITON_TUNER_BROKER_SOCKET"

    echo ""
    echo "All processes force stopped"
}

cmd_restart() {
    cmd_stop
    echo "Waiting 2 seconds before restart..."
    sleep 2
    cmd_start
}

cmd_status() {
    echo "Process status for arch=$ARCH:"
    echo ""

    # Auto-detect NUM_GPUS for status
    local NUM_GPUS_ALL=$(rocm_agent_enumerator | grep -v gfx000 | wc -l)

    local running=0
    local stopped=0

    # 1. Broker
    echo "=== Broker ==="
    if process_status "Broker" "$(get_pidfile "broker")"; then
        running=$((running + 1))
    else
        stopped=$((stopped + 1))
    fi
    echo ""

    # 2. GPU workers
    echo "=== GPU Workers ==="
    for i in $(seq 0 $((NUM_GPUS_ALL - 1))); do
        local pidfile="$(get_pidfile "gpu-worker-$i")"
        if [ -f "$pidfile" ]; then
            if process_status "GPU worker $i" "$pidfile"; then
                running=$((running + 1))
            else
                stopped=$((stopped + 1))
            fi
        fi
    done
    echo ""

    # 3. PG readers
    echo "=== PG Readers ==="
    for i in $(seq 0 3); do
        if process_status "PG reader $i" "$(get_pidfile "pg-reader-$ARCH-$i")"; then
            running=$((running + 1))
        else
            stopped=$((stopped + 1))
        fi
    done
    echo ""

    # 4. CPU workers
    echo "=== CPU Workers ==="
    for i in $(seq 0 3); do
        if process_status "CPU worker $i" "$(get_pidfile "cpu-worker-$i")"; then
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
