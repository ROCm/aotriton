#!/usr/bin/env bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Run tuning workers as a foreground command-line program.
# Delegates start/stop to worker_service.sh; stays in foreground so the
# terminal (or SLURM job) is held open until shutdown.
#
# Shutdown is triggered by Ctrl+C/SIGTERM, or automatically via --time_limit.
# With --grace_period G and --time_limit T:
#   t = T - G : graceful stop (PG readers → workers drain → stop)
#   t = T     : force-stop any remaining processes
#
# NOTE: GPU selection is NOT read from the workers DB.
# Pass --multi_gpu explicitly, or use wkctl start for automatic GPU selection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SH="$SCRIPT_DIR/worker_service.sh"

usage() {
    cat >&2 <<EOF
Usage: $0 <workdir> <arch> <session_name> [options]

Run tuning workers as a foreground command-line program.
Delegates to worker_service.sh for start/stop; holds the terminal open.

Arguments:
    workdir                  Path to workdir containing config.rc
    arch                     GPU architecture (e.g., gfx942, gfx90a)
    session_name             Unique name for this run; used as subdirectory
                             under <workdir>/run/pids/ and <workdir>/run/logs/
                             to avoid collisions between concurrent runs

Options:
    --multi_gpu <id>...      GPU IDs to use (-1 for all; default: auto-detect)
    --hostname <name>        Node hostname for heartbeat (default: auto-detect)
    --nreader <n>            Number of PG reader workers (default: 4)
    --ncpu <n>               Number of CPU workers (default: 4)
    --time_limit <sec>       Stop automatically after this many seconds
    --grace_period <sec>     Seconds before time_limit to begin graceful stop;
                             anything still running at time_limit is force-killed
                             (requires --time_limit)

NOTE: GPU selection is not read from the workers DB. Pass --multi_gpu
explicitly, or use wkctl start for automatic GPU assignment.

Examples:
    $0 /path/to/workdir gfx942 interactive
    $0 /path/to/workdir gfx942 slurm-job-42 --multi_gpu 0 1 2 3
    $0 /path/to/workdir gfx942 slurm-job-42 --multi_gpu -1 --time_limit 3600
    $0 /path/to/workdir gfx942 run1 --time_limit 7200 --grace_period 300
EOF
    exit 1
}

# --- Argument parsing ---

if [ $# -lt 3 ]; then
    usage
fi

WORKDIR="$1"
ARCH="$2"
SESSION_NAME="$3"
shift 3

# Args forwarded to all worker_service.sh commands (start, stop, force-stop)
SERVICE_ARGS=("--session_name" "$SESSION_NAME")
TIME_LIMIT=""
GRACE_PERIOD=""

while [ $# -gt 0 ]; do
    case "$1" in
        --multi_gpu)
            SERVICE_ARGS+=("--multi_gpu")
            shift
            while [ $# -gt 0 ] && [[ "$1" =~ ^-?[0-9]+$ ]]; do
                SERVICE_ARGS+=("$1")
                shift
            done
            ;;
        --hostname)
            SERVICE_ARGS+=("--hostname" "$2")
            shift 2
            ;;
        --nreader)
            SERVICE_ARGS+=("--nreader" "$2")
            shift 2
            ;;
        --ncpu)
            SERVICE_ARGS+=("--ncpu" "$2")
            shift 2
            ;;
        --time_limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --grace_period)
            GRACE_PERIOD="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ -n "$GRACE_PERIOD" ] && [ -z "$TIME_LIMIT" ]; then
    echo "Error: --grace_period requires --time_limit" >&2
    exit 1
fi

if [ -n "$GRACE_PERIOD" ] && [ "$GRACE_PERIOD" -ge "$TIME_LIMIT" ]; then
    echo "Error: --grace_period must be less than --time_limit" >&2
    exit 1
fi

# --- Start workers via worker_service.sh ---

echo "Starting workers via worker_service.sh (session=$SESSION_NAME)..."
"$SERVICE_SH" start "$WORKDIR" "$ARCH" "${SERVICE_ARGS[@]}"

echo ""
if [ -n "$TIME_LIMIT" ] && [ -n "$GRACE_PERIOD" ]; then
    echo "Running for $((TIME_LIMIT - GRACE_PERIOD))s, then graceful stop over ${GRACE_PERIOD}s. Press Ctrl+C to stop early."
elif [ -n "$TIME_LIMIT" ]; then
    echo "Running for ${TIME_LIMIT}s. Press Ctrl+C to stop early."
else
    echo "Running. Press Ctrl+C to stop."
fi
echo ""

# --- Signal/timer handling ---
#
# All stop logic lives in the main process traps; timer subshells only send
# signals so they never race with on_stop.
#
# INT/TERM → on_stop  : cancel timer, graceful stop, exit
# USR1     → on_grace : graceful stop, sleep GRACE_PERIOD, force-stop, exit
#   Timer with --grace_period : sleep (TIME_LIMIT - GRACE_PERIOD), send USR1
#   Timer without             : sleep TIME_LIMIT, send TERM

TIMER_PID=""

on_stop() {
    echo ""
    echo "Stopping workers..."
    [ -n "$TIMER_PID" ] && kill "$TIMER_PID" 2>/dev/null || true
    "$SERVICE_SH" stop "$WORKDIR" "$ARCH" "${SERVICE_ARGS[@]}"
    exit 0
}

on_grace() {
    echo ""
    echo "Time limit approaching: graceful stop..."
    [ -n "$TIMER_PID" ] && kill "$TIMER_PID" 2>/dev/null || true
    "$SERVICE_SH" stop "$WORKDIR" "$ARCH" --graceful "${SERVICE_ARGS[@]}"
    exit 0
}

trap on_stop INT TERM
trap on_grace USR1

# Start background timer if --time_limit is set
if [ -n "$TIME_LIMIT" ] && [ -n "$GRACE_PERIOD" ]; then
    ( sleep "$((TIME_LIMIT - GRACE_PERIOD))" && kill -USR1 $$ 2>/dev/null || true ) &
    TIMER_PID=$!
elif [ -n "$TIME_LIMIT" ]; then
    ( sleep "$TIME_LIMIT" && kill -TERM $$ 2>/dev/null || true ) &
    TIMER_PID=$!
fi

# Block in the foreground. wait $! is interruptible by signals (unlike a
# foreground sleep), so INT/TERM/USR1 fire the traps immediately.
sleep infinity &
wait $! || true
