#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Worker Heartbeat Monitor

Monitors worker log files and updates PostgreSQL worker_heartbeat table
to track worker process liveness.
"""

import argparse
import signal
import sys
import time
import logging
from pathlib import Path

# Add aotriton root to path
sys.path.insert(0, Path(__file__).resolve().parent.parent.parent.parent.as_posix())

from v3python.tune.utils import get_db_connection_params
import psycopg

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Received signal {signum}, shutting down...")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Monitor worker log files and update heartbeat'
    )
    parser.add_argument(
        '--workdir',
        type=str,
        required=True,
        help='Working directory containing config.rc and logs'
    )
    parser.add_argument(
        '--hostname',
        type=str,
        required=True,
        help='Hostname as configured in workers.db'
    )
    parser.add_argument(
        '--arch',
        type=str,
        required=True,
        help='GPU architecture (e.g., gfx942)'
    )
    parser.add_argument(
        '--check_interval',
        type=int,
        default=30,
        help='How often to check log files in seconds (default: 30)'
    )
    return parser.parse_args()


def update_heartbeat(conn, hostname: str, worker_name: str, arch: str):
    """Update heartbeat for a worker in PostgreSQL"""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO worker_heartbeat (node_hostname, worker_name, arch, last_heartbeat)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (node_hostname, worker_name)
            DO UPDATE SET
                last_heartbeat = NOW(),
                arch = EXCLUDED.arch
        """, (hostname, worker_name, arch))
    logger.debug(f"Updated heartbeat: {hostname}/{worker_name}")


def monitor_logs(workdir: Path, hostname: str, arch: str, check_interval: int):
    """Monitor log directory and update heartbeats"""
    log_dir = workdir / 'run' / 'logs'

    if not log_dir.exists():
        logger.error(f"Log directory not found: {log_dir}")
        return

    logger.info(f"Monitoring log directory: {log_dir}")
    logger.info(f"Hostname: {hostname}, Arch: {arch}, Check interval: {check_interval}s")

    # Get database connection
    conn_params = get_db_connection_params(workdir)
    conn = psycopg.connect(**conn_params, autocommit=True)

    # Track file modification times
    file_mtimes = {}

    try:
        while not shutdown_requested:
            # Scan for all .log files
            log_files = list(log_dir.glob('*.log'))

            for log_file in log_files:
                try:
                    # Get current mtime
                    current_mtime = log_file.stat().st_mtime

                    # Check if file was modified
                    if current_mtime > file_mtimes.get(log_file.name, 0):
                        # Extract worker name from filename (remove .log extension)
                        worker_name = log_file.stem

                        # Update heartbeat
                        update_heartbeat(conn, hostname, worker_name, arch)

                        # Update tracked mtime
                        file_mtimes[log_file.name] = current_mtime

                except Exception as e:
                    logger.error(f"Error processing {log_file.name}: {e}")

            # Sleep before next check
            time.sleep(check_interval)

    finally:
        conn.close()
        logger.info("Heartbeat monitor stopped")


def main():
    """Main entry point"""
    args = parse_args()

    # Install signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    workdir = Path(args.workdir).resolve()

    logger.info("Starting worker heartbeat monitor")
    logger.info(f"Workdir: {workdir}")
    logger.info(f"Hostname: {args.hostname}")
    logger.info(f"Architecture: {args.arch}")

    try:
        monitor_logs(workdir, args.hostname, args.arch, args.check_interval)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
