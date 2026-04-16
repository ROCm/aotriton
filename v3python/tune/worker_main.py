#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuner v3.5 Worker Process

Runs a single worker that fetches tasks from PostgreSQL queue and executes
them using Ray framework. Supports daemonization for production deployment.
"""

import sys
import os
import argparse
import logging
import signal
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .pq import Worker
from .pq.queue import Task


def setup_logging(log_file: Path = None, log_level: str = 'INFO'):
    """
    Setup logging configuration.

    Args:
        log_file: Optional log file path
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    handlers = []

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_db_connection_params(workdir: Path) -> Dict[str, Any]:
    """
    Get PostgreSQL connection parameters from workdir config.

    Args:
        workdir: Path to workdir containing config.rc

    Returns:
        Connection parameters dictionary
    """
    import subprocess

    # Source config.rc and extract environment variables
    config_rc = workdir / 'config.rc'
    if not config_rc.exists():
        raise FileNotFoundError(f"Config file not found: {config_rc}")

    # Source the config file and print environment variables
    result = subprocess.run(
        f'. {config_rc} && env',
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash'
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to source config.rc: {result.stderr}")

    # Parse environment variables
    env = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            env[key] = value

    return {
        'host': env.get('CELERY_SERVICE_HOST', 'localhost'),
        'port': int(env.get('POSTGRES_PORT', 5432)),
        'user': env.get('POSTGRES_USER', 'aotriton'),
        'password': env.get('POSTGRES_PASSWORD')
    }


def create_task_executor(workdir: Path, arch: str):
    """
    Create task executor function that uses Ray framework.

    Args:
        workdir: Path to workdir
        arch: GPU architecture

    Returns:
        Executor function
    """
    from .localq import init_ray, TuningOrchestrator

    # Initialize Ray once
    init_ray()

    # Create orchestrator instance (reused for all tasks)
    orchestrator = TuningOrchestrator()

    def executor(task: Task) -> Any:
        """
        Execute tuning task using Ray framework.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Executing task {task.id}: module={task.module}, arch={task.arch}")
        logger.debug(f"Task config: {task.task_config}")

        try:
            # Execute full tuning DAG via orchestrator
            result = orchestrator.execute_tuning_dag(str(task.id), task.task_config)
            logger.info(f"Task {task.id} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}", exc_info=True)
            raise

    return executor


def daemonize(pidfile: Path, log_file: Path):
    """
    Daemonize the current process.

    Args:
        pidfile: Path to PID file
        log_file: Path to log file
    """
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"Fork #1 failed: {e}\n")
        sys.exit(1)

    # Decouple from parent environment
    # Keep CWD (aotriton root) instead of chdir('/') so Python can import v3python
    # Safe in container environment where workdir is always mounted
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"Fork #2 failed: {e}\n")
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Redirect stdin to /dev/null
    with open('/dev/null', 'r') as f:
        os.dup2(f.fileno(), sys.stdin.fileno())

    # Redirect stdout/stderr to log file
    with open(log_file, 'a+') as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

    # Write pidfile
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    with open(pidfile, 'w') as f:
        f.write(str(os.getpid()))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Tuner v3.5 Worker Process')

    parser.add_argument('workdir', type=Path,
                        help='Path to workdir containing config.rc')
    parser.add_argument('arch', type=str,
                        help='GPU architecture (e.g., gfx942, gfx90a)')
    parser.add_argument('--worker_id', type=int, default=0,
                        help='Worker ID number (for multiple workers on same node)')
    parser.add_argument('--daemonize', action='store_true',
                        help='Run as daemon process')
    parser.add_argument('--pidfile', type=Path,
                        help='PID file path (required with --daemonize)')
    parser.add_argument('--logfile', type=Path,
                        help='Log file path')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of tasks to fetch per poll')
    parser.add_argument('--poll_interval', type=float, default=1.0,
                        help='Initial poll interval in seconds')
    parser.add_argument('--max_poll_interval', type=float, default=30.0,
                        help='Maximum poll interval (backoff cap)')

    args = parser.parse_args()

    # Validate arguments
    if args.daemonize and not args.pidfile:
        parser.error('--pidfile is required when --daemonize is used')

    # Setup logging before daemonization
    if not args.daemonize:
        setup_logging(args.logfile, args.log_level)

    # Daemonize if requested
    if args.daemonize:
        log_file = args.logfile or args.workdir / 'logs' / f'worker-{args.arch}-{args.worker_id}.log'
        daemonize(args.pidfile, log_file)
        # Re-setup logging after daemonization
        setup_logging(log_file, args.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting Tuner v3.5 worker: arch={args.arch}, worker_id={args.worker_id}, pid={os.getpid()}")

    try:
        # Get database connection parameters
        conn_params = get_db_connection_params(args.workdir)
        logger.info(f"Connected to PostgreSQL: {conn_params['host']}:{conn_params['port']}")

        # Create task executor
        executor = create_task_executor(args.workdir, args.arch)

        # Create and start worker
        worker = Worker(
            conn_params=conn_params,
            arch=args.arch,
            executor=executor,
            batch_size=args.batch_size,
            poll_interval=args.poll_interval,
            max_poll_interval=args.max_poll_interval
        )

        logger.info("Worker initialized, starting main loop")
        worker.start()

    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup pidfile
        if args.daemonize and args.pidfile and args.pidfile.exists():
            args.pidfile.unlink()
        logger.info("Worker stopped")


if __name__ == '__main__':
    main()
