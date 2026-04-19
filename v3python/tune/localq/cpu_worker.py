# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
CPU Worker for Unix socket-based local queue.

Pulls tasks from cpu_queue and executes CPU operations (write results, postprocess).
"""

import sys
import os
import logging
import argparse
import signal

from .generic_worker import GenericWorker
from .handlers import WriteHsacoResultHandler, PostprocessHandler
from ..utils import get_db_connection_params, configure_logging_with_flush

configure_logging_with_flush()

logger = logging.getLogger(__name__)


def main():
    """CPU worker main entry point"""
    parser = argparse.ArgumentParser(description='CPU worker for local queue')
    parser.add_argument('--worker_id', type=str, default='cpu-0',
                       help='Worker identifier')
    parser.add_argument('--workdir', type=str, required=True,
                       help='Path to workdir containing config.rc')
    parser.add_argument('--broker_socket', type=str,
                       default=os.environ.get('AOTRITON_TUNER_BROKER_SOCKET', '/tmp/aotriton-broker.sock'),
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    # Get database connection parameters
    from pathlib import Path
    conn_params = get_db_connection_params(Path(args.workdir))

    # Create handlers for CPU tasks
    handlers = [
        WriteHsacoResultHandler(conn_params),
        PostprocessHandler(conn_params),
    ]

    # Create and run worker
    worker = GenericWorker(
        worker_id=args.worker_id,
        queue_name='cpu_queue',
        handlers=handlers,
        broker_socket=args.broker_socket
    )

    logger.info(f"Starting CPU worker {args.worker_id}")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        worker.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Set wakeup fd to interrupt blocking I/O on signals
    signal.set_wakeup_fd(worker.wakeup_write_fd)

    try:
        worker.run()
        logger.info(f"CPU worker {args.worker_id} run() returned, main() exiting (PID={os.getpid()})")
    except KeyboardInterrupt:
        logger.info(f"CPU worker {args.worker_id} interrupted (PID={os.getpid()})")
        worker.shutdown()
    except Exception as e:
        logger.error(f"CPU worker {args.worker_id} failed: {e} (PID={os.getpid()})", exc_info=True)
        worker.shutdown()
        sys.exit(1)

    logger.info(f"CPU worker {args.worker_id} main() complete (PID={os.getpid()})")


if __name__ == '__main__':
    main()
