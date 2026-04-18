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
from ..utils import get_db_connection_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """CPU worker main entry point"""
    parser = argparse.ArgumentParser(description='CPU worker for local queue')
    parser.add_argument('--worker_id', type=str, default='cpu-0',
                       help='Worker identifier')
    parser.add_argument('--broker_socket', type=str,
                       default=os.environ.get('AOTRITON_TUNER_BROKER_SOCKET', '/tmp/aotriton-broker.sock'),
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    # Get database connection parameters
    conn_params = get_db_connection_params()

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

    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info(f"CPU worker {args.worker_id} interrupted")
        worker.shutdown()
    except Exception as e:
        logger.error(f"CPU worker {args.worker_id} failed: {e}", exc_info=True)
        worker.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
