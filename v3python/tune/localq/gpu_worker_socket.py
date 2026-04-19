# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GPU Worker for Unix socket-based local queue.

Pulls tasks from gpu_queue and executes GPU operations.
"""

import sys
import os
import logging
import argparse
import signal

from .generic_worker import GenericWorker
from .handlers import TuneKernelHandler, PreprocessHandler, ProbeHandler, TuneHsacoHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """GPU worker main entry point"""
    parser = argparse.ArgumentParser(description='GPU worker for local queue')
    parser.add_argument('--gpu_id', type=int, required=True,
                       help='GPU device ID')
    parser.add_argument('--broker_socket', type=str,
                       default=os.environ.get('AOTRITON_TUNER_BROKER_SOCKET', '/tmp/aotriton-broker.sock'),
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    worker_id = f"gpu-{gpu_id}"

    # Create handlers for GPU tasks
    handlers = [
        TuneKernelHandler(),
        PreprocessHandler(gpu_id),
        ProbeHandler(gpu_id),
        TuneHsacoHandler(gpu_id),
    ]

    # Create and run worker
    worker = GenericWorker(
        worker_id=worker_id,
        queue_name='gpu_queue',
        handlers=handlers,
        broker_socket=args.broker_socket
    )

    logger.info(f"Starting GPU worker {worker_id}")

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
        logger.info(f"GPU worker {worker_id} run() returned, main() exiting (PID={os.getpid()})")
    except KeyboardInterrupt:
        logger.info(f"GPU worker {worker_id} interrupted (PID={os.getpid()})")
        worker.shutdown()
    except Exception as e:
        logger.error(f"GPU worker {worker_id} failed: {e} (PID={os.getpid()})", exc_info=True)
        worker.shutdown()
        sys.exit(1)

    logger.info(f"GPU worker {worker_id} main() complete (PID={os.getpid()})")


if __name__ == '__main__':
    main()
