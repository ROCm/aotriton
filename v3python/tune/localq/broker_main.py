# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Broker main entry point.
"""

import sys
import os
import logging
import argparse
import signal

from .broker import LocalBroker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Broker main entry point"""
    parser = argparse.ArgumentParser(description='Local queue broker')
    parser.add_argument('--socket_path', type=str,
                       default=os.environ.get('AOTRITON_TUNER_BROKER_SOCKET', '/tmp/aotriton-broker.sock'),
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    broker = LocalBroker(socket_path=args.socket_path)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        broker.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"Starting LocalBroker (PID={os.getpid()})")

    try:
        broker.start()
        broker.run()
        logger.info(f"Broker run() returned, main() exiting (PID={os.getpid()})")
    except KeyboardInterrupt:
        logger.info(f"Broker interrupted (PID={os.getpid()})")
        broker.shutdown()
        broker.run()  # Let it exit cleanly and cleanup
    except Exception as e:
        logger.error(f"Broker failed: {e} (PID={os.getpid()})", exc_info=True)
        sys.exit(1)

    logger.info(f"Broker main() complete (PID={os.getpid()})")


if __name__ == '__main__':
    main()
