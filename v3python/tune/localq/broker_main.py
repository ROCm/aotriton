# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Broker main entry point.
"""

import sys
import logging
import argparse

from .broker import LocalBroker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Broker main entry point"""
    parser = argparse.ArgumentParser(description='Local queue broker')
    parser.add_argument('--socket', type=str,
                       default='/tmp/aotriton-broker.sock',
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    broker = LocalBroker(socket_path=args.socket)

    logger.info("Starting LocalBroker")

    try:
        broker.start()
        broker.run()
    except KeyboardInterrupt:
        logger.info("Broker interrupted")
    except Exception as e:
        logger.error(f"Broker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
