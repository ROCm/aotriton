# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
PostgreSQL Reader Worker with throttling.

Fetches tasks from PostgreSQL task_queue and sends to broker.
Blocks until tune_kernel completes (via ack) to throttle task fetching.
"""

import sys
import os
import time
import logging
import argparse
import socket
import signal
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .protocol import send_message, recv_message

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class PGReaderWorker:
    """
    Fetches tasks from PostgreSQL and sends to broker.
    Blocks until tune_kernel completes (via ack).
    """

    def __init__(self, worker_id: str, arch: str, broker_socket: str, conn_params: dict):
        """
        Initialize PG reader worker.

        Args:
            worker_id: Unique worker identifier
            arch: GPU architecture to fetch tasks for
            broker_socket: Path to broker Unix socket
            conn_params: PostgreSQL connection parameters
        """
        self.worker_id = worker_id
        self.arch = arch
        self.broker_socket = broker_socket
        self.conn_params = conn_params
        self.sock = None
        self.running = False

    def run(self):
        """Main PG reader loop"""
        # Connect to broker
        self._connect_to_broker()

        logger.info(f"PG Reader {self.worker_id} started for arch={self.arch}")
        self.running = True

        while self.running:
            try:
                # Fetch task from PostgreSQL
                task = self._fetch_pg_task()

                if task is None:
                    # No tasks available
                    time.sleep(1)
                    continue

                task_id = task['id']
                task_config = task['task_config']

                logger.info(f"Fetched task_id={task_id} from PostgreSQL")

                # Register for ack
                send_message(self.sock, {
                    'type': 'register_ack',
                    'task_id': task_id,
                    'worker_id': self.worker_id
                })

                # Send tune_kernel message to broker
                tune_kernel_msg = {
                    'class': 'tune_kernel',
                    'target_queue': 'gpu_queue',
                    'task_id': task_id,
                    'task_config': task_config
                }

                send_message(self.sock, {
                    'type': 'forward',
                    'message': tune_kernel_msg
                })

                logger.debug(f"Forwarded tune_kernel for task_id={task_id}")

                # Wait for ack (BLOCKING - this throttles PG fetching)
                while True:
                    response = recv_message(self.sock)

                    if response is None:
                        logger.error("Broker connection closed")
                        return

                    if response['type'] == 'ack' and response['task_id'] == task_id:
                        logger.info(f"Received ack for task_id={task_id}, continuing")
                        break

            except KeyboardInterrupt:
                logger.info(f"PG Reader {self.worker_id} interrupted")
                break

            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                logger.error(f"PG Reader {self.worker_id} lost broker connection: {e}")
                break

            except Exception as e:
                logger.error(f"PG Reader {self.worker_id} error: {e}", exc_info=True)
                time.sleep(5)  # Backoff on error

        # Cleanup
        if self.sock:
            self.sock.close()

    def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"PG Reader {self.worker_id} shutting down")
        self.running = False

    def _connect_to_broker(self):
        """Connect to broker socket"""
        max_retries = 10
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.broker_socket)
                logger.info(f"PG Reader {self.worker_id} connected to broker")
                return

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def _fetch_pg_task(self):
        """
        Fetch and claim task from PostgreSQL task_queue.

        Returns:
            Task dict or None if no tasks available
        """
        partition_table = f"task_queue_{self.arch}"

        with psycopg.connect(**self.conn_params, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Atomic task claiming using UPDATE ... RETURNING
                cur.execute(f"""
                    UPDATE {partition_table}
                    SET status = 'running',
                        worker_id = %s,
                        started_at = NOW()
                    WHERE id IN (
                        SELECT id FROM {partition_table}
                        WHERE status = 'pending'
                        ORDER BY priority DESC, id ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, arch, module, task_config
                """, (self.worker_id,))

                row = cur.fetchone()
                conn.commit()

                if row:
                    return dict(row)
                else:
                    return None


def get_db_connection_params():
    """Get PostgreSQL connection parameters from environment"""
    return {
        'host': os.environ.get('CELERY_SERVICE_HOST', 'localhost'),
        'port': int(os.environ.get('POSTGRES_PORT', 5432)),
        'user': os.environ.get('POSTGRES_USER'),
        'password': os.environ.get('POSTGRES_PASSWORD'),
    }


def main():
    """PG reader worker main entry point"""
    parser = argparse.ArgumentParser(description='PostgreSQL reader worker')
    parser.add_argument('--worker_id', type=str, required=True,
                       help='Worker identifier (e.g., pg-reader-0)')
    parser.add_argument('--arch', type=str, required=True,
                       help='GPU architecture to fetch tasks for')
    parser.add_argument('--broker_socket', type=str,
                       default=os.environ.get('AOTRITON_TUNER_BROKER_SOCKET', '/tmp/aotriton-broker.sock'),
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    # Get database connection parameters
    conn_params = get_db_connection_params()

    # Create and run worker
    worker = PGReaderWorker(
        worker_id=args.worker_id,
        arch=args.arch,
        broker_socket=args.broker_socket,
        conn_params=conn_params
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        worker.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        worker.run()
    except KeyboardInterrupt:
        logger.info(f"PG Reader {args.worker_id} interrupted")
        worker.shutdown()
    except Exception as e:
        logger.error(f"PG Reader {args.worker_id} failed: {e}", exc_info=True)
        worker.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
