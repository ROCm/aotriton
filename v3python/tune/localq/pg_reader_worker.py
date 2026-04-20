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
import select
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .protocol import send_message, recv_message
from ..utils import get_db_connection_params, configure_logging_with_flush
from ..pq.queue import TaskQueue

configure_logging_with_flush()

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
        self.db_conn = None
        self.running = False

        # Create wakeup pipe for signal handling
        self.wakeup_read_fd, self.wakeup_write_fd = os.pipe()
        os.set_blocking(self.wakeup_read_fd, False)
        os.set_blocking(self.wakeup_write_fd, False)

    def run(self):
        """Main PG reader loop"""
        # Connect to broker
        self._connect_to_broker()

        # Connect to database (reuse connection)
        self._connect_to_database()

        logger.info(f"PG Reader {self.worker_id} started for arch={self.arch} (PID={os.getpid()})")
        self.running = True

        while self.running:
            try:
                # Fetch task from PostgreSQL
                logger.debug(f"PG Reader {self.worker_id} calling _fetch_pg_task()")
                task = self._fetch_pg_task()

                if task is None:
                    # No tasks available
                    logger.debug(f"PG Reader {self.worker_id} no tasks available, sleeping 1s")
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
                while self.running:
                    # Wait for socket with signal interruption support
                    if not self._wait_for_socket():
                        # Signal received, check running flag
                        if not self.running:
                            logger.info("Shutdown signal received during ack wait")
                            return
                        continue

                    response = recv_message(self.sock)

                    if response is None:
                        logger.error("Broker connection closed")
                        return

                    if response['type'] == 'ack' and response['task_id'] == task_id:
                        logger.info(f"Received ack for task_id={task_id}, continuing")
                        break

                logger.info(f"PG Reader {self.worker_id} finished ack wait loop, back to main loop")

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
        logger.info(f"PG Reader {self.worker_id} cleanup starting (PID={os.getpid()})")
        if self.db_conn:
            self.db_conn.close()
        if self.sock:
            self.sock.close()
        logger.info(f"PG Reader {self.worker_id} cleanup complete, exiting run() (PID={os.getpid()})")

    def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"PG Reader {self.worker_id} shutting down")
        self.running = False

    def _wait_for_socket(self, timeout=None):
        """
        Wait for socket to be readable, with signal interruption support.

        Returns:
            True if socket is readable, False if signal received or timeout
        """
        ready, _, _ = select.select([self.sock.fileno(), self.wakeup_read_fd], [], [], timeout)

        if self.wakeup_read_fd in ready:
            # Signal received, drain the wakeup pipe
            try:
                os.read(self.wakeup_read_fd, 1024)
            except BlockingIOError:
                pass
            return False

        return self.sock.fileno() in ready

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

    def _connect_to_database(self):
        """Connect to PostgreSQL database (persistent connection)"""
        # Set statement_timeout to 1 second to prevent blocking queries
        options = f"-c statement_timeout=1000"  # 1000ms = 1s
        self.db_conn = psycopg.connect(
            **self.conn_params,
            row_factory=dict_row,
            options=options,
            autocommit=True  # Auto-commit mode for simpler transaction handling
        )
        logger.info(f"PG Reader {self.worker_id} connected to database")

    def _fetch_pg_task(self):
        """
        Fetch and claim task from PostgreSQL task_queue.

        Returns:
            Task dict or None if no tasks available
        """
        try:
            task_queue = TaskQueue(self.db_conn)
            tasks = task_queue.fetch_tasks(self.arch, batch_size=1)

            if tasks:
                task = tasks[0]
                logger.info(f"PG Reader {self.worker_id} fetched task from database: "
                           f"id={task.id}, arch={task.arch}, module={task.module}, "
                           f"status=pending→running")
                return {
                    'id': task.id,
                    'arch': task.arch,
                    'module': task.module,
                    'task_config': task.task_config
                }
            else:
                logger.debug(f"PG Reader {self.worker_id} no tasks available")
                return None

        except psycopg.errors.QueryCanceled:
            # Statement timeout - no tasks available
            logger.warning(f"PG Reader {self.worker_id} query timeout (no pending tasks or lock contention)")
            return None
        except Exception as e:
            logger.error(f"PG Reader {self.worker_id} database error in _fetch_pg_task: {e}", exc_info=True)
            return None


def main():
    """PG reader worker main entry point"""
    parser = argparse.ArgumentParser(description='PostgreSQL reader worker')
    parser.add_argument('--worker_id', type=str, required=True,
                       help='Worker identifier (e.g., pg-reader-0)')
    parser.add_argument('--arch', type=str, required=True,
                       help='GPU architecture to fetch tasks for')
    parser.add_argument('--workdir', type=str, required=True,
                       help='Path to workdir containing config.rc')
    parser.add_argument('--broker_socket', type=str,
                       default=os.environ.get('AOTRITON_TUNER_BROKER_SOCKET', '/tmp/aotriton-broker.sock'),
                       help='Path to broker Unix socket')
    args = parser.parse_args()

    # Get database connection parameters
    from pathlib import Path
    conn_params = get_db_connection_params(Path(args.workdir))

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

    # Set wakeup fd to interrupt blocking I/O on signals
    signal.set_wakeup_fd(worker.wakeup_write_fd)

    try:
        worker.run()
        logger.info(f"PG Reader {args.worker_id} run() returned, main() exiting (PID={os.getpid()})")
    except KeyboardInterrupt:
        logger.info(f"PG Reader {args.worker_id} interrupted (PID={os.getpid()})")
        worker.shutdown()
    except Exception as e:
        logger.error(f"PG Reader {args.worker_id} failed: {e} (PID={os.getpid()})", exc_info=True)
        worker.shutdown()
        sys.exit(1)

    logger.info(f"PG Reader {args.worker_id} main() complete (PID={os.getpid()})")


if __name__ == '__main__':
    main()
