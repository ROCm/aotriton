# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generic worker that pulls from queue, handles messages, and forwards results.
"""

import logging
import os
import select
import socket
import time
from pathlib import Path
from typing import Dict, List

from .protocol import send_message, recv_message
from ..pq.queue import TaskQueue

logger = logging.getLogger(__name__)


class GenericWorker:
    """
    Generic worker that pulls from queue, handles, and forwards.

    Workflow:
    1. get_task() from broker
    2. Find handler for message class
    3. Execute handler
    4. Forward result(s) to broker
    """

    def __init__(self, worker_id: str, queue_name: str, handlers: list,
                 broker_socket: str = '/tmp/aotriton-broker.sock',
                 db_conn = None):
        """
        Initialize generic worker.

        Args:
            worker_id: Unique worker identifier
            queue_name: Queue to pull from
            handlers: List of MessageHandler instances
            broker_socket: Path to broker Unix socket
            db_conn: PostgreSQL connection (for error handling)
        """
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.broker_socket = broker_socket
        self.db_conn = db_conn

        # Build handler registry
        self.handlers = {h.get_class_name(): h for h in handlers}

        # Socket connection to broker
        self.sock = None
        self.running = False

        # Create wakeup pipe for signal handling
        self.wakeup_read_fd, self.wakeup_write_fd = os.pipe()
        os.set_blocking(self.wakeup_read_fd, False)
        os.set_blocking(self.wakeup_write_fd, False)

    def run(self):
        """Main worker loop"""
        # Connect to broker
        self._connect_to_broker()

        logger.info(f"Worker {self.worker_id} started, pulling from {self.queue_name}")
        self.running = True

        while self.running:
            try:
                # Get task from queue
                send_message(self.sock, {
                    'type': 'get_task',
                    'queue_name': self.queue_name,
                    'worker_id': self.worker_id
                })

                # Wait for socket with signal interruption support
                if not self._wait_for_socket():
                    # Signal received, check running flag
                    if not self.running:
                        logger.info("Shutdown signal received")
                        break
                    continue

                task = recv_message(self.sock)

                if task is None:
                    # Connection closed
                    logger.error("Broker connection closed")
                    break

                if task['type'] == 'no_task':
                    # No work available, sleep and retry
                    time.sleep(0.5)
                    continue

                elif task['type'] == 'shutdown':
                    logger.info(f"Worker {self.worker_id} received shutdown")
                    break

                elif task['type'] == 'task':
                    # Execute task
                    message = task['message']
                    logger.info(f"Worker {self.worker_id} received task: {message['class']} (task_id={message.get('task_id')})")
                    self._handle_task(message)

            except KeyboardInterrupt:
                logger.info(f"Worker {self.worker_id} interrupted")
                break

            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                logger.error(f"Worker {self.worker_id} lost broker connection: {e}")
                break

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                time.sleep(1)  # Backoff on error

        # Cleanup
        logger.info(f"Worker {self.worker_id} cleanup starting")
        if self.sock:
            self.sock.close()
        logger.info(f"Worker {self.worker_id} cleanup complete, exiting run()")

    def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Worker {self.worker_id} shutting down")
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

                # Increase socket buffer sizes to reduce blocking on send
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB send buffer
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB recv buffer

                self.sock.connect(self.broker_socket)
                logger.info(f"Worker {self.worker_id} connected to broker")
                return

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def _handle_task(self, message: dict):
        """
        Execute task handler and forward results.

        Args:
            message: Task message to handle
        """
        msg_class = message['class']

        # Find handler
        handler = self.handlers.get(msg_class)

        if not handler:
            logger.error(f"No handler for message class: {msg_class}")
            return

        try:
            # Execute handler
            logger.info(f"Handling {msg_class} (task_id={message.get('task_id')})")
            result = handler.handle(message)

            logger.info(f"Handler returned for {msg_class} (task_id={message.get('task_id')})")

            # Forward result(s)
            if isinstance(result, list):
                # Multiple results (e.g., probe returns many messages)
                for i, r in enumerate(result):
                    if r:
                        logger.info(f"Forwarding result {i+1}/{len(result)} for {msg_class}")
                        self._forward_message(r)
            elif result:
                # Single result
                logger.info(f"Forwarding result for {msg_class}")
                self._forward_message(result)
                logger.info(f"Forward completed for {msg_class}")
            # else: result is None, no forwarding

            logger.info(f"Completed {msg_class} (task_id={message.get('task_id')})")

        except Exception as e:
            task_id = message.get('task_id')
            logger.error(f"Handler error for {msg_class} (task_id={task_id}): {e}", exc_info=True)

            # Mark task as failed in database if this is a top-level task handler
            if msg_class in ['tune_kernel', 'preprocess', 'probe'] and task_id and self.db_conn:
                try:
                    task_queue = TaskQueue(self.db_conn)
                    error_msg = f"{msg_class} failed: {type(e).__name__}: {str(e)}"
                    logger.error(f"Marking task_id={task_id} as failed in database: {error_msg}")
                    task_queue.mark_failed(task_id, error_msg)
                except Exception as db_error:
                    logger.error(f"Failed to mark task_id={task_id} as failed in database: {db_error}",
                                exc_info=True)

            # Mark task as failed in database if this is a top-level task handler
            if msg_class in ['tune_kernel', 'preprocess', 'probe'] and task_id:
                self._mark_task_failed(task_id, str(e))

    def _forward_message(self, message: dict):
        """
        Forward message to broker.

        Args:
            message: Message to forward
        """
        send_message(self.sock, {
            'type': 'forward',
            'message': message
        })
