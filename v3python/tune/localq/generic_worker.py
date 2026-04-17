# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generic worker that pulls from queue, handles messages, and forwards results.
"""

import logging
import socket
import time
from pathlib import Path
from typing import Dict, List

from .protocol import send_message, recv_message

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
                 broker_socket: str = '/tmp/aotriton-broker.sock'):
        """
        Initialize generic worker.

        Args:
            worker_id: Unique worker identifier
            queue_name: Queue to pull from
            handlers: List of MessageHandler instances
            broker_socket: Path to broker Unix socket
        """
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.broker_socket = broker_socket

        # Build handler registry
        self.handlers = {h.get_class_name(): h for h in handlers}

        # Socket connection to broker
        self.sock = None
        self.running = False

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

                task = recv_message(self.sock)

                if task is None:
                    # Connection closed
                    logger.error("Broker connection closed")
                    break

                if task['type'] == 'no_task':
                    # No work available
                    time.sleep(0.5)
                    continue

                elif task['type'] == 'shutdown':
                    logger.info(f"Worker {self.worker_id} received shutdown")
                    break

                elif task['type'] == 'task':
                    # Execute task
                    message = task['message']
                    self._handle_task(message)

            except KeyboardInterrupt:
                logger.info(f"Worker {self.worker_id} interrupted")
                break

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                time.sleep(1)  # Backoff on error

        # Cleanup
        if self.sock:
            self.sock.close()

    def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Worker {self.worker_id} shutting down")
        self.running = False

    def _connect_to_broker(self):
        """Connect to broker socket"""
        max_retries = 10
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
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
            logger.debug(f"Handling {msg_class} (task_id={message.get('task_id')})")
            result = handler.handle(message)

            # Forward result(s)
            if isinstance(result, list):
                # Multiple results (e.g., probe returns many messages)
                for r in result:
                    if r:
                        self._forward_message(r)
            elif result:
                # Single result
                self._forward_message(result)
            # else: result is None, no forwarding

            logger.debug(f"Completed {msg_class} (task_id={message.get('task_id')})")

        except Exception as e:
            logger.error(f"Handler error for {msg_class}: {e}", exc_info=True)
            # TODO: Send error message to broker

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
