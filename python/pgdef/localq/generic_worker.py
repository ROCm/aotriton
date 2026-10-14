# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Generic worker that pulls from queue, handles messages, and forwards results.

Two things happen here that no handler has to do:

* every message a handler returns is checked against the handler's EMITS, and
  its `target_queue` is filled in from the target handler's QUEUE;
* any exception from a handler, on any queue, becomes a `task_failed` message:
  the engine's single failure path (pq.tasks.fail, then a negative ack to the
  PG reader). A handler never builds a failure message, and an unexpected
  exception can no longer strand a task or a reader slot.
"""

import logging
import os
import select
import socket
import time
from pathlib import Path

from .handlers import (BROKER_CLASSES, GRACEFUL_CANCEL, TASK_FAILED, EmitsViolation, HandlerSet,
                       NodeContext, Permanent, ack, task_header)
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

    def __init__(self, worker_id: str, queue_name: str, handler_set: HandlerSet,
                 ctx: NodeContext, broker_socket: str = '/tmp/aotriton-broker.sock',
                 idle_sleep_s: float = 0.5):
        """
        Initialize generic worker.

        Args:
            worker_id: Unique worker identifier
            queue_name: Queue to pull from
            handler_set: every handler class of the application (and the
                engine's); this worker instantiates those of its own queue
            ctx: the resources handlers may use (db_conn, runners, gpu_id)
            broker_socket: Path to broker Unix socket
            idle_sleep_s: how long to sleep when the queue is empty
        """
        self.idle_sleep_s = idle_sleep_s
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.broker_socket = broker_socket
        self.handler_set = handler_set
        self.ctx = ctx

        # One instance per handler class of this queue, keyed by message class
        self.handlers = {cls.CLASS: cls() for cls in handler_set.for_queue(queue_name)}

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
                    time.sleep(self.idle_sleep_s)
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
        task_id = message.get('task_id')

        try:
            handler = self.handlers.get(msg_class)
            if handler is None:
                raise RuntimeError(f"no handler for message class {msg_class!r} "
                                   f"on queue {self.queue_name!r}")

            logger.info(f"Handling {msg_class} (task_id={task_id})")
            result = handler.handle(message, self.ctx)
            results = self.prepare_results(handler, result)
        except Exception as e:
            logger.error(f"Handler error for {msg_class} (task_id={task_id}): {e}", exc_info=True)
            self._report_failure(message, e)
            return

        for i, r in enumerate(results):
            logger.info(f"Forwarding result {i + 1}/{len(results)} for {msg_class}")
            self._forward_message(r)
        logger.info(f"Completed {msg_class} (task_id={task_id})")

    def prepare_results(self, handler, result) -> list[dict]:
        """Normalise a handler's return value to a list of messages, check each
        against `handler.EMITS` and fill in `target_queue`.

        Raises EmitsViolation for a message the handler did not declare, so a
        graph edge cannot exist without being written next to its handler.
        """
        if result is None:
            return []
        results = [r for r in (result if isinstance(result, list) else [result]) if r]
        for r in results:
            cls = r.get('class')
            if cls not in handler.EMITS:
                raise EmitsViolation(f"{type(handler).__name__} (CLASS={handler.CLASS!r}) returned a "
                                     f"{cls!r} message, but EMITS is {handler.EMITS}")
            if cls in BROKER_CLASSES:
                continue
            queue = self.handler_set.queue_of(cls)
            if queue is None:
                raise EmitsViolation(f"{type(handler).__name__} emits {cls!r}, which no handler handles")
            if r.setdefault('target_queue', queue) != queue:
                raise EmitsViolation(f"{type(handler).__name__} set target_queue={r['target_queue']!r} "
                                     f"on {cls!r}, whose handler runs on {queue!r}")
        return results

    def _report_failure(self, message: dict, exc: Exception):
        """Turn a handler's exception into the engine's failure message."""
        if message.get('task_id') is None:
            logger.error("Cannot fail a message that carries no task_id; dropped")
            return
        if message['class'] in (TASK_FAILED, GRACEFUL_CANCEL):
            # The failure path itself failed (the database is probably down).
            # Do not loop: just release the PG reader. The task stays 'running'
            # with a lease nobody renews any more, so it lapses and the task is
            # reaped back to pending (or to failed, when out of attempts).
            self._forward_message(ack(message, negative=True))
            return
        failure = {
            'class': TASK_FAILED,
            'target_queue': self.handler_set.queue_of(TASK_FAILED),
            **task_header(message),
            'error': f"{message['class']}: {type(exc).__name__}: {exc}",
            'permanent': isinstance(exc, Permanent),
            'origin': {k: v for k, v in message.items() if k not in ('payload', 'fanin')},
        }
        self._forward_message(failure)

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
