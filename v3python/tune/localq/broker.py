# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
LocalBroker: Message router with dependency tracking for local queue.
"""

import logging
import select
import socket
import os
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List

from .buffered_socket import BufferedSocket

logger = logging.getLogger(__name__)


class LocalBroker:
    """
    Message router with dependency tracking.

    Manages:
    - Named queues (gpu_queue, cpu_queue, dispatcher_queue)
    - Dependency resolution (blocks messages until deps satisfied)
    - PG reader ack tracking (throttling mechanism)
    """

    def __init__(self, socket_path: str = '/tmp/aotriton-broker.sock'):
        self.socket_path = socket_path
        self.running = False

        # Named queues
        self.queues = {
            'gpu_queue': deque(),
            'cpu_queue': deque(),
            'dispatcher_queue': deque()
        }

        # Blocked messages waiting for dependencies
        # Map: depends_class_name → list of blocked messages
        self.blocked_messages = defaultdict(list)

        # PG reader ack tracking
        # Map: task_id → list of PG reader worker IDs waiting for ack
        self.pending_acks = defaultdict(list)

        # Worker connections
        self.workers = {}  # fd → BufferedSocket

        # epoll for efficient socket polling
        self.epoll = select.epoll()

        # Server socket
        self.server_sock = None

    def start(self):
        """Start broker server"""
        # Remove existing socket if present
        socket_path = Path(self.socket_path)
        if socket_path.exists():
            socket_path.unlink()

        # Create Unix domain socket
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen(128)  # Support many workers
        self.server_sock.setblocking(False)

        # Register server socket with epoll
        self.epoll.register(self.server_sock.fileno(), select.EPOLLIN)

        logger.info(f"Broker listening on {self.socket_path}")

    def shutdown(self):
        """Graceful shutdown - just set running flag, cleanup happens in run()"""
        logger.info("Shutting down broker")
        self.running = False

    def run(self):
        """Main broker event loop"""
        self.running = True
        while self.running:
            # Wait for socket events (timeout 0.1s for periodic tasks)
            events = self.epoll.poll(timeout=0.1)

            for fd, event in events:
                if fd == self.server_sock.fileno():
                    # New worker connection
                    self._accept_worker()
                elif event & select.EPOLLIN:
                    # Worker has data to read
                    self._handle_worker_message(fd)
                elif event & select.EPOLLOUT:
                    # Worker ready for writing
                    self._handle_worker_write(fd)
                elif event & (select.EPOLLHUP | select.EPOLLERR):
                    # Worker disconnected
                    self._remove_worker(fd)

        logger.info("Broker run loop exited, cleanup starting")

        # Cleanup resources
        if self.epoll:
            self.epoll.close()

        if self.server_sock:
            self.server_sock.close()

        # Clean up socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        logger.info("Broker run() complete")

    def _accept_worker(self):
        """Accept new worker connection"""
        try:
            conn, _ = self.server_sock.accept()
            # Set to non-blocking mode for multiplexed I/O with buffering
            conn.setblocking(False)

            # Register with epoll
            fd = conn.fileno()
            self.epoll.register(fd, select.EPOLLIN)

            # Create buffered socket (worker_id set on first message)
            buffered_sock = BufferedSocket(conn, worker_id=None)
            self.workers[fd] = buffered_sock

            logger.debug(f"Accepted worker connection (fd={fd})")

        except Exception as e:
            logger.error(f"Error accepting worker: {e}")

    def _handle_worker_message(self, fd: int):
        """Handle messages from worker"""
        buffered_sock = self.workers.get(fd)
        if not buffered_sock:
            return

        try:
            # Receive and parse messages (generator yields complete messages)
            for msg in buffered_sock.recv_messages():
                msg_type = msg['type']

                if msg_type == 'get_task':
                    self._handle_get_task(buffered_sock, msg)

                elif msg_type == 'forward':
                    self._handle_forward(msg)

                elif msg_type == 'register_ack':
                    self._handle_register_ack(msg)

        except ConnectionError:
            # Connection closed
            logger.info(f"Worker {buffered_sock.worker_id} connection closed")
            self._remove_worker(fd)

        except Exception as e:
            logger.error(f"Error handling worker message: {e}", exc_info=True)
            self._remove_worker(fd)

    def _handle_worker_write(self, fd: int):
        """Handle worker ready for writing (EPOLLOUT)"""
        buffered_sock = self.workers.get(fd)
        if not buffered_sock:
            return

        try:
            # Flush send buffer
            all_sent = buffered_sock.flush_send()

            if all_sent:
                # All data sent, unregister EPOLLOUT
                self.epoll.modify(fd, select.EPOLLIN)

        except Exception as e:
            logger.error(f"Error handling worker write: {e}", exc_info=True)
            self._remove_worker(fd)

    def _handle_get_task(self, buffered_sock: BufferedSocket, msg: dict):
        """Worker requesting task from queue"""
        queue_name = msg['queue_name']
        worker_id = msg['worker_id']

        # Update worker_id if not set
        if buffered_sock.worker_id is None:
            buffered_sock.worker_id = worker_id
            logger.info(f"Worker {worker_id} connected to {queue_name} (fd={buffered_sock.fileno})")

        # Dequeue task
        task_msg = self._dequeue_task(queue_name)

        if task_msg:
            self._send_to_worker(buffered_sock,{
                'type': 'task',
                'message': task_msg
            })
        else:
            self._send_to_worker(buffered_sock,{'type': 'no_task'})

    def _handle_forward(self, msg: dict):
        """Worker forwarding result message"""
        message = msg['message']
        self.forward(message)

    def _handle_register_ack(self, msg: dict):
        """PG reader registering for ack"""
        task_id = msg['task_id']
        worker_id = msg['worker_id']
        self.pending_acks[task_id].append(worker_id)
        logger.debug(f"Registered ack for task_id={task_id} from {worker_id}")

    def _send_to_worker(self, buffered_sock: BufferedSocket, msg: dict):
        """
        Send message to worker (non-blocking with buffering).

        Args:
            buffered_sock: Worker's buffered socket
            msg: Message to send
        """
        # Queue message for sending
        buffered_sock.queue_message(msg)

        # Register for EPOLLOUT if not already
        fd = buffered_sock.fileno
        try:
            self.epoll.modify(fd, select.EPOLLIN | select.EPOLLOUT)
        except Exception:
            # Socket may not be registered yet, ignore
            pass

    def _remove_worker(self, fd: int):
        """Remove disconnected worker"""
        buffered_sock = self.workers.pop(fd, None)
        if buffered_sock:
            logger.info(f"Worker {buffered_sock.worker_id} disconnected (fd={fd})")
            try:
                self.epoll.unregister(fd)
                buffered_sock.sock.close()
            except Exception:
                pass

    def forward(self, message: dict):
        """
        Forward message to its target queue or handle dependencies.

        Args:
            message: Message to forward
        """
        if message is None:
            return

        # Check if message has dependencies
        if 'depends' in message and message['depends']:
            # Message is blocked, store it
            for dep_class in message['depends']:
                self.blocked_messages[dep_class].append(message)
            logger.debug(f"Blocked {message['class']} (task_id={message.get('task_id')}) "
                        f"waiting for {message['depends']}")
            return

        # Check if this message resolves any dependencies
        self._resolve_dependencies(message)

        # Check if this is an ack message
        if message['class'] == 'tune_kernel_ack':
            self._handle_ack(message)
            return

        # Enqueue message to target queue
        target_queue = message.get('target_queue')
        if target_queue:
            self._enqueue_with_priority(target_queue, message)

    def _resolve_dependencies(self, incoming_msg: dict):
        """
        Check if incoming message resolves any blocked messages.

        Args:
            incoming_msg: Newly arrived message
        """
        msg_class = incoming_msg['class']

        if msg_class not in self.blocked_messages:
            return

        # Get handler registry from first worker that has it
        # (This is a simplification - in real impl, broker should have handler references)
        # For now, we use PostprocessHandler.resolve_dependency directly

        blocked_list = self.blocked_messages[msg_class]
        unblocked = []

        for blocked_msg in blocked_list:
            # Import here to avoid circular dependency
            from .handlers import PostprocessHandler

            # Create temporary handler instance to call resolve_dependency
            # (In production, broker should maintain handler registry)
            handler = PostprocessHandler(conn_params={})

            if handler.resolve_dependency(blocked_msg, incoming_msg):
                # Dependency resolved
                unblocked.append(blocked_msg)
                logger.debug(f"Unblocked {blocked_msg['class']} "
                           f"(task_id={blocked_msg.get('task_id')})")

        # Remove unblocked messages and forward them
        for msg in unblocked:
            blocked_list.remove(msg)

            # Remove 'depends' field since it's now resolved
            msg.pop('depends', None)

            # Forward to target queue
            self.forward(msg)

    def _enqueue_with_priority(self, queue_name: str, message: dict):
        """
        Enqueue message with priority ordering.

        Args:
            queue_name: Target queue name
            message: Message to enqueue
        """
        if queue_name not in self.queues:
            logger.error(f"Unknown queue: {queue_name}")
            return

        queue = self.queues[queue_name]
        priority = self._get_priority(message['class'])

        # Insert in priority order
        insert_pos = len(queue)
        for i, existing_msg in enumerate(queue):
            existing_priority = self._get_priority(existing_msg['class'])
            if priority > existing_priority:
                insert_pos = i
                break

        queue.insert(insert_pos, message)
        logger.debug(f"Enqueued {message['class']} to {queue_name} "
                    f"at position {insert_pos} (priority={priority})")

    def _get_priority(self, msg_class: str) -> int:
        """
        Get message priority (higher = more urgent).

        Args:
            msg_class: Message class name

        Returns:
            Priority value
        """
        priorities = {
            'postprocess': 4,     # Highest - frees resources, sends ack
            'probe': 3,           # High - generates tune_hsaco tasks
            'tune_hsaco': 2,      # Medium - actual GPU work
            'preprocess': 1,      # Low - just setup
            'hsaco_result': 0     # Lowest - CPU write
        }
        return priorities.get(msg_class, 0)

    def _dequeue_task(self, queue_name: str) -> dict | None:
        """
        Get next task from named queue (FIFO within priority).

        Args:
            queue_name: Queue to dequeue from

        Returns:
            Task message or None if queue empty
        """
        if queue_name not in self.queues:
            logger.error(f"Unknown queue: {queue_name}")
            return None

        queue = self.queues[queue_name]

        if queue:
            msg = queue.popleft()
            logger.debug(f"Dequeued {msg['class']} from {queue_name}")
            return msg
        else:
            return None

    def _handle_ack(self, ack_msg: dict):
        """
        Handle tune_kernel_ack - notify PG reader workers.

        Args:
            ack_msg: Ack message with task_id
        """
        task_id = ack_msg['task_id']

        if task_id not in self.pending_acks:
            logger.warning(f"Received ack for task_id={task_id} but no pending acks")
            return

        # Send ack to waiting PG reader workers
        for worker_id in self.pending_acks[task_id]:
            buffered_sock = self._find_worker_by_id(worker_id)
            if buffered_sock:
                self._send_to_worker(buffered_sock,{
                    'type': 'ack',
                    'task_id': task_id
                })
                logger.debug(f"Sent ack to {worker_id} for task_id={task_id}")

        del self.pending_acks[task_id]

    def _find_worker_by_id(self, worker_id: str) -> BufferedSocket | None:
        """
        Find worker by ID.

        Args:
            worker_id: Worker identifier

        Returns:
            BufferedSocket or None
        """
        for buffered_sock in self.workers.values():
            if buffered_sock.worker_id == worker_id:
                return buffered_sock
        return None
