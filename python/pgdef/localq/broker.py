# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
LocalBroker: message router with dependency tracking for the local queue.

The broker knows no application. It is started with the application's handler
classes (class -> handler, for joins and teardown), their priorities and the
names of the queues that have workers on this node.
"""

import logging
import select
import socket
import os
from collections import deque, defaultdict
from pathlib import Path

from .buffered_socket import BufferedSocket
from .handlers import DAG_ACK, QUEUES, MessageHandler

logger = logging.getLogger(__name__)


class LocalBroker:
    """
    Message router with dependency tracking.

    Manages:
    - Named queues ('gpu', 'cpu'): only the ones that have workers on this node
    - Dependency resolution (blocks messages until deps satisfied)
    - PG reader ack tracking (throttling mechanism)

    A message for a queue with no workers is a configuration error, not
    something to wait for: it fails the task (the reader holding it gets a
    negative ack carrying the reason) and the task's other queued and blocked
    messages are dropped.
    """

    def __init__(self, socket_path: str = '/tmp/aotriton-broker.sock', *,
                 handlers: dict | None = None,
                 priorities: dict | None = None,
                 ack_class: str = DAG_ACK,
                 queues: tuple[str, ...] = QUEUES):
        """
        Args:
            socket_path: Unix socket to listen on
            handlers: message class -> object providing resolve_dependency()
                and teardown_with_unmet_dependency(), for every class that
                can be blocked on a dependency. The application supplies it;
                the broker knows no class names.
            priorities: message class -> priority (higher = more urgent);
                classes not listed get 0.
            ack_class: class of the message that releases a PG reader.
            queues: names of the queues that have workers. A zero-GPU node
                passes ('cpu',), a zero-CPU node ('gpu',).
        """
        self.socket_path = socket_path
        self.handlers = dict(handlers or {})
        self.priorities = dict(priorities or {})
        self.ack_class = ack_class
        self.running = False
        self.graceful_shutdown_requested = False

        # Named queues
        self.queues = {name: deque() for name in queues}

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

    def graceful_shutdown(self):
        """Mark broker for graceful shutdown with teardown (called from SIGHUP handler)"""
        logger.info("Graceful shutdown requested - will teardown blocked messages")
        self.graceful_shutdown_requested = True
        # Do NOT set self.running = False - broker must continue serving
        # teardown messages to CPU workers after SIGHUP

    def run(self):
        """Main broker event loop"""
        self.running = True
        torn_down = False
        while self.running:
            # Graceful shutdown (SIGHUP): tear blocked messages down WHILE the
            # loop still runs, so the CPU workers (still alive at this point,
            # see worker_service.sh's stop order) receive the graceful_cancel
            # messages and give the tasks back. Tearing down only after the loop
            # has exited would enqueue them into a broker nobody reads anymore.
            if self.graceful_shutdown_requested and not torn_down:
                logger.info("Graceful shutdown flag set, tearing down blocked messages")
                self._teardown_blocked_messages()
                torn_down = True

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

        # Graceful shutdown requested after the last loop iteration: nothing
        # will serve the teardown messages any more, but still run it so the
        # handlers' teardown side effects (e.g. freeing prepared data) happen.
        if self.graceful_shutdown_requested and not torn_down:
            logger.info("Graceful shutdown flag set, tearing down blocked messages")
            self._teardown_blocked_messages()

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
                worker_id = buffered_sock.worker_id or f"fd={fd}"

                logger.debug(f"Broker received message from {worker_id}: type={msg_type}")

                if msg_type == 'get_task':
                    self._handle_get_task(buffered_sock, msg)

                elif msg_type == 'forward':
                    self._handle_forward(msg)

                elif msg_type == 'register_ack':
                    self._handle_register_ack(buffered_sock, msg)

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
        logger.info(f"Broker handling forward: class={message['class']}, task_id={message.get('task_id')}")
        self.forward(message)

    def _handle_register_ack(self, buffered_sock: BufferedSocket, msg: dict):
        """PG reader registering for ack"""
        task_id = msg['task_id']
        worker_id = msg['worker_id']

        # Set worker_id on socket if not already set (PG readers don't call get_task)
        if buffered_sock.worker_id is None:
            buffered_sock.worker_id = worker_id
            logger.info(f"PG Reader {worker_id} connected (fd={buffered_sock.fileno})")

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

        # An ack is for the broker itself: it has no queue.
        if message['class'] != self.ack_class and message.get('target_queue') not in self.queues:
            self._undeliverable(message)
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
        if message['class'] == self.ack_class:
            self._handle_ack(message)
            return

        self._enqueue_with_priority(message['target_queue'], message)

    def _resolve_dependencies(self, incoming_msg: dict):
        """
        Check if incoming message resolves any blocked messages.

        Args:
            incoming_msg: Newly arrived message
        """
        msg_class = incoming_msg['class']

        if msg_class not in self.blocked_messages:
            return

        blocked_list = self.blocked_messages[msg_class]
        unblocked = []

        for blocked_msg in blocked_list:
            handler = self.handlers.get(blocked_msg['class'], MessageHandler)

            if handler.resolve_dependency(blocked_msg, incoming_msg):
                # Dependency resolved
                unblocked.append(blocked_msg)
                logger.debug(f"Unblocked {blocked_msg['class']} "
                           f"(task_id={blocked_msg.get('task_id')})")

        # Remove unblocked messages and forward them. A message that depends on
        # several classes is parked under each of them: drop every copy, or the
        # stale ones would linger and be torn down again at shutdown.
        for msg in unblocked:
            for dep_class in msg.get('depends') or (msg_class,):
                parked = self.blocked_messages.get(dep_class)
                if parked is None:
                    continue
                parked[:] = [m for m in parked if m is not msg]
                if not parked:
                    del self.blocked_messages[dep_class]

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
            self._undeliverable(message)
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
        return self.priorities.get(msg_class, 0)

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
        Handle the ack message - notify PG reader workers.

        Args:
            ack_msg: Ack message with task_id
        """
        task_id = ack_msg['task_id']

        logger.info(f"Broker received {self.ack_class} for task_id={task_id}")

        if task_id not in self.pending_acks:
            logger.warning(f"Received ack for task_id={task_id} but no pending acks")
            return

        # Send ack to waiting PG reader workers
        for worker_id in self.pending_acks[task_id]:
            buffered_sock = self._find_worker_by_id(worker_id)
            if buffered_sock:
                reply = {'type': 'ack', 'task_id': task_id}
                if ack_msg.get('negative'):
                    reply['negative'] = True
                if ack_msg.get('error') is not None:
                    reply['error'] = ack_msg['error']
                self._send_to_worker(buffered_sock, reply)
                logger.info(f"Sent ack to {worker_id} for task_id={task_id}")
            else:
                logger.warning(f"Could not find worker {worker_id} to send ack for task_id={task_id}")

        del self.pending_acks[task_id]

    def _undeliverable(self, message: dict):
        """`message` targets a queue this node has no workers for.

        That is a configuration error (an application whose graph needs a GPU
        step started on a node with zero GPU workers, say), and waiting would
        hang the task forever. Fail it instead: the reader holding the task is
        sent a negative ack that carries the reason, and records the failure in
        the database (it has the connection; the broker has none). The task's
        other messages are dropped so that nothing keeps running for a task
        that has already failed.
        """
        task_id = message.get('task_id')
        reason = (f"message {message.get('class')!r} is for queue {message.get('target_queue')!r}, "
                  f"but this node has workers only for {sorted(self.queues)}")
        logger.error(f"Undeliverable (task_id={task_id}): {reason}")
        if task_id is None:
            return
        self._purge_task(task_id)
        self._handle_ack({'class': self.ack_class, 'task_id': task_id,
                          'negative': True, 'error': reason})

    def _purge_task(self, task_id):
        """Drop every queued and blocked message of one task."""
        for queue in self.queues.values():
            keep = [m for m in queue if m.get('task_id') != task_id]
            queue.clear()
            queue.extend(keep)
        for dep_class in list(self.blocked_messages):
            self.blocked_messages[dep_class] = [
                m for m in self.blocked_messages[dep_class] if m.get('task_id') != task_id]
            if not self.blocked_messages[dep_class]:
                del self.blocked_messages[dep_class]

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

    def _teardown_blocked_messages(self):
        """Teardown all blocked messages (tasks with unmet dependencies)"""
        # A message that depends on several classes is parked under each: tear
        # it down once.
        unique: dict[int, dict] = {}
        for messages in self.blocked_messages.values():
            for msg in messages:
                unique.setdefault(id(msg), msg)
        logger.info(f"Tearing down {len(unique)} blocked messages")

        for msg in unique.values():
            msg_class = msg.get('class')
            if not msg_class:
                continue

            handler = self.handlers.get(msg_class, MessageHandler)
            logger.info(f"Calling teardown for {msg_class}: task_id={msg.get('task_id')}")
            result_msg = handler.teardown_with_unmet_dependency(msg)

            if result_msg:
                self._enqueue_with_priority(result_msg['target_queue'], result_msg)

        self.blocked_messages.clear()
        logger.info("Blocked messages teardown complete")
