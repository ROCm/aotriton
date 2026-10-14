# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
PostgreSQL reader with throttling.

Claims one task from pgdef.task, sends the application's start message to the
broker, and holds the task until the broker acknowledges it. One task per
reader is what throttles how much work a node takes: N readers, N tasks in
flight.

While it holds a task the reader renews the task's lease every `lease_s / 3`,
but only while its broker connection is alive. If the broker dies the reader
exits and the lease lapses, so the task is reaped back to pending (or to failed
when out of attempts) by another reader's `reap_expired`, with no daemon and no
manual reset. Every reader runs a rate-limited reap before it claims.

The reader also upserts its pgdef.worker row on claim, renewal and idle poll,
so worker health reflects real progress.
"""

import logging
import os
import select
import socket
import time

from psycopg.rows import dict_row

from aotriton.pgdef.pq import tasks as pgtasks
from aotriton.pgdef.pq import workers as pgworkers
from aotriton.pgdef.pq.category import Category
from aotriton.pgdef.pq.conn import ReconnectableConn

from .handlers import StartHandler
from .protocol import recv_message, send_message

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 600.0
REAP_INTERVAL_SECONDS = 60.0


class PGReader:
    """
    Fetches tasks from PostgreSQL and sends them to the broker. Blocks until
    the task's graph acknowledges (or fails).

    TODO: Consider redesigning into a single reader that tracks multiple in-flight ACKs
    (one dict of task_id→pending instead of one blocking reader per slot). Benefits:
    - Reduce from N PostgreSQL connections to 1
    - Eliminate SELECT...FOR UPDATE SKIP LOCKED contention between reader processes
    Risks to address before implementing:
    - Need explicit in-flight cap for backpressure (currently implicit via num_readers)
    - Single point of failure (one crash drops all in-flight tasks to lease expiry)
    - Signal handling must be redesigned (set_wakeup_fd only works on main thread)
    Hold off until PostgreSQL connection count becomes a real bottleneck.
    """

    def __init__(self, *, worker_id: str, pool: str, subpool: str | None, category: Category,
                 broker_socket: str, conn_params: dict | None = None, db_conn=None,
                 start: type[StartHandler], enrich=None,
                 lease_s: float = DEFAULT_LEASE_SECONDS,
                 poll_s: float = 1.0, reap_interval_s: float = REAP_INTERVAL_SECONDS,
                 host: str | None = None):
        """
        Args:
            worker_id: '<host>/<reader-n>'; the identity in pgdef.worker and in
                task leases
            pool, subpool, category: what this reader serves. A NULL subpool or
                deeper category level means it claims only generic tasks; a
                value means generic tasks plus tasks restricted to that value.
            broker_socket: path of the broker's Unix socket
            conn_params: PostgreSQL parameters (host, port, user, password;
                never a database name). Or pass an open `db_conn`.
            start: the application's start handler class (what the first
                message of a task is)
            enrich: optional callable(conn, task) -> dict, merged into the
                start message under `enrich`. How an application adds
                per-task data that is not in the payload.
        """
        self.worker_id = worker_id
        self.host = host or worker_id.split('/')[0]
        self.pool = pool
        self.subpool = subpool
        self.category = category
        self.broker_socket = broker_socket
        self.conn_params = conn_params
        self.start = start
        self.enrich = enrich
        self.lease_s = float(lease_s)
        self.poll_s = poll_s
        self.reap_interval_s = reap_interval_s
        self.sock = None
        self.db_conn = db_conn
        self.running = False
        self._last_reap = float('-inf')

        # Create wakeup pipe for signal handling
        self.wakeup_read_fd, self.wakeup_write_fd = os.pipe()
        os.set_blocking(self.wakeup_read_fd, False)
        os.set_blocking(self.wakeup_write_fd, False)

    # -- main loop ------------------------------------------------------------

    def run(self):
        """Main PG reader loop"""
        self._connect_to_broker()
        if self.db_conn is None:
            self._connect_to_database()

        logger.info(f"PG Reader {self.worker_id} started for pool={self.pool} subpool={self.subpool} "
                    f"category={self.category} (PID={os.getpid()})")
        self.running = True
        try:
            while self.running:
                try:
                    if not self.step():
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
        finally:
            self._cleanup()

    def step(self) -> bool:
        """One iteration: reap, claim, hand over, hold. False means stop."""
        self._maybe_reap()
        task = self.claim()
        if task is None:
            pgworkers.upsert(self.db_conn, worker_id=self.worker_id, host=self.host, pool=self.pool,
                             subpool=self.subpool, category=self.category)
            time.sleep(self.poll_s)
            return True
        return self.hold(task)

    def claim(self):
        """Claim one task, or None. A database error is logged and means None."""
        try:
            claimed = pgtasks.claim(self.db_conn, pool=self.pool, category=self.category,
                                    subpool=self.subpool, worker=self.worker_id,
                                    lease_s=self.lease_s, limit=1)
        except Exception as e:
            logger.error(f"PG Reader {self.worker_id} database error in claim: {e}", exc_info=True)
            return None
        if not claimed:
            return None
        task = claimed[0]
        logger.info(f"PG Reader {self.worker_id} claimed task_id={task.id} pool={task.pool} "
                    f"category={task.category} attempt={task.logistics.attempt}")
        pgworkers.upsert(self.db_conn, worker_id=self.worker_id, host=self.host, pool=self.pool,
                         subpool=self.subpool, category=self.category, current_task=task.id)
        return task

    def start_message(self, task) -> dict:
        """The first message of a task's graph. It carries the task's columns,
        `payload` and `appcookie`, and whatever `enrich` adds; the reader
        stamps nothing into the payload."""
        extra = self.enrich(self.db_conn, task) if self.enrich else {}
        return {
            'class': self.start.CLASS,
            'target_queue': self.start.QUEUE,
            'task_id': task.id,
            'pool': task.pool,
            'subpool': task.subpool,
            'category': list(task.category),
            'attempt': task.logistics.attempt,
            'payload': task.payload,
            'appcookie': task.appcookie,
            'enrich': extra or {},
        }

    def hold(self, task) -> bool:
        """Send the task to the broker and wait for its ack, renewing the lease."""
        try:
            message = self.start_message(task)
        except Exception as e:
            # enrich() failed: nothing was sent, so there is no ack to wait for.
            logger.error(f"start message for task_id={task.id} failed: {e}", exc_info=True)
            pgtasks.fail(self.db_conn, task.ref, f"enrich failed: {type(e).__name__}: {e}")
            return True

        send_message(self.sock, {'type': 'register_ack', 'task_id': task.id, 'worker_id': self.worker_id})
        send_message(self.sock, {'type': 'forward', 'message': message})
        logger.debug(f"Forwarded {message['class']} for task_id={task.id}")

        renew_every = max(self.lease_s / 3.0, 0.1)
        next_renew = time.monotonic() + renew_every
        holding = True      # False once the lease is known lost

        while self.running:
            state = self._wait_for_socket(timeout=max(next_renew - time.monotonic(), 0.0))
            if state == 'signal':
                if not self.running:
                    logger.info("Shutdown signal received during ack wait")
                    return False
                continue
            if state == 'timeout':
                next_renew = time.monotonic() + renew_every
                if holding:
                    holding = self._renew(task)
                continue

            response = recv_message(self.sock)
            if response is None:
                logger.error("Broker connection closed")
                return False
            if response['type'] == 'ack' and response['task_id'] == task.id:
                self._on_ack(task, response)
                return True
        return False

    def _on_ack(self, task, response: dict):
        if response.get('negative'):
            logger.warning(f"Received NAK for task_id={task.id} (task failed), continuing")
            if response.get('error') is not None:
                # The broker could not deliver a message of this task (a queue
                # without workers). It has no database; record the failure here.
                # Permanent: no retry will make the node's configuration right.
                pgtasks.fail(self.db_conn, task.ref, str(response['error']), permanent=True)
        else:
            logger.info(f"Received ACK for task_id={task.id} (task completed), continuing")

    def _renew(self, task) -> bool:
        try:
            ok = pgtasks.renew(self.db_conn, task.ref, worker=self.worker_id, lease_s=self.lease_s)
            pgworkers.upsert(self.db_conn, worker_id=self.worker_id, host=self.host, pool=self.pool,
                             subpool=self.subpool, category=self.category, current_task=task.id)
        except Exception as e:
            logger.error(f"lease renewal for task_id={task.id} failed: {e}")
            return True       # transient database trouble: try again next time
        if not ok:
            logger.warning(f"task_id={task.id} is no longer leased to {self.worker_id} "
                           f"(reaped, reset or cancelled); no longer renewing")
        return ok

    def _maybe_reap(self):
        now = time.monotonic()
        if now - self._last_reap < self.reap_interval_s:
            return
        self._last_reap = now
        try:
            reaped = pgtasks.reap_expired(self.db_conn, self.pool)
        except Exception as e:
            logger.error(f"reap_expired failed: {e}", exc_info=True)
            return
        if reaped:
            logger.warning(f"Reaped {len(reaped)} task(s) with expired leases in pool {self.pool}: {reaped}")

    # -- plumbing ---------------------------------------------------------------

    def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"PG Reader {self.worker_id} shutting down")
        self.running = False

    def _cleanup(self):
        logger.info(f"PG Reader {self.worker_id} cleanup starting (PID={os.getpid()})")
        if self.db_conn:
            try:
                pgworkers.remove(self.db_conn, self.worker_id)
            except Exception:
                pass
            self.db_conn.close()
        if self.sock:
            self.sock.close()
        logger.info(f"PG Reader {self.worker_id} cleanup complete (PID={os.getpid()})")

    def _wait_for_socket(self, timeout=None) -> str:
        """Wait for the broker socket, with signal interruption support.

        Returns 'ready', 'signal' (the wakeup pipe fired) or 'timeout'."""
        ready, _, _ = select.select([self.sock.fileno(), self.wakeup_read_fd], [], [], timeout)

        if self.wakeup_read_fd in ready:
            # Signal received, drain the wakeup pipe
            try:
                os.read(self.wakeup_read_fd, 1024)
            except BlockingIOError:
                pass
            return 'signal'

        return 'ready' if self.sock.fileno() in ready else 'timeout'

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
        options = "-c statement_timeout=1000"  # 1000ms = 1s
        self.db_conn = ReconnectableConn(
            self.conn_params,
            row_factory=dict_row,
            options=options,
            autocommit=True
        )
        logger.info(f"PG Reader {self.worker_id} connected to database")
