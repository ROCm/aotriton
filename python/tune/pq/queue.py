# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Core queue operations for Tuner v3.5

Provides atomic task fetching, status updates, and queue management
using PostgreSQL SELECT FOR UPDATE SKIP LOCKED.
"""

import psycopg
from psycopg.rows import dict_row
from dataclasses import dataclass
from datetime import datetime
import socket
import os
import logging

logger = logging.getLogger(__name__)


class TuningLevelMismatch(RuntimeError):
    """fetch_tasks() claimed a task belonging to the other tuning level.

    Only reachable if the UPDATE's `tuning_level` predicate is broken, since
    PostgreSQL would otherwise not return the row. Systemic rather than
    transient: the filter is the sole thing keeping a worker off tasks its
    pyaotriton build cannot execute, so every later claim is suspect too.
    """


@dataclass
class Task:
    """Task representation"""
    id: int
    arch: str
    module: str
    tuning_level: str
    task_config: dict
    status: str
    priority: int = 5
    worker_id: str | None = None
    node_hostname: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0


class TaskQueue:
    """PostgreSQL-based task queue with architecture partitioning"""

    def __init__(self, conn):
        """
        Initialize task queue.

        Args:
            conn: PostgreSQL connection (from psycopg.connect)
        """
        self.conn = conn
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}"
        self.node_hostname = socket.gethostname()

    def fetch_tasks(self, arch: str, batch_size: int = 10, *, tuning_mode: str) -> list[Task]:
        """
        Fetch pending tasks for a specific architecture.

        Uses SELECT FOR UPDATE SKIP LOCKED for atomic task claiming.
        Queries the architecture-specific partition directly for performance.

        Args:
            arch: GPU architecture (e.g., 'gfx942', 'gfx90a')
            batch_size: Number of tasks to fetch
            tuning_mode: 'kernel' or 'op' (REQUIRED, keyword-only, no default --
                a kernel worker must never claim an op task and vice versa,
                see modular-tune.md F16). Filters on the denormalized
                task_queue.tuning_level column, not a `module` string pattern.

        Returns:
            List of claimed Task objects
        """
        partition_table = f"task_queue_{arch}"

        with self.conn.cursor(row_factory=dict_row) as cur:
            # Atomic task claiming using UPDATE ... RETURNING
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'running',
                    worker_id = %s,
                    node_hostname = %s,
                    started_at = NOW()
                WHERE id IN (
                    SELECT id FROM {partition_table}
                    WHERE status = 'pending'
                      AND tuning_level = %s
                    ORDER BY priority DESC, id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, arch, module, tuning_level, task_config, status, priority,
                          worker_id, node_hostname, created_at, started_at,
                          completed_at, error, retry_count
            """, (self.worker_id, self.node_hostname, tuning_mode, batch_size))

            try:
                rows = cur.fetchall()
            except psycopg.errors.QueryCanceled:
                logger.warning(f"TaskQueue.fetch_tasks: statement_timeout hit for {partition_table}")
                return []

            tasks = [Task(**row) for row in rows]

            # The UPDATE above filters on tuning_level, so a row of the wrong
            # level means that predicate is broken. Release the whole batch --
            # connections here are autocommit, so the claim is already durable
            # and raising without this would strand every row in 'running' --
            # then fail. The batch is released entirely, not just the offending
            # rows: this raises out of the worker, so correctly-claimed tasks
            # would be stranded too.
            wrong = [t for t in tasks if t.tuning_level != tuning_mode]
            if wrong:
                cur.execute(f"""
                    UPDATE {partition_table}
                       SET status = 'pending', worker_id = NULL,
                           node_hostname = NULL, started_at = NULL
                     WHERE id = ANY(%s)
                """, ([t.id for t in tasks],))
                raise TuningLevelMismatch(
                    f"fetch_tasks({arch!r}, tuning_mode={tuning_mode!r}) claimed "
                    f"task_ids={[t.id for t in wrong]} with tuning_level="
                    f"{sorted({t.tuning_level for t in wrong})}; the tuning_level "
                    f"filter is not doing its job. Released "
                    f"{len(tasks)} claim(s) back to pending.")

            if tasks:
                task_ids = [t.id for t in tasks]
                logger.info(f"TaskQueue.fetch_tasks: Claimed {len(tasks)} task(s) from {partition_table}: "
                           f"task_ids={task_ids}, status=pending→running, worker_id={self.worker_id}")

            return tasks

    def mark_completed(self, task_id: int, arch: str) -> None:
        """
        Mark task as completed.

        Args:
            task_id: Task ID
            arch: GPU architecture (for partition routing)
        """
        partition_table = f"task_queue_{arch}"

        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'completed',
                    completed_at = NOW()
                WHERE id = %s
            """, (task_id,))

            logger.info(f"TaskQueue.mark_completed: task_id={task_id}, arch={arch}, "
                       f"status=→completed, partition={partition_table}")

    def mark_failed(self, task_id: int, *, arch: str | None = None, error_message: str) -> None:
        """
        Mark task as failed with error message.

        Args:
            task_id: Task ID
            arch: GPU architecture (for partition routing, optional, keyword-only)
            error_message: Error message (keyword-only)
        """
        with self.conn.cursor() as cur:
            if arch:
                partition_table = f"task_queue_{arch}"
                cur.execute(f"""
                    UPDATE {partition_table}
                    SET status = 'failed',
                        completed_at = NOW(),
                        error = %s
                    WHERE id = %s
                """, (error_message, task_id))
                logger.error(f"TaskQueue.mark_failed: task_id={task_id}, arch={arch}, "
                            f"status=→failed, partition={partition_table}, error={error_message}")
            else:
                # Update parent table when arch unknown
                cur.execute("""
                    UPDATE task_queue
                    SET status = 'failed',
                        completed_at = NOW(),
                        error = %s
                    WHERE id = %s
                """, (error_message, task_id))
                logger.error(f"TaskQueue.mark_failed: task_id={task_id}, arch=unknown, "
                            f"status=→failed, partition=task_queue (parent), error={error_message}")

    def mark_pending(self, task_id: int, arch: str) -> None:
        """
        Mark task as pending (used during graceful shutdown to cancel running tasks).

        Status only -- existing results are left alone. See reset_to_pending()
        for the bulk re-run path, which can also discard them.

        Args:
            task_id: Task ID
            arch: GPU architecture (for partition routing)
        """
        partition_table = f"task_queue_{arch}"

        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'pending',
                    worker_id = NULL,
                    node_hostname = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    error = NULL
                WHERE id = %s
            """, (task_id,))

            logger.info(f"TaskQueue.mark_pending: task_id={task_id}, arch={arch}, "
                       f"status=→pending, partition={partition_table}")

    def retry_task(self, task_id: int, arch: str, max_retries: int = 3) -> bool:
        """
        Retry a failed task if under retry limit.

        Args:
            task_id: Task ID
            arch: GPU architecture
            max_retries: Maximum retry attempts

        Returns:
            True if task was retried, False if max retries exceeded
        """
        partition_table = f"task_queue_{arch}"

        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'pending',
                    retry_count = retry_count + 1,
                    worker_id = NULL,
                    node_hostname = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    error = NULL
                WHERE id = %s
                  AND retry_count < %s
                RETURNING id
            """, (task_id, max_retries))

            result = cur.fetchone()
            return result is not None

    def get_queue_stats(self, arch: str | None = None) -> dict[str, int]:
        """
        Get queue statistics.

        Args:
            arch: Optional architecture filter (None = all architectures)

        Returns:
            Dictionary with pending, running, completed, failed, cancelled counts
        """
        with self.conn.cursor() as cur:
            if arch:
                partition_table = f"task_queue_{arch}"
                cur.execute(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'running') as running,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed
                    FROM {partition_table}
                """)
            else:
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'running') as running,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed
                    FROM task_queue
                """)

            row = cur.fetchone()
            return dict(row) if row else {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}

    def detect_stale_tasks(self, timeout_seconds: int = 7200) -> list[Task]:
        """
        Detect tasks running longer than timeout.

        Args:
            timeout_seconds: Task timeout in seconds (default: 2 hours)

        Returns:
            List of stale tasks
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT id, arch, module, tuning_level, task_config, status, priority,
                       worker_id, node_hostname, created_at, started_at,
                       completed_at, error, retry_count
                FROM task_queue
                WHERE status = 'running'
                  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                ORDER BY started_at ASC
            """, (timeout_seconds,))

            rows = cur.fetchall()
            return [Task(**row) for row in rows]

    def reset_stale_tasks(self, timeout_seconds: int = 7200) -> int:
        """
        Reset stale tasks back to pending status.

        Args:
            timeout_seconds: Task timeout in seconds

        Returns:
            Number of tasks reset
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE task_queue
                SET status = 'pending',
                    worker_id = NULL,
                    node_hostname = NULL,
                    started_at = NULL,
                    retry_count = retry_count + 1
                WHERE status = 'running'
                  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                RETURNING id
            """, (timeout_seconds,))

            count = len(cur.fetchall())
            return count

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    _PROGRESS_VIEW = {'kernel': 'kernel_queue_progress', 'op': 'op_queue_progress'}

    def get_progress(self, tuning_level: str, *,
                     recent_window: str = '5 minutes',
                     stale_seconds: int = 7200) -> dict:
        """Per-arch queue progress for one tuning level.

        Returns {'progress': [...], 'speed': [...], 'stale': [...]}: the
        level's queue-progress view, recent completion counts, and long-running
        task counts. Callers merge and format these; the SQL and the
        tuning_level predicates live here so a queue-schema change does not
        have to be mirrored into the web UI (see pq/README.md).
        """
        try:
            view = self._PROGRESS_VIEW[tuning_level]
        except KeyError:
            raise ValueError(f"unknown tuning_level {tuning_level!r}; "
                             f"expected one of {sorted(self._PROGRESS_VIEW)}")
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f'SELECT * FROM {view} ORDER BY arch')
            progress = cur.fetchall()

            cur.execute("""
                SELECT arch, COUNT(*) AS recent_completions
                FROM task_queue
                WHERE status = 'completed'
                  AND completed_at > NOW() - %s::interval
                  AND tuning_level = %s
                GROUP BY arch
            """, (recent_window, tuning_level))
            speed = cur.fetchall()

            cur.execute("""
                SELECT arch, COUNT(*) AS stale_count
                FROM task_queue
                WHERE status = 'running'
                  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                  AND tuning_level = %s
                GROUP BY arch
            """, (stale_seconds, tuning_level))
            stale = cur.fetchall()
        return {'progress': progress, 'speed': speed, 'stale': stale}

    def reset_to_pending(self, row_ids: list[int], tuning_level: str, *,
                         delete_results: bool) -> int:
        """Reset the given task_queue rows to pending, for re-running.

        delete_results is keyword-only and REQUIRED because it is destructive
        and the destruction is not implied by the method name. With it set,
        the tasks' tuning_results and most_accurate_tuning_results rows are
        dropped as well -- GPU-hours of measurements -- so that a re-run
        starts clean instead of mixing new results with stale ones. Pass False
        to requeue while keeping the existing rows.

        Compare mark_pending(), which only moves a single task back to pending
        and never touches results.

        Every statement is scoped by tuning_level as well as id: callers select
        ids by arch/entry, which both levels share, so an id list can span
        levels. Returns the number of task_queue rows actually reset.
        """
        if not row_ids:
            return 0
        with self.conn.cursor() as cur:
            if delete_results:
                cur.execute(
                    'DELETE FROM most_accurate_tuning_results '
                    'WHERE tuning_level = %s AND task_id = ANY(%s)',
                    (tuning_level, row_ids))
                cur.execute(
                    'DELETE FROM tuning_results '
                    'WHERE tuning_level = %s AND task_id = ANY(%s)',
                    (tuning_level, row_ids))
            cur.execute("""
                UPDATE task_queue
                   SET status       = 'pending',
                       worker_id    = NULL,
                       node_hostname= NULL,
                       started_at   = NULL,
                       completed_at = NULL,
                       error        = NULL
                 WHERE tuning_level = %s AND id = ANY(%s)
            """, (tuning_level, row_ids))
            return cur.rowcount
