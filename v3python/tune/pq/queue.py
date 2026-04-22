# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Core queue operations for Tuner v3.5

Provides atomic task fetching, status updates, and queue management
using PostgreSQL SELECT FOR UPDATE SKIP LOCKED.
"""

import psycopg
from psycopg.rows import dict_row
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import socket
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task representation"""
    id: int
    arch: str
    module: str
    task_config: Dict[str, Any]
    status: str
    priority: int = 5
    worker_id: Optional[str] = None
    node_hostname: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
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

    def fetch_tasks(self, arch: str, batch_size: int = 10) -> List[Task]:
        """
        Fetch pending tasks for a specific architecture.

        Uses SELECT FOR UPDATE SKIP LOCKED for atomic task claiming.
        Queries the architecture-specific partition directly for performance.

        Args:
            arch: GPU architecture (e.g., 'gfx942', 'gfx90a')
            batch_size: Number of tasks to fetch

        Returns:
            List of claimed Task objects
        """
        partition_table = f"task_queue_{arch}"

        with self.conn.cursor() as cur:
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
                    ORDER BY priority DESC, id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, arch, module, task_config, status, priority,
                          worker_id, node_hostname, created_at, started_at,
                          completed_at, error, retry_count
            """, (self.worker_id, self.node_hostname, batch_size))

            rows = cur.fetchall()

            tasks = [Task(**row) for row in rows]

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

    def mark_failed(self, task_id: int, *, arch: str = None, error_message: str) -> None:
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

    def get_queue_stats(self, arch: Optional[str] = None) -> Dict[str, int]:
        """
        Get queue statistics.

        Args:
            arch: Optional architecture filter (None = all architectures)

        Returns:
            Dictionary with pending, running, completed, failed counts
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

    def detect_stale_tasks(self, timeout_seconds: int = 7200) -> List[Task]:
        """
        Detect tasks running longer than timeout.

        Args:
            timeout_seconds: Task timeout in seconds (default: 2 hours)

        Returns:
            List of stale tasks
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, arch, module, task_config, status, priority,
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

