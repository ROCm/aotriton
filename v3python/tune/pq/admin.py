# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Administrative utilities for Tuner v3.5 queue

Provides schema initialization, partition management, and maintenance tasks.
"""

import psycopg
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class QueueAdmin:
    """Administrative operations for PostgreSQL queue"""

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize queue admin.

        Args:
            conn_params: PostgreSQL connection parameters
        """
        self.conn_params = conn_params

    def _get_connection(self):
        """Get database connection"""
        return psycopg.connect(**self.conn_params, autocommit=True)

    def init_schema(self, schema_file: Path = None) -> None:
        """
        Initialize database schema from SQL file.

        Args:
            schema_file: Path to schema.sql file (default: pq/schema.sql)
        """
        if schema_file is None:
            schema_file = Path(__file__).parent / 'schema.sql'

        logger.info(f"Initializing schema from {schema_file}")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                sql = schema_file.read_text()
                cur.execute(sql)

        logger.info("Schema initialized successfully")

    def create_partition(self, arch: str) -> None:
        """
        Create partition for an architecture.

        Args:
            arch: GPU architecture (e.g., 'gfx942')
        """
        logger.info(f"Creating partition for architecture: {arch}")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT create_arch_partition(%s)", (arch,))

        logger.info(f"Partition task_queue_{arch} created successfully")

    def create_partitions(self, architectures: List[str]) -> None:
        """
        Create partitions for multiple architectures.

        Args:
            architectures: List of GPU architectures
        """
        for arch in architectures:
            self.create_partition(arch)

    def list_partitions(self) -> List[str]:
        """
        List all existing task_queue partitions.

        Returns:
            List of partition names
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename LIKE 'task_queue_%'
                    ORDER BY tablename
                """)

                rows = cur.fetchall()
                return [row[0] for row in rows]

    def reset_stale_tasks(self, timeout_seconds: int = 7200) -> int:
        """
        Reset tasks that have been running too long.

        Args:
            timeout_seconds: Task timeout in seconds (default: 2 hours)

        Returns:
            Number of tasks reset
        """
        logger.info(f"Resetting tasks running longer than {timeout_seconds}s")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE task_queue
                    SET status = 'pending',
                        worker_id = NULL,
                        node_hostname = NULL,
                        started_at = NULL,
                        retry_count = retry_count + 1
                    WHERE status = 'running'
                      AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                    RETURNING id, arch
                """, (timeout_seconds,))

                rows = cur.fetchall()
                count = len(rows)

        if count > 0:
            logger.info(f"Reset {count} stale tasks")
        else:
            logger.info("No stale tasks found")

        return count

    def cleanup_dead_workers(self, threshold_seconds: int = 300) -> int:
        """
        Mark workers as dead if no heartbeat within threshold.

        Args:
            threshold_seconds: Heartbeat timeout (default: 5 minutes)

        Returns:
            Number of workers marked as dead
        """
        logger.info(f"Cleaning up workers with no heartbeat for {threshold_seconds}s")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE worker_heartbeat
                    SET status = 'dead'
                    WHERE status != 'dead'
                      AND EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) > %s
                    RETURNING worker_id
                """, (threshold_seconds,))

                rows = cur.fetchall()
                count = len(rows)

        if count > 0:
            logger.info(f"Marked {count} workers as dead")
        else:
            logger.info("No dead workers found")

        return count


    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall queue statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Overall task counts
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'running') as running,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed,
                        COUNT(*) as total
                    FROM task_queue
                """)
                task_stats = dict(cur.fetchone())

                # Worker counts
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'active') as active,
                        COUNT(*) FILTER (WHERE status = 'idle') as idle,
                        COUNT(*) FILTER (WHERE status = 'dead') as dead,
                        COUNT(*) as total
                    FROM worker_heartbeat
                """)
                worker_stats = dict(cur.fetchone())

                # Partition count
                cur.execute("""
                    SELECT COUNT(*)
                    FROM pg_tables
                    WHERE schemaname = 'public'
                      AND tablename LIKE 'task_queue_%'
                """)
                partition_count = cur.fetchone()[0]

        return {
            'tasks': task_stats,
            'workers': worker_stats,
            'partitions': partition_count
        }
