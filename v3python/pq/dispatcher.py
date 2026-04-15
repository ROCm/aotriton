# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Task dispatcher for Tuner v3.5

Bulk INSERT tasks into PostgreSQL queue, replacing Celery task dispatch.
"""

import psycopg
from psycopg.types.json import Jsonb
from typing import Dict, Any, List, Iterable
from dataclasses import asdict


class TaskDispatcher:
    """Dispatches tuning tasks to PostgreSQL queue"""

    def __init__(self, conn_params: Dict[str, Any]):
        """
        Initialize task dispatcher.

        Args:
            conn_params: PostgreSQL connection parameters
        """
        self.conn_params = conn_params

    def _get_connection(self):
        """Get database connection"""
        return psycopg.connect(**self.conn_params)

    def dispatch_bulk(
        self,
        tasks: Iterable[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Dispatch tasks in bulk using efficient batch INSERT.

        Args:
            tasks: Iterable of task configurations, each with keys:
                   - arch: GPU architecture (str)
                   - module: Module name (str)
                   - task_config: Task configuration (dict)
                   - priority: Optional priority (int, default: 5)
            batch_size: Number of tasks per INSERT statement

        Returns:
            Total number of tasks dispatched
        """
        total_dispatched = 0
        batch = []

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for task in tasks:
                    batch.append({
                        'arch': task['arch'],
                        'module': task['module'],
                        'task_config': Jsonb(task['task_config']),
                        'priority': task.get('priority', 5)
                    })

                    if len(batch) >= batch_size:
                        self._insert_batch(cur, batch)
                        total_dispatched += len(batch)
                        batch = []

                # Insert remaining tasks
                if batch:
                    self._insert_batch(cur, batch)
                    total_dispatched += len(batch)

                conn.commit()

        return total_dispatched

    def _insert_batch(self, cur, batch: List[Dict[str, Any]]) -> None:
        """
        Insert a batch of tasks using executemany.

        Args:
            cur: Database cursor
            batch: List of task dictionaries
        """
        cur.executemany("""
            INSERT INTO task_queue (arch, module, task_config, priority)
            VALUES (%(arch)s, %(module)s, %(task_config)s, %(priority)s)
        """, batch)

    def dispatch_single(self, arch: str, module: str, task_config: Dict[str, Any], priority: int = 5) -> int:
        """
        Dispatch a single task.

        Args:
            arch: GPU architecture
            module: Module name
            task_config: Task configuration
            priority: Task priority (higher = more urgent)

        Returns:
            Task ID
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO task_queue (arch, module, task_config, priority)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (arch, module, Jsonb(task_config), priority))

                task_id = cur.fetchone()[0]
                conn.commit()
                return task_id

    def ensure_partition(self, arch: str) -> None:
        """
        Ensure partition exists for architecture.

        Args:
            arch: GPU architecture
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT create_arch_partition(%s)", (arch,))
                conn.commit()
