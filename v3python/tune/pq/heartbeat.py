# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Worker heartbeat management for Tuner v3.5

Tracks worker health and activity for monitoring and dead worker detection.
"""

import psycopg
from typing import Dict, Any
import socket
import os


class HeartbeatManager:
    """Manages worker heartbeat updates"""

    def __init__(self, conn_params: Dict[str, Any], arch: str):
        """
        Initialize heartbeat manager.

        Args:
            conn_params: PostgreSQL connection parameters
            arch: GPU architecture this worker handles
        """
        self.conn_params = conn_params
        self.arch = arch
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}"
        self.node_hostname = socket.gethostname()

    def _get_connection(self):
        """Get database connection"""
        return psycopg.connect(**self.conn_params, autocommit=True)

    def update(self, status: str = 'active') -> None:
        """
        Update worker heartbeat.

        Args:
            status: Worker status ('active' or 'idle')
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO worker_heartbeat (worker_id, node_hostname, arch, last_heartbeat, status)
                    VALUES (%s, %s, %s, NOW(), %s)
                    ON CONFLICT (worker_id) DO UPDATE
                    SET last_heartbeat = NOW(),
                        status = EXCLUDED.status
                """, (self.worker_id, self.node_hostname, self.arch, status))

    def increment_completed(self) -> None:
        """Increment completed task counter"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE worker_heartbeat
                    SET tasks_completed = tasks_completed + 1,
                        last_heartbeat = NOW()
                    WHERE worker_id = %s
                """, (self.worker_id,))

    def increment_failed(self) -> None:
        """Increment failed task counter"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE worker_heartbeat
                    SET tasks_failed = tasks_failed + 1,
                        last_heartbeat = NOW()
                    WHERE worker_id = %s
                """, (self.worker_id,))

    def mark_dead(self) -> None:
        """Mark this worker as dead (graceful shutdown)"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE worker_heartbeat
                    SET status = 'dead',
                        last_heartbeat = NOW()
                    WHERE worker_id = %s
                """, (self.worker_id,))

    def cleanup_dead_workers(self, threshold_seconds: int = 300) -> int:
        """
        Mark workers as dead if no heartbeat within threshold.

        Args:
            threshold_seconds: Heartbeat timeout (default: 5 minutes)

        Returns:
            Number of workers marked as dead
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE worker_heartbeat
                    SET status = 'dead'
                    WHERE status != 'dead'
                      AND EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) > %s
                    RETURNING worker_id
                """, (threshold_seconds,))

                count = len(cur.fetchall())
                return count
