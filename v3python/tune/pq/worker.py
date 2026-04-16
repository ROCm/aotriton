# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Worker implementation for Tuner v3.5

Polls PostgreSQL queue with exponential backoff, executes tasks using Ray,
and tracks worker health via heartbeats.
"""

import time
import signal
import logging
from typing import Dict, Any, Callable, Optional
from .queue import TaskQueue, Task
from .heartbeat import HeartbeatManager

logger = logging.getLogger(__name__)


class Worker:
    """PostgreSQL queue worker with exponential backoff polling"""

    def __init__(
        self,
        conn_params: Dict[str, Any],
        arch: str,
        executor: Callable[[Task], Any],
        batch_size: int = 10,
        poll_interval: float = 1.0,
        max_poll_interval: float = 30.0,
        backoff_multiplier: float = 1.5
    ):
        """
        Initialize worker.

        Args:
            conn_params: PostgreSQL connection parameters
            arch: GPU architecture this worker handles
            executor: Callable that executes a task and returns result
            batch_size: Number of tasks to fetch per poll
            poll_interval: Initial poll interval in seconds
            max_poll_interval: Maximum poll interval (backoff cap)
            backoff_multiplier: Exponential backoff multiplier
        """
        self.queue = TaskQueue(conn_params)
        self.heartbeat = HeartbeatManager(conn_params, arch)
        self.arch = arch
        self.executor = executor
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        self.backoff_multiplier = backoff_multiplier

        self.running = False
        self.current_interval = poll_interval

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def start(self) -> None:
        """Start worker loop"""
        self.running = True
        logger.info(f"Worker started for architecture: {self.arch}")

        # Initial heartbeat
        self.heartbeat.update('active')

        while self.running:
            try:
                tasks = self.queue.fetch_tasks(self.arch, self.batch_size)

                if not tasks:
                    # No tasks: exponential backoff
                    self.current_interval = min(
                        self.current_interval * self.backoff_multiplier,
                        self.max_poll_interval
                    )
                    self.heartbeat.update('idle')
                    logger.debug(f"No tasks, backing off to {self.current_interval:.1f}s")
                    time.sleep(self.current_interval)
                    continue

                # Got tasks: reset to aggressive polling
                self.current_interval = self.poll_interval
                self.heartbeat.update('active')
                logger.info(f"Fetched {len(tasks)} tasks")

                # Execute tasks
                for task in tasks:
                    if not self.running:
                        # Shutdown requested, mark tasks as pending again
                        for remaining_task in tasks:
                            self.queue.retry_task(remaining_task.id, remaining_task.arch)
                        break

                    try:
                        logger.info(f"Executing task {task.id} (module={task.module})")
                        result = self.executor(task)
                        self.queue.mark_completed(task.id, task.arch)
                        self.heartbeat.increment_completed()
                        logger.info(f"Task {task.id} completed")

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        logger.error(f"Task {task.id} failed: {error_msg}")
                        self.queue.mark_failed(task.id, task.arch, error_msg)
                        self.heartbeat.increment_failed()

                # Update heartbeat after processing batch
                self.heartbeat.update('active')

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                time.sleep(self.poll_interval)

        # Graceful shutdown
        self.heartbeat.mark_dead()
        logger.info("Worker stopped")

    def stop(self) -> None:
        """Stop worker loop"""
        self.running = False

    def run_once(self) -> int:
        """
        Execute one iteration of the worker loop (for testing).

        Returns:
            Number of tasks processed
        """
        tasks = self.queue.fetch_tasks(self.arch, self.batch_size)

        for task in tasks:
            try:
                self.executor(task)
                self.queue.mark_completed(task.id, task.arch)
                self.heartbeat.increment_completed()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.queue.mark_failed(task.id, task.arch, error_msg)
                self.heartbeat.increment_failed()

        self.heartbeat.update('active' if tasks else 'idle')
        return len(tasks)
