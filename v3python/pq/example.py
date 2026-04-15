#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Example usage of Tuner v3.5 PostgreSQL queue

Demonstrates initialization, task dispatch, and worker execution.
"""

import os
import time
import logging
from pq import TaskQueue, TaskDispatcher, Worker, HeartbeatManager
from pq.admin import QueueAdmin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def get_conn_params():
    """Get PostgreSQL connection parameters from environment"""
    return {
        'host': os.environ.get('CELERY_SERVICE_HOST', 'localhost'),
        'port': int(os.environ.get('POSTGRES_PORT', 5432)),
        'user': os.environ.get('POSTGRES_USER', 'aotriton'),
        'password': os.environ.get('POSTGRES_PASSWORD'),
        'dbname': os.environ.get('POSTGRES_DB', 'aotriton')
    }


def example_task_executor(task):
    """
    Example task executor function.

    In production, this would call the actual tuning DAG execution.
    """
    logger.info(f"Executing task {task.id}: {task.module}")
    logger.info(f"Config: {task.task_config}")

    # Simulate work
    time.sleep(0.5)

    return {'status': 'success', 'result': 'example'}


def example_init_schema():
    """Initialize database schema"""
    conn_params = get_conn_params()
    admin = QueueAdmin(conn_params)

    logger.info("Initializing schema...")
    admin.init_schema()

    logger.info("Creating partitions for common architectures...")
    admin.create_partitions(['gfx942', 'gfx90a', 'gfx1100'])

    logger.info("Schema initialization complete")


def example_dispatch_tasks():
    """Dispatch example tasks"""
    conn_params = get_conn_params()
    dispatcher = TaskDispatcher(conn_params)

    # Ensure partition exists
    dispatcher.ensure_partition('gfx942')

    # Create example tasks
    tasks = []
    for i in range(10):
        tasks.append({
            'arch': 'gfx942',
            'module': 'attn_fwd',
            'task_config': {
                'BATCH': 4,
                'H': 32,
                'N_CTX': 1024 + i * 256,
                'D_HEAD': 64,
            },
            'priority': 5
        })

    logger.info(f"Dispatching {len(tasks)} tasks...")
    count = dispatcher.dispatch_bulk(tasks)
    logger.info(f"Dispatched {count} tasks")


def example_run_worker():
    """Run a worker"""
    conn_params = get_conn_params()

    logger.info("Starting worker for gfx942...")
    worker = Worker(
        conn_params=conn_params,
        arch='gfx942',
        executor=example_task_executor,
        batch_size=5,
        poll_interval=1.0,
        max_poll_interval=10.0
    )

    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Stopping worker...")
        worker.stop()


def example_get_stats():
    """Get queue statistics"""
    conn_params = get_conn_params()
    admin = QueueAdmin(conn_params)

    stats = admin.get_statistics()
    logger.info("Queue Statistics:")
    logger.info(f"  Tasks: {stats['tasks']}")
    logger.info(f"  Workers: {stats['workers']}")
    logger.info(f"  Partitions: {stats['partitions']}")

    queue = TaskQueue(conn_params)
    arch_stats = queue.get_queue_stats('gfx942')
    logger.info(f"  gfx942: {arch_stats}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python example.py <command>")
        print("Commands:")
        print("  init       - Initialize schema and partitions")
        print("  dispatch   - Dispatch example tasks")
        print("  worker     - Run a worker")
        print("  stats      - Show queue statistics")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'init':
        example_init_schema()
    elif command == 'dispatch':
        example_dispatch_tasks()
    elif command == 'worker':
        example_run_worker()
    elif command == 'stats':
        example_get_stats()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
