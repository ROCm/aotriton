# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuner v3.5 PostgreSQL Queue

PostgreSQL-based distributed task queue replacing Celery + RabbitMQ.
Provides lower network overhead, simpler infrastructure, and better visibility.
"""

from .queue import TaskQueue
from .dispatcher import TaskDispatcher
from .worker import Worker
from .heartbeat import HeartbeatManager
from .results import save_tuning_result, get_task_results

__all__ = ['TaskQueue', 'TaskDispatcher', 'Worker', 'HeartbeatManager', 'save_tuning_result', 'get_task_results']

__version__ = '3.5.0'
