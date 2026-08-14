# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuner v3.5 PostgreSQL Queue

PostgreSQL-based distributed task queue replacing Celery + RabbitMQ.
Provides lower network overhead, simpler infrastructure, and better visibility.
"""

from .queue import TaskQueue
from .dispatcher import TaskDispatcher
from .heartbeat import HeartbeatManager
from .results import save_tuning_result, get_task_results

# NOTE: `Worker` (a PostgreSQL-polling worker loop) used to be exported here
# from `.worker`, but it was dead code: `Worker.start()`/`run_once()` called
# `fetch_tasks(self.arch, self.batch_size)` with no `tuning_mode` (which
# `fetch_tasks` now requires, keyword-only, per modular-tune.md F16) and
# `mark_failed(task.id, task.arch, error_msg)` positionally against a
# keyword-only signature -- both would already raise `TypeError` if ever
# executed, and nothing in this repo constructs `Worker` (see
# `python/tune/localq/` for the actual worker implementation, built on
# `TaskQueue` directly). Deleted rather than fixed since it has no callers.
__all__ = ['TaskQueue', 'TaskDispatcher', 'HeartbeatManager', 'save_tuning_result', 'get_task_results']

__version__ = '3.5.0'
