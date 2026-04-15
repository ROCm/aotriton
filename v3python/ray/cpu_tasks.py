# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
CPU Tasks for Ray-based execution

Handles database writes and postprocessing without consuming GPU resources.
"""

import ray
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List


# Global connection parameters (read once at module load)
CONN_PARAMS = {
    'host': os.environ.get('CELERY_SERVICE_HOST', 'localhost'),
    'port': int(os.environ.get('POSTGRES_PORT', 5432)),
    'user': os.environ.get('POSTGRES_USER', 'aotriton'),
    'password': os.environ.get('POSTGRES_PASSWORD')
}


@ray.remote
def db_writer_task(task_id: str, report: Dict[str, Any]) -> bool:
    """
    Database writer: write one report to PostgreSQL (CPU task).
    Offloads I/O from GPU workers so they can return to benchmarking.

    Args:
        task_id: Task ID for database association
        report: Benchmark report dictionary

    Returns:
        True on success

    Raises:
        Exception: Database errors
    """
    from v3python.pq import save_tuning_result

    try:
        save_tuning_result(task_id, report, CONN_PARAMS)
        kname = report['kernel_name']
        hsaco_idx = report['hsaco_index']
        result = report['result']
        print(f'[DB] Wrote {kname}[{hsaco_idx}] result={result}')
        return True

    except Exception as e:
        print(f'[DB] Failed to write report: {e}')
        raise


@ray.remote
def postprocess_task(task_id: str, reports: List[Dict[str, Any]], task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Postprocessing: aggregate, write aggregation to database, and cleanup (CPU task, NO GPU needed).

    Aggregates all benchmark reports into a summary, writes aggregation to database,
    and cleans up temporary data.

    Args:
        task_id: Task ID for database association
        reports: List of benchmark reports from GPU tasks
        task_config: Task configuration with tmpdir

    Returns:
        Aggregation dictionary with brief summary
    """
    from v3python.pq import complete_task

    # Aggregate results into brief summary
    brief = {}
    for r in reports:
        kname = r["kernel_name"]
        index = r["hsaco_index"]
        result = r["result"]

        if kname not in brief:
            brief[kname] = {}

        brief[kname][index] = result

    aggregation = {
        "task_config": task_config,
        "brief": brief,
    }

    # Write aggregation to database (update task_queue with completed status)
    try:
        arch = task_config.get('arch', 'unknown')
        complete_task(task_id, arch, CONN_PARAMS)
        print(f'[Postprocess] Marked task {task_id} as completed in database')
    except Exception as e:
        print(f'[Postprocess] Failed to mark task completed in database: {e}')
        # Don't raise - aggregation is already in memory

    # Cleanup tmpdir
    tmpdir = Path(task_config.get('tmpdir', ''))
    if tmpdir and tmpdir.exists():
        try:
            shutil.rmtree(tmpdir)
            print(f'[Postprocess] Cleaned up {tmpdir}')
        except Exception as e:
            print(f'[Postprocess] Failed to cleanup {tmpdir}: {e}')

    print(f'[Postprocess] Aggregated {len(reports)} reports')
    return aggregation
