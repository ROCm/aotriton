#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuning results: one pgdef.report row per tuned impl variant.

The raw rows live in the engine's generic report table (so that a generic reset
can purge them without tune code); tune reads them back through the
tune.tuning_results view. ImplSelector's iface_name/impl_index are stored in the
report's rkey together with the tuning level (key `cat_level1`), since
iface_name collides across the two levels (e.g. 'attn_fwd' is valid at both the kernel and op level).
"""

from psycopg.rows import dict_row

from aotriton.pgdef.pq import reports as pgreports
from aotriton.pgdef.pq.tasks import TaskRef

from .tasks import APP, check_level


def save_tuning_result(conn, task_ref: TaskRef, report: dict, *, attempt: int) -> int:
    """
    Save a single tuning result to the database, as a pgdef.report row.

    Args:
        conn: PostgreSQL connection
        task_ref: (task id, pool) of the task the result belongs to
        report: Benchmark report dictionary with keys:
            - tuning_level: 'kernel' | 'op'
            - iface_name: Interface name (e.g. 'attn_fwd')
            - impl_index: Variant index (HSACO index for kernel level,
              backend index for op level)
            - result: Result status (OK/NotOK/crash/ERROR)
            - result_data: Optional benchmark data (JSONB)
            - error: Optional error information (JSONB)
            - complete_on_gpu: GPU ID used for benchmark
        attempt: the task's attempt number (logistics.attempt)

    A retried attempt re-benchmarks everything, so a result replaces any
    earlier report of the same (tuning level, iface, impl) of this task: each
    impl is reported once.

    Returns the report id.
    """
    gpu = report.get('complete_on_gpu')
    return pgreports.insert(
        conn, task_ref, cat_level0=APP, attempt=attempt,
        rkey={'cat_level1': check_level(report['tuning_level']),
              'iface': report['iface_name'], 'impl': report['impl_index']},
        status=report['result'],
        payload=report.get('result_data') or None,
        error=report.get('error') or None,
        worker=None if gpu is None else f'gpu-{gpu}',
        replace=True)


def get_task_results(conn, task_id: int, tuning_level: str | None = None) -> list:
    """
    Retrieve all results for a task, from the tune.tuning_results view.

    Args:
        conn: PostgreSQL connection
        task_id: Task ID
        tuning_level: Optional 'kernel' | 'op' filter (None = both levels).
            Only meaningful for tasks whose iface_name is ambiguous across
            levels; ordinarily a task_id already implies exactly one level.

    Returns:
        List of result dictionaries
    """
    query = """
        SELECT id, cat_level1, iface_name, impl_index, result, result_data,
               error, worker, created_at
          FROM tune.tuning_results
         WHERE task_id = %s
    """
    params = [task_id]
    if tuning_level is not None:
        query += " AND cat_level1 = %s"
        params.append(tuning_level)
    query += " ORDER BY iface_name, impl_index"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return [{**row, 'created_at': row['created_at'].isoformat() if row['created_at'] else None}
                for row in cur.fetchall()]


def get_task_debug_snapshot(conn, task_id: int) -> dict:
    """Every row related to one task_id, for the web UI's Debug page.

    Returns keys: task, tuning_results, best_results, accurate_results,
    optune_results, best_optune_results. The last two keep those names because
    the templates use them as labels; both read the unified tables filtered to
    tuning level 'op' (column cat_level1).

    A task_id already implies exactly one level via its task row, so splitting
    kernel from op here is presentation -- two sections on the page -- not a
    correctness requirement.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM pgdef.task_status WHERE id = %s', (task_id,))
        task = cur.fetchone()
        if task is not None:
            cur.execute('SELECT payload, appcookie, logistics FROM pgdef.task WHERE id = %s', (task_id,))
            task.update(cur.fetchone())

        cur.execute(
            'SELECT id, task_id, cat_level1, iface_name, impl_index, result,'
            ' result_data, error, worker, created_at FROM tune.tuning_results'
            " WHERE task_id = %s AND cat_level1 = 'kernel'"
            ' ORDER BY iface_name, impl_index', (task_id,))
        tuning_results = cur.fetchall()

        cur.execute(
            'SELECT * FROM tune.best_tuning_results WHERE task_id = %s'
            " AND cat_level1 = 'kernel' ORDER BY iface_name", (task_id,))
        best_results = cur.fetchall()

        cur.execute(
            'SELECT iface_name, test_case, tensor_name,'
            ' target_fudge_factor, absolute_error'
            ' FROM tune.most_accurate_tuning_results WHERE task_id = %s'
            ' ORDER BY iface_name, test_case, tensor_name', (task_id,))
        accurate_results = cur.fetchall()

        cur.execute(
            'SELECT id, cat_level1, iface_name, impl_index, result, result_data,'
            ' error, worker, created_at FROM tune.tuning_results'
            " WHERE task_id = %s AND cat_level1 = 'op'"
            ' ORDER BY iface_name, impl_index', (task_id,))
        optune_results = cur.fetchall()

        cur.execute(
            'SELECT iface_name, impl_index, median_time, arch, impl_desc, computed_at'
            ' FROM tune.best_tuning_results WHERE task_id = %s'
            " AND cat_level1 = 'op' ORDER BY iface_name", (task_id,))
        best_optune_results = cur.fetchall()

    return {
        'task': task,
        'tuning_results': tuning_results,
        'best_results': best_results,
        'accurate_results': accurate_results,
        'optune_results': optune_results,
        'best_optune_results': best_optune_results,
    }
