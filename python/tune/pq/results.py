#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuning Results Storage

Handles writing individual per-impl-variant benchmark results to PostgreSQL.

Phase 2 (modularization unification, modular-tune.md §4.3/§4.7): the former
save_tuning_result (kernel level, tuning_results table) / save_optune_result
(op level, optune_results table) pair is unified into a single function and
a single table -- ImplSelector's iface_name/impl_index replace kernel_name/
hsaco_index and op_name/backend_index, and tuning_level is stored alongside
them (denormalized, no task_queue join) since iface_name collides across
levels (e.g. 'attn_fwd' is valid at both the kernel and op level).
"""

import psycopg
from psycopg.types.json import Jsonb


def save_tuning_result(task_id: str, report: dict, conn) -> None:
    """
    Save a single tuning result to the database.

    Args:
        task_id: Task ID from task_queue
        report: Benchmark report dictionary with keys:
            - tuning_level: 'kernel' | 'op'
            - iface_name: Interface name (e.g. 'attn_fwd')
            - impl_index: Variant index (HSACO index for kernel level,
              backend index for op level)
            - result: Result status (OK/NotOK/crash/ERROR)
            - result_data: Optional benchmark data (JSONB)
            - error: Optional error information (JSONB)
            - complete_on_gpu: GPU ID used for benchmark
        conn: PostgreSQL connection (from psycopg.connect)

    Raises:
        psycopg.Error: Database errors
    """
    with conn.cursor() as cur:
        # Extract fields from report
        tuning_level = report['tuning_level']
        iface_name = report['iface_name']
        impl_index = report['impl_index']
        result = report['result']
        result_data = report.get('result_data')
        error = report.get('error')
        gpu_id = report.get('complete_on_gpu')

        # Insert result using Jsonb type
        cur.execute("""
            INSERT INTO tuning_results
                (task_id, tuning_level, iface_name, impl_index, result, result_data, error, gpu_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            task_id,
            tuning_level,
            iface_name,
            impl_index,
            result,
            Jsonb(result_data) if result_data else None,
            Jsonb(error) if error else None,
            gpu_id
        ))


def get_task_results(task_id: str, conn, tuning_level: str | None = None) -> list:
    """
    Retrieve all results for a task.

    Args:
        task_id: Task ID
        conn: PostgreSQL connection (from psycopg.connect)
        tuning_level: Optional 'kernel' | 'op' filter (None = both levels).
            Only meaningful for tasks whose iface_name is ambiguous across
            levels; ordinarily a task_id already implies exactly one level
            via its task_queue row.

    Returns:
        List of result dictionaries
    """
    query = """
        SELECT
            id,
            tuning_level,
            iface_name,
            impl_index,
            result,
            result_data,
            error,
            gpu_id,
            created_at
        FROM tuning_results
        WHERE task_id = %s
    """
    params = [task_id]
    if tuning_level is not None:
        query += " AND tuning_level = %s"
        params.append(tuning_level)
    query += " ORDER BY iface_name, impl_index"

    with conn.cursor() as cur:
        cur.execute(query, params)

        results = []
        for row in cur.fetchall():
            results.append({
                'id': row[0],
                'tuning_level': row[1],
                'iface_name': row[2],
                'impl_index': row[3],
                'result': row[4],
                'result_data': row[5],
                'error': row[6],
                'gpu_id': row[7],
                'created_at': row[8].isoformat() if row[8] else None
            })

        return results
