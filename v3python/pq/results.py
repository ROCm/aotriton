#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuning Results Storage

Handles writing individual hsaco benchmark results to PostgreSQL.
"""

import psycopg
import json
from typing import Dict, Any


def save_tuning_result(task_id: str, report: Dict[str, Any], conn_params: Dict[str, Any]) -> None:
    """
    Save a single tuning result to the database.

    Args:
        task_id: Task ID from task_queue
        report: Benchmark report dictionary with keys:
            - kernel_name: Kernel name
            - hsaco_index: HSACO variant index
            - result: Result status (OK/NotOK/crash/ERROR)
            - result_data: Optional benchmark data (JSONB)
            - error: Optional error information (JSONB)
            - complete_on_gpu: GPU ID used for benchmark
        conn_params: PostgreSQL connection parameters

    Raises:
        psycopg.Error: Database errors
    """
    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            # Extract fields from report
            kernel_name = report['kernel_name']
            hsaco_index = report['hsaco_index']
            result = report['result']
            result_data = report.get('result_data')
            error = report.get('error')
            gpu_id = report.get('complete_on_gpu')

            # Convert to JSON if needed
            result_data_json = json.dumps(result_data) if result_data else None
            error_json = json.dumps(error) if error else None

            # Insert result
            cur.execute("""
                INSERT INTO tuning_results
                    (task_id, kernel_name, hsaco_index, result, result_data, error, gpu_id)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)
            """, (
                task_id,
                kernel_name,
                hsaco_index,
                result,
                result_data_json,
                error_json,
                gpu_id
            ))


def get_task_results(task_id: str, conn_params: Dict[str, Any]) -> list:
    """
    Retrieve all results for a task.

    Args:
        task_id: Task ID
        conn_params: PostgreSQL connection parameters

    Returns:
        List of result dictionaries
    """
    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    kernel_name,
                    hsaco_index,
                    result,
                    result_data,
                    error,
                    gpu_id,
                    created_at
                FROM tuning_results
                WHERE task_id = %s
                ORDER BY kernel_name, hsaco_index
            """, (task_id,))

            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'kernel_name': row[1],
                    'hsaco_index': row[2],
                    'result': row[3],
                    'result_data': row[4],
                    'error': row[5],
                    'gpu_id': row[6],
                    'created_at': row[7].isoformat() if row[7] else None
                })

            return results


def complete_task(task_id: str, arch: str, conn_params: Dict[str, Any]) -> None:
    """
    Mark a task as completed in the task_queue.

    Args:
        task_id: Task ID
        arch: GPU architecture (for partition routing)
        conn_params: PostgreSQL connection parameters

    Raises:
        psycopg.Error: Database errors
    """
    with psycopg.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            partition_table = f"task_queue_{arch}"
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'completed',
                    completed_at = NOW()
                WHERE id = %s
            """, (task_id,))
