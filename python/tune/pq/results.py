#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuning Results Storage

Handles writing individual hsaco benchmark results to PostgreSQL.
"""

import psycopg
from psycopg.types.json import Jsonb


def save_tuning_result(task_id: str, report: dict, conn) -> None:
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
        conn: PostgreSQL connection (from psycopg.connect)

    Raises:
        psycopg.Error: Database errors
    """
    with conn.cursor() as cur:
        # Extract fields from report
        kernel_name = report['kernel_name']
        hsaco_index = report['hsaco_index']
        result = report['result']
        result_data = report.get('result_data')
        error = report.get('error')
        gpu_id = report.get('complete_on_gpu')

        # Insert result using Jsonb type
        cur.execute("""
            INSERT INTO tuning_results
                (task_id, kernel_name, hsaco_index, result, result_data, error, gpu_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            task_id,
            kernel_name,
            hsaco_index,
            result,
            Jsonb(result_data) if result_data else None,
            Jsonb(error) if error else None,
            gpu_id
        ))


def save_optune_result(task_id: str, report: dict, conn) -> None:
    """
    Save a single op-tuning backend result to optune_results.

    Args:
        task_id: Task ID from task_queue
        report: Benchmark report dictionary with keys:
            - op_name: Operator name
            - backend_index: Backend variant index
            - result: Result status (OK/NotOK/crash/ERROR)
            - result_data: Optional benchmark data (JSONB)
            - error: Optional error information (JSONB)
            - complete_on_gpu: GPU ID used for benchmark
        conn: PostgreSQL connection (from psycopg.connect)
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO optune_results
                (task_id, op_name, backend_index, result, result_data, error, gpu_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            task_id,
            report['op_name'],
            report['backend_index'],
            report['result'],
            Jsonb(report['result_data']) if report.get('result_data') else None,
            Jsonb(report['error']) if report.get('error') else None,
            report.get('complete_on_gpu'),
        ))


def get_task_results(task_id: str, conn) -> list:
    """
    Retrieve all results for a task.

    Args:
        task_id: Task ID
        conn: PostgreSQL connection (from psycopg.connect)

    Returns:
        List of result dictionaries
    """
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

