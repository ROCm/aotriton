#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared PostgreSQL helpers for retry_missing_entries and reset_broken_to_pending.

Functions here operate on an open psycopg connection (dict_row factory assumed).
Connection management is the caller's responsibility.
"""

from collections import Counter
from dataclasses import asdict

from v3python.tune.flash.module import FlashEntry


def _entry_to_jsonb_filter(entry: FlashEntry) -> tuple[str, list]:
    """
    Build a WHERE clause fragment matching task_config->'entry' fields.
    Returns (sql_fragment, params).
    """
    d = asdict(entry)
    clauses = []
    params: list = []
    for field, value in d.items():
        col = f"task_config->'entry'->>'{field}'"
        if isinstance(value, bool):
            clauses.append(f"({col})::boolean = %s")
        elif isinstance(value, int):
            clauses.append(f"({col})::integer = %s")
        elif isinstance(value, float):
            clauses.append(f"({col})::float = %s")
        else:
            clauses.append(f"{col} = %s")
        params.append(value)
    return ' AND '.join(clauses), params


def fetch_matches(conn, entries: list[tuple[str, FlashEntry]]) -> list[dict]:
    """Query task_queue for all matching rows, return list of row dicts."""
    rows: list[dict] = []
    with conn.cursor() as cur:
        for arch, entry in entries:
            entry_sql, entry_params = _entry_to_jsonb_filter(entry)
            sql = (
                f"SELECT id, arch, status FROM task_queue "
                f"WHERE task_config->>'arch' = %s AND {entry_sql}"
            )
            cur.execute(sql, [arch] + entry_params)
            rows.extend(cur.fetchall())
    return rows


def fetch_matches_by_ids(conn, task_ids: list[int]) -> list[dict]:
    """Query task_queue for specific task_ids, return list of row dicts."""
    if not task_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            'SELECT id, arch, status FROM task_queue WHERE id = ANY(%s)',
            (task_ids,),
        )
        return cur.fetchall()


def print_summary(label: str, count: int, matches: list[dict]) -> None:
    """Print a summary of matched task_queue rows by arch and status."""
    by_arch: Counter = Counter()
    by_status: Counter = Counter()
    for r in matches:
        by_arch[r['arch']] += 1
        by_status[r['status']] += 1

    print(f'\n{label} (de-duplicated): {count}')
    print(f'Matching task_queue rows:        {len(matches)}')

    if not matches:
        return

    print('\nBy arch:')
    for arch, n in sorted(by_arch.items()):
        print(f'  {arch}: {n}')

    print('\nBy current status:')
    for status, n in sorted(by_status.items()):
        print(f'  {status}: {n}')


def reset_to_pending(conn, row_ids: list[int], module: str = 'flash') -> int:
    """
    Reset the given task_queue ids to pending. Returns affected row count.

    Also deletes related rows from the results and accuracy tables so stale
    results don't contaminate future re-runs.  The tables chosen depend on
    whether module ends with '_op' (operator tuning) or not (kernel tuning).
    """
    if not row_ids:
        return 0
    if module.endswith('_op'):
        results_table  = 'optune_results'
        accuracy_table = 'most_accurate_optune_results'
    else:
        results_table  = 'tuning_results'
        accuracy_table = 'most_accurate_tuning_results'
    with conn.cursor() as cur:
        cur.execute(
            f'DELETE FROM {accuracy_table} WHERE task_id = ANY(%s)',
            (row_ids,),
        )
        cur.execute(
            f'DELETE FROM {results_table} WHERE task_id = ANY(%s)',
            (row_ids,),
        )
        cur.execute(
            """
            UPDATE task_queue
               SET status       = 'pending',
                   worker_id    = NULL,
                   node_hostname= NULL,
                   started_at   = NULL,
                   completed_at = NULL,
                   error        = NULL
             WHERE id = ANY(%s)
            """,
            (row_ids,),
        )
        return cur.rowcount
