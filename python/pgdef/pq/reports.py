# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pgdef.report: raw per-item results.

One row per tuned impl, measured backend, ... The application owns the
vocabulary of `rkey` (which item) and `status` (OK / NotOK / crash / ERROR).
Keeping the raw rows in the engine is what lets a generic reset purge results
without application code: application tables derived from reports reference
them with ON DELETE CASCADE.
"""

from typing import Any

from psycopg.rows import dict_row, tuple_row
from psycopg.types.json import Jsonb

from .tasks import TaskRef

REPORT_COLUMNS = 'id, task_id, pool, cat_level0, attempt, rkey, status, payload, error, worker, created_at'


def insert(conn, task_ref: TaskRef, *, cat_level0: str, attempt: int, rkey: dict, status: str,
           payload: Any = None, error: Any = None, worker: str | None = None,
           replace: bool = False) -> int:
    """Insert one report row and return its id.

    With `replace`, the task's existing reports with the SAME rkey are deleted
    in the same statement, so the item is reported once: a retried attempt (or
    a zombie writer after a lease expiry) re-measures what an earlier attempt
    already reported without leaving two rows. Rows derived from the deleted
    ones cascade.
    """
    task_id, pool = task_ref
    params = (task_id, pool, cat_level0, attempt, Jsonb(rkey), status,
              None if payload is None else Jsonb(payload),
              None if error is None else Jsonb(error), worker)
    if replace:
        query = ('WITH replaced AS (DELETE FROM pgdef.report WHERE task_id = %s AND pool = %s '
                 'AND cat_level0 = %s AND rkey = %s RETURNING id) '
                 'INSERT INTO pgdef.report (task_id, pool, cat_level0, attempt, rkey, status, payload, error, worker) '
                 'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id')
        params = (task_id, pool, cat_level0, Jsonb(rkey)) + params
    else:
        query = ('INSERT INTO pgdef.report (task_id, pool, cat_level0, attempt, rkey, status, payload, error, worker) '
                 'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id')
        with conn.cursor(row_factory=tuple_row) as cur:
            cur.execute(query, params)
            return cur.fetchone()[0]
    # Replacements of the same task are serialized: lock the task row first, in
    # one transaction, so a second replacer waits and then, with a fresh
    # snapshot, sees (and deletes) the first one's row. Without the lock, an
    # expired attempt and its retry could both find nothing to delete and
    # leave two rows for one item.
    with conn.transaction():
        with conn.cursor(row_factory=tuple_row) as cur:
            cur.execute('SELECT 1 FROM pgdef.task WHERE id = %s AND pool = %s FOR UPDATE',
                        (task_id, pool))
            cur.execute(query, params)
            return cur.fetchone()[0]


def for_task(conn, task_ref: TaskRef, *, rkey_contains: dict | None = None) -> list[dict]:
    """The task's reports, oldest first. `rkey_contains` narrows by jsonb containment."""
    task_id, pool = task_ref
    query = f'SELECT {REPORT_COLUMNS} FROM pgdef.report WHERE task_id = %s AND pool = %s'
    params: list = [task_id, pool]
    if rkey_contains:
        query += ' AND rkey @> %s'
        params.append(Jsonb(rkey_contains))
    query += ' ORDER BY id'
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchall()
