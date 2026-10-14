# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
CRUD helpers for tune.extra_uts -- extra input metadata associated with a task,
populated when re-queuing entries that failed pytest correctness checks. Rows
accumulate across passes and are never deleted by a reset.
"""

from psycopg.rows import dict_row

from aotriton.pgdef.pq.tasks import TaskRef


def insert_extra_uts(conn, task_ref: TaskRef, im_texts: list[str]) -> None:
    if not im_texts:
        return
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.executemany(
            '''INSERT INTO tune.extra_uts (task_id, pool, im_text, active) VALUES (%s, %s, %s, TRUE)
               ON CONFLICT (task_id, im_text) DO UPDATE SET active = TRUE''',
            [(task_id, pool, t) for t in im_texts],
        )


def get_extra_uts(conn, task_id: int) -> list[str]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            'SELECT im_text FROM tune.extra_uts WHERE task_id = %s AND active = TRUE ORDER BY id',
            (task_id,),
        )
        return [row['im_text'] for row in cur.fetchall()]


def delete_extra_uts(conn, task_ids: list[int]) -> None:
    """For full task deletion only -- not called by a reset. (Deleting the
    tasks cascades here anyway; this is for deleting the rows alone.)"""
    if not task_ids:
        return
    with conn.cursor() as cur:
        cur.execute('DELETE FROM tune.extra_uts WHERE task_id = ANY(%s)', (task_ids,))
