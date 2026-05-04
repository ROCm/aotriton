# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
CRUD helpers for task_extra_uts — extra input metadata associated with a
task_queue entry, populated when re-queuing entries that failed pytest
correctness checks.  Rows accumulate across passes and are never deleted
by reset_to_pending.
"""


def insert_extra_uts(conn, task_id: int, im_texts: list[str]) -> None:
    if not im_texts:
        return
    with conn.cursor() as cur:
        cur.executemany(
            '''INSERT INTO task_extra_uts (task_id, im_text, active) VALUES (%s, %s, TRUE)
               ON CONFLICT (task_id, im_text) DO UPDATE SET active = TRUE''',
            [(task_id, t) for t in im_texts],
        )


def get_extra_uts(conn, task_id: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            'SELECT im_text FROM task_extra_uts WHERE task_id = %s AND active = TRUE',
            (task_id,),
        )
        return [row[0] for row in cur.fetchall()]


def delete_extra_uts(conn, task_ids: list[int]) -> None:
    """For full task deletion only — not called by reset_to_pending."""
    if not task_ids:
        return
    with conn.cursor() as cur:
        cur.execute('DELETE FROM task_extra_uts WHERE task_id = ANY(%s)', (task_ids,))
