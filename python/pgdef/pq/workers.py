# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pgdef.worker: one row per PG reader, upserted by the reader itself.

Called on claim, lease renewal and idle poll, so liveness is tied to real
progress instead of log-file mtimes.
"""

from psycopg.rows import dict_row

from .category import Category


def upsert(conn, *, worker_id: str, host: str, pool: str, subpool: str | None,
           category: Category, current_task: int | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            'INSERT INTO pgdef.worker '
            '    (worker_id, host, pool, subpool, cat_level0, cat_level1, last_seen, current_task) '
            'VALUES (%s, %s, %s, %s, %s, %s, now(), %s) '
            'ON CONFLICT (worker_id) DO UPDATE SET '
            '    host = EXCLUDED.host, pool = EXCLUDED.pool, subpool = EXCLUDED.subpool, '
            '    cat_level0 = EXCLUDED.cat_level0, cat_level1 = EXCLUDED.cat_level1, '
            '    last_seen = now(), current_task = EXCLUDED.current_task',
            (worker_id, host, pool, subpool, category[0], category[1], current_task))


def remove(conn, worker_id: str) -> None:
    """Forget a reader (graceful shutdown)."""
    with conn.cursor() as cur:
        cur.execute('DELETE FROM pgdef.worker WHERE worker_id = %s', (worker_id,))


def list_health(conn) -> list[dict]:
    """pgdef.worker_health rows, most recently seen first."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM pgdef.worker_health ORDER BY last_seen DESC')
        return cur.fetchall()
