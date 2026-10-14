# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Progress and health reads over the engine views.

They know no application: category filters are plain level values.
"""

from psycopg.rows import dict_row

from .logistics import NOW


def _category_where(category, alias: str = '') -> tuple[list[str], list]:
    a = f'{alias}.' if alias else ''
    clauses, params = [], []
    for n, level in enumerate(category or ()):
        if level is not None:
            clauses.append(f'{a}cat_level{n} = %s')
            params.append(level)
    return clauses, params


def queue_progress(conn, category=None) -> list[dict]:
    """pgdef.queue_progress rows (per pool, subpool and category), optionally
    for one category. A None level is not filtered on."""
    clauses, params = _category_where(category)
    query = 'SELECT * FROM pgdef.queue_progress'
    if clauses:
        query += ' WHERE ' + ' AND '.join(clauses)
    query += ' ORDER BY pool, subpool NULLS FIRST, cat_level0, cat_level1'
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def recent_completions(conn, category=None, *, window_s: float = 300.0) -> list[dict]:
    """Done tasks per pool whose completion fell in the last `window_s` seconds."""
    clauses, params = _category_where(category)
    where = ' AND '.join(["status = 'done'",
                          f"(logistics->>'finished')::float8 > {NOW} - %s"] + clauses)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f'SELECT pool, COUNT(*) AS recent_completions FROM pgdef.task '
                    f'WHERE {where} GROUP BY pool', [float(window_s)] + params)
        return cur.fetchall()


def expired_leases(conn, category=None) -> list[dict]:
    """Per pool, running tasks whose lease has lapsed and that nobody has reaped."""
    clauses, params = _category_where(category)
    query = 'SELECT pool, COUNT(*) AS expired_count FROM pgdef.expired_leases'
    if clauses:
        query += ' WHERE ' + ' AND '.join(clauses)
    query += ' GROUP BY pool'
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchall()
