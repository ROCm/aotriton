# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pgdef.task: dispatch, claim, completion and administration.

Every function takes a psycopg connection (`conn`), never connection
parameters; the caller owns the connection. Reads ask for `dict_row`
explicitly, so they work on a connection with any default row factory (and on
`ReconnectableConn`, which only offers `cursor()`).

All statements target the PARENT table with `pool = %s`, relying on partition
pruning; no code here builds a leaf table name. Partitions are created by the
SQL function pgdef.ensure_pool(), which applications call from their schema.sql.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import category as _category
from . import logistics as _logistics
from .category import Category
from .logistics import Logistics, TASK_COLUMNS

#: (id, pool): what identifies a task. `id` alone is unique in practice; pool
#: is carried because PostgreSQL needs the partition key to address a row
#: efficiently.
TaskRef = tuple[int, str]

POOL_RE = re.compile(r'^[a-z0-9_]+$')

#: Partitioned tables that get one leaf per pool.


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: int
    pool: str
    subpool: str | None
    cat_level0: str
    cat_level1: str | None
    payload: dict
    appcookie: dict | None
    priority: int
    status: str
    logistics: Logistics
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict) -> 'Task':
        return cls(
            id=row['id'], pool=row['pool'], subpool=row['subpool'],
            cat_level0=row['cat_level0'], cat_level1=row['cat_level1'],
            payload=row['payload'], appcookie=row['appcookie'],
            priority=row['priority'], status=row['status'],
            logistics=Logistics.from_json(row['logistics']),
            created_at=row.get('created_at'))

    @property
    def ref(self) -> TaskRef:
        return (self.id, self.pool)

    @property
    def category(self) -> Category:
        return (self.cat_level0, self.cat_level1)


@dataclass
class TaskSpec:
    """What an application's TaskSource yields for each task to dispatch."""
    payload: dict
    category: Category | None = None   # None: the driver fills in the command line's --category
    subpool: str | None = None
    priority: int = 5
    max_attempts: int | None = None   # None: the engine default applies


# ---------------------------------------------------------------------------
# Partitions and insertion
# ---------------------------------------------------------------------------

def validate_pool(pool: str) -> str:
    if not isinstance(pool, str) or not POOL_RE.match(pool):
        raise ValueError(f"invalid pool {pool!r}: expected {POOL_RE.pattern}")
    return pool


def list_pools(conn) -> list[str]:
    """Pools that have a task leaf, from the catalog."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT pg_get_expr(c.relpartbound, c.oid) AS bound
              FROM pg_inherits i
              JOIN pg_class c ON c.oid = i.inhrelid
              JOIN pg_class p ON p.oid = i.inhparent
              JOIN pg_namespace n ON n.oid = p.relnamespace
             WHERE n.nspname = 'pgdef' AND p.relname = 'task'
        """)
        bounds = [r['bound'] for r in cur.fetchall()]
    # bound looks like: FOR VALUES IN ('gfx950')
    return sorted(m.group(1) for b in bounds if (m := re.search(r"IN \('([^']*)'\)", b)))


def insert(conn, pool: str, specs: Iterable[TaskSpec], *, batch_size: int = 1000) -> int:
    """Insert tasks into one pool and return how many were inserted.

    No uniqueness is enforced: identical tasks are legal. Skipping work that is
    already done is dispatch's opt-in `--skip_completed` (see find()).
    """
    validate_pool(pool)
    total = 0
    batch: list[dict] = []
    with conn.cursor() as cur:
        def flush():
            nonlocal total, batch
            if batch:
                cur.executemany(
                    'INSERT INTO pgdef.task (pool, subpool, cat_level0, cat_level1, payload, priority, logistics) '
                    'VALUES (%(pool)s, %(subpool)s, %(cat_level0)s, %(cat_level1)s, %(payload)s, '
                    '%(priority)s, %(logistics)s)', batch)
                total += len(batch)
                batch = []

        for spec in specs:
            batch.append({
                'pool': pool, 'subpool': spec.subpool,
                **_category.to_columns(spec.category),
                'payload': Jsonb(spec.payload), 'priority': spec.priority,
                'logistics': Jsonb(_logistics.initial_json(spec.max_attempts)),
            })
            if len(batch) >= batch_size:
                flush()
        flush()
    return total


# ---------------------------------------------------------------------------
# Selection (admin operations, dispatch's --skip_completed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskSelector:
    """Which tasks an admin operation applies to. All given fields must match.

    category   (level0, level1); a level that is None is not filtered on, so
               ('tune', None) selects every tune task.
    status     one status or a tuple of them
    ids        explicit task ids
    payload_contains
               jsonb containment: `payload @> value`. The application builds
               the dict from its own entry dataclass.
    payload_exact
               {path: value}: `payload #> path = value` exactly. Containment
               treats arrays as sets (`[1,2] @> [2,1]`), so a tuple-valued field
               needs this to be compared as the ordered array it is.
    """
    pool: str | None = None
    subpool: str | None = None
    category: tuple[str | None, str | None] | None = None
    status: str | tuple[str, ...] | None = None
    ids: tuple[int, ...] | None = None
    payload_contains: dict | None = None
    payload_exact: dict | None = None

    def is_empty(self) -> bool:
        category_empty = self.category is None or all(level is None for level in self.category)
        return category_empty and all(
            getattr(self, f) in (None, {}, ()) for f in
            ('pool', 'subpool', 'status', 'ids', 'payload_contains', 'payload_exact'))

    def where(self, alias: str = 't') -> tuple[str, list]:
        """(SQL over `alias`, positional params). TRUE if nothing is selected."""
        a = f'{alias}.' if alias else ''
        clauses: list[str] = []
        params: list[Any] = []
        if self.pool is not None:
            clauses.append(f'{a}pool = %s')
            params.append(validate_pool(self.pool))
        if self.subpool is not None:
            clauses.append(f'{a}subpool = %s')
            params.append(self.subpool)
        if self.category is not None:
            for n, level in enumerate(self.category):
                if level is not None:
                    clauses.append(f'{a}cat_level{n} = %s')
                    params.append(level)
        if self.status is not None:
            statuses = (self.status,) if isinstance(self.status, str) else tuple(self.status)
            clauses.append(f'{a}status = ANY(%s)')
            params.append(list(statuses))
        if self.ids is not None:
            clauses.append(f'{a}id = ANY(%s)')
            params.append([int(i) for i in self.ids])
        if self.payload_contains:
            clauses.append(f'{a}payload @> %s')
            params.append(Jsonb(self.payload_contains))
        for path, value in (self.payload_exact or {}).items():
            clauses.append(f'{a}payload #> %s = %s')
            params.append(list(path))
            params.append(Jsonb(value))
        return (' AND '.join(clauses) or 'TRUE'), params


def find(conn, selector: TaskSelector, *, limit: int | None = None) -> list[Task]:
    """Tasks matching `selector`, ordered by (pool, id)."""
    where, params = selector.where('t')
    query = f'SELECT {TASK_COLUMNS} FROM pgdef.task t WHERE {where} ORDER BY pool, id'
    if limit is not None:
        query += ' LIMIT %s'
        params = params + [int(limit)]
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return [Task.from_row(r) for r in cur.fetchall()]


def get(conn, task_ref: TaskRef) -> Task | None:
    task_id, pool = task_ref
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f'SELECT {TASK_COLUMNS} FROM pgdef.task WHERE id = %s AND pool = %s',
                    (task_id, pool))
        row = cur.fetchone()
    return Task.from_row(row) if row else None


def count(conn, selector: TaskSelector) -> int:
    where, params = selector.where('t')
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM pgdef.task t WHERE {where}', params)
        return cur.fetchone()[0]


def payloads(conn, selector: TaskSelector) -> list[dict]:
    """Just the payloads of the matching tasks (dispatch's --skip_completed)."""
    where, params = selector.where('t')
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f'SELECT payload FROM pgdef.task t WHERE {where}', params)
        return [r['payload'] for r in cur.fetchall()]


def _require_selection(selector: TaskSelector, what: str) -> None:
    if selector.is_empty():
        raise ValueError(f'refusing to {what} every task: the selector selects nothing in particular')


def reset(conn, selector: TaskSelector, *, purge_reports: bool, keep_appcookie: bool = False) -> int:
    """Reset the selected tasks to pending, as fresh tasks.

    `purge_reports` is required and keyword-only: it is destructive (GPU-hours
    of measurements) and the destruction is not implied by the name. It deletes
    the tasks' pgdef.report rows; application tables derived from reports
    cascade, so no application code runs. `keep_appcookie` keeps the
    application's progress state so the redo can skip what the application
    chooses to skip; the default clears it, so a reset means a full redo.
    """
    _require_selection(selector, 'reset')
    where, params = selector.where('t')
    return _logistics.reset(conn, where, params, purge_reports=purge_reports,
                            keep_appcookie=keep_appcookie)


def cancel(conn, selector: TaskSelector) -> int:
    """Cancel selected pending or running tasks."""
    _require_selection(selector, 'cancel')
    where, params = selector.where('t')
    return _logistics.cancel(conn, where, params)


def delete(conn, selector: TaskSelector) -> int:
    """Delete the selected tasks; their reports (and anything derived from
    them) cascade."""
    _require_selection(selector, 'delete')
    where, params = selector.where('t')
    with conn.cursor() as cur:
        cur.execute(f'DELETE FROM pgdef.task t WHERE {where}', params)
        return cur.rowcount


# ---------------------------------------------------------------------------
# Running a task (names re-exported from logistics, the sole writer of the
# maintenance state)
# ---------------------------------------------------------------------------

def claim(conn, *, pool: str, category: Category, subpool: str | None, worker: str,
          lease_s: float = _logistics.DEFAULT_LEASE_SECONDS, limit: int = 1) -> list[Task]:
    """Claim up to `limit` pending tasks for a reader of (pool, subpool, category).

    A reader claims tasks whose subpool and deeper category levels are NULL
    (generic) or equal to its own; order is priority DESC, id.
    """
    return _logistics.claim(conn, Task, pool=pool, category=category, subpool=subpool,
                            worker=worker, lease_s=lease_s, limit=limit)


def renew(conn, task_ref: TaskRef, *, worker: str,
          lease_s: float = _logistics.DEFAULT_LEASE_SECONDS) -> bool:
    return _logistics.renew(conn, task_ref, worker=worker, lease_s=lease_s)


def complete(conn, task_ref: TaskRef) -> bool:
    return _logistics.complete(conn, task_ref)


def fail(conn, task_ref: TaskRef, error: str, *, permanent: bool = False) -> str | None:
    return _logistics.fail(conn, task_ref, error, permanent=permanent)


def release(conn, task_ref: TaskRef) -> bool:
    """Give a running task back (graceful shutdown). The attempt is not consumed."""
    return _logistics.release(conn, task_ref)


def reap_expired(conn, pool: str) -> list[tuple[int, str]]:
    return _logistics.reap_expired(conn, pool)


# ---------------------------------------------------------------------------
# appcookie: the application's progress state. Opaque to the engine.
# ---------------------------------------------------------------------------

APPEND_COOKIE_SQL = """
UPDATE pgdef.task SET appcookie = jsonb_set(
        COALESCE(appcookie, '{}'::jsonb),
        ARRAY[(%(key)s)::text],
        COALESCE(appcookie->((%(key)s)::text), '[]'::jsonb) || %(values)s::jsonb)
 WHERE id = %(id)s AND pool = %(pool)s
"""


def append_cookie(conn, task_ref: TaskRef, key: str, values: Sequence) -> None:
    """Append `values` to the array under top-level `key` of the cookie,
    creating it if absent, in one statement.

    The append happens in SQL under the row lock, so concurrent CPU handlers
    of the same task cannot lose each other's entries, which a client-side
    read-modify-write could. There is deliberately no merge (`||` of objects):
    it silently replaces an existing key's value.
    """
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.execute(APPEND_COOKIE_SQL, {'key': key, 'values': Jsonb(list(values)),
                                        'id': task_id, 'pool': pool})


def replace_cookie(conn, task_ref: TaskRef, cookie: dict | None) -> None:
    """Overwrite the whole cookie (compaction, or state that is not a list)."""
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.execute('UPDATE pgdef.task SET appcookie = %s WHERE id = %s AND pool = %s',
                    (None if cookie is None else Jsonb(cookie), task_id, pool))
