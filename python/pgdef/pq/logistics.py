# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pgdef.task.logistics: the engine's schemaless maintenance state for a task.

THIS MODULE IS THE ONLY WRITER of the `logistics` column. Every statement that
changes it -- insert, claim, lease renewal, completion, failure, reaping,
reset, cancel -- lives here. `pq.tasks` re-exports the public ones under the
names applications use (`tasks.claim`, `tasks.fail`, ...).

Why a jsonb document and not columns: every new maintenance capability would
otherwise be DDL on a partitioned table, plus a coordinated stop of every node
(old workers SELECT a fixed column list). With jsonb a capability is a code
change only: the claim/renew/reap SQL gains a predicate or a jsonb update, rows
written by older code read as "feature absent" through the defaults below, and
nodes upgrade one at a time.

Document, version 1 (all times are epoch seconds, float):

    {
      "v": 1,
      "attempt": 2,                      incremented by claim
      "retry": {"max_attempts": 3},      stamped at dispatch (only if not the default)
      "lease": {"owner": "node7/reader-2", "until": 1790000000.0, "renewed": 1789999940.0},
      "started": 1789999000.0, "finished": null,
      "errors": [{"attempt": 1, "reason": "lease_expired", "at": ..., "worker": "..."}]   # capped
    }

Times are stored as float8 epochs, compared against `extract(epoch from now())`
(the database's clock, so nodes need no synchronised clocks): `text::timestamptz`
depends on the TimeZone setting and is not IMMUTABLE, so it cannot be indexed.

Readers of a task use the typed `Logistics` accessor, which applies defaults for
absent keys and ignores unknown ones. SQL below reads the same keys directly.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import category as _category

logger = logging.getLogger(__name__)

VERSION = 1
# One attempt: automatic retry is supported but unused. A failure, or an
# expired lease, ends the task in 'failed', and redo is an operator reset. A
# task gets retries only if it is dispatched with a larger budget
# (TaskSpec.max_attempts, --max_attempts, or a source's default_max_attempts).
DEFAULT_MAX_ATTEMPTS = 1
ERROR_HISTORY_CAP = 8
DEFAULT_LEASE_SECONDS = 300.0

# The database's clock. Used instead of the client's so that nodes with skewed
# clocks still agree on whether a lease has lapsed.
NOW = 'extract(epoch from now())'

# Columns every task read returns, in the order Task.from_row() expects.
TASK_COLUMNS = ('id, pool, subpool, cat_level0, cat_level1, payload, appcookie, '
                'priority, status, logistics, created_at')


# ---------------------------------------------------------------------------
# Typed accessor (read side)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lease:
    owner: str | None = None
    until: float | None = None
    renewed: float | None = None


@dataclass(frozen=True)
class Logistics:
    """Read-only view of a logistics document, with defaults for absent keys.

    Unknown keys are ignored, so rows written by newer code can be read by older
    code. A different major `v` is upgraded on read by `from_json`; today there
    is only version 1.
    """
    v: int = VERSION
    attempt: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    lease: Lease = field(default_factory=Lease)
    started: float | None = None
    finished: float | None = None
    errors: tuple[dict, ...] = ()

    @classmethod
    def from_json(cls, doc: dict | None) -> 'Logistics':
        doc = doc or {}
        lease = doc.get('lease') or {}
        retry = doc.get('retry') or {}
        return cls(
            v=VERSION,   # upgrade-on-read: every older shape reads as the current one
            attempt=int(doc.get('attempt') or 0),
            max_attempts=int(retry.get('max_attempts') or DEFAULT_MAX_ATTEMPTS),
            lease=Lease(owner=lease.get('owner'),
                        until=lease.get('until'),
                        renewed=lease.get('renewed')),
            started=doc.get('started'),
            finished=doc.get('finished'),
            errors=tuple(doc.get('errors') or ()),
        )

    @property
    def attempts_left(self) -> int:
        return max(self.max_attempts - self.attempt, 0)

    def lease_expired(self, now: float) -> bool:
        return self.lease.until is not None and self.lease.until < now


def initial_json(max_attempts: int | None = None) -> dict:
    """The logistics document of a newly dispatched task."""
    doc: dict[str, Any] = {'v': VERSION}
    if max_attempts is not None:
        doc['retry'] = {'max_attempts': int(max_attempts)}
    return doc


# ---------------------------------------------------------------------------
# SQL fragments (write side)
# ---------------------------------------------------------------------------

# Attempt budget, read from the row itself. `COALESCE` supplies the default for
# rows whose dispatcher stamped no policy.
_ATTEMPT = "COALESCE((t.logistics->>'attempt')::int, 0)"
_MAX_ATTEMPTS = (f"COALESCE((t.logistics->'retry'->>'max_attempts')::int, "
                 f"{DEFAULT_MAX_ATTEMPTS})")


def _push_error(entry_sql: str) -> str:
    """jsonb array: the newest ERROR_HISTORY_CAP-1 old errors, then `entry_sql`."""
    keep = ERROR_HISTORY_CAP - 1
    return (
        "(SELECT COALESCE(jsonb_agg(e.value ORDER BY e.ord), '[]'::jsonb) "
        "   FROM (SELECT value, ord "
        "           FROM jsonb_array_elements(COALESCE(t.logistics->'errors', '[]'::jsonb)) "
        "                WITH ORDINALITY AS x(value, ord) "
        f"          ORDER BY ord DESC LIMIT {keep}) AS e) "
        f"|| jsonb_build_array({entry_sql})")


def claim_sql(levels: int = _category.LEVELS) -> str:
    """The claim statement as text (also what the unit tests look at).

    One statement on the PARENT table: `pool = %(pool)s` lets partition pruning
    pick the leaf. `FOR UPDATE SKIP LOCKED` keeps concurrent readers off each
    other's rows. `x = NULL` is never true, so a reader whose subpool or
    deeper category level is NULL claims only generic tasks.

    Future logistics capabilities add a predicate to the inner WHERE, e.g.
        AND COALESCE((logistics->>'not_before')::float8, 0) <= extract(epoch from now())
    """
    return f"""
UPDATE pgdef.task t SET
    status = 'running',
    logistics = t.logistics || jsonb_build_object(
        'attempt', {_ATTEMPT} + 1,
        'lease', jsonb_build_object('owner', %(worker)s::text,
                                    'until', {NOW} + %(lease_s)s::float8,
                                    'renewed', {NOW}),
        'started', {NOW})
 WHERE t.pool = %(pool)s
   AND (t.id, t.pool) IN (
        SELECT id, pool FROM pgdef.task
         WHERE pool = %(pool)s
           AND {_category.claim_terms(levels)}
           AND status = 'pending'
           AND (subpool IS NULL OR subpool = %(subpool)s)
         ORDER BY priority DESC, id
         LIMIT %(limit)s
           FOR UPDATE SKIP LOCKED)
RETURNING {', '.join('t.' + c.strip() for c in TASK_COLUMNS.split(','))}
"""


def renew_sql() -> str:
    return f"""
UPDATE pgdef.task t SET
    logistics = jsonb_set(jsonb_set(t.logistics, '{{lease,until}}',
                                    to_jsonb({NOW} + %(lease_s)s::float8)),
                          '{{lease,renewed}}', to_jsonb({NOW}))
 WHERE t.id = %(id)s AND t.pool = %(pool)s
   AND t.status = 'running'
   AND t.logistics->'lease'->>'owner' = %(worker)s
"""


def _settle_sql(reason: str | None) -> str:
    """UPDATE ... FROM cur: move a running task to pending or failed.

    `cur` (a CTE the caller defines) supplies `dead BOOLEAN` per row. The task
    becomes failed if `dead`, pending otherwise; the lease is dropped either
    way; `finished` is stamped only when it fails. `reason` non-NULL appends
    an entry to the capped error history.
    """
    if reason is None:
        errors = ''
    else:
        entry = ("jsonb_build_object('attempt', " + _ATTEMPT + ", 'reason', %(reason)s::text, "
                 f"'at', {NOW}, 'worker', t.logistics->'lease'->>'owner', "
                 "'detail', %(detail)s::text)")
        errors = f"|| jsonb_build_object('errors', {_push_error(entry)})"
    return f"""
UPDATE pgdef.task t SET
    status = CASE WHEN cur.dead THEN 'failed' ELSE 'pending' END,
    logistics = (t.logistics - 'lease')
                {errors}
                || CASE WHEN cur.dead THEN jsonb_build_object('finished', {NOW})
                        ELSE '{{}}'::jsonb END
  FROM cur
 WHERE t.id = cur.id AND t.pool = cur.pool
"""


def fail_sql() -> str:
    return f"""
WITH cur AS (
    SELECT t.id, t.pool,
           (%(permanent)s::boolean OR {_ATTEMPT} >= {_MAX_ATTEMPTS}) AS dead
      FROM pgdef.task t
     WHERE t.id = %(id)s AND t.pool = %(pool)s AND t.status = 'running'
       FOR UPDATE)
{_settle_sql('error')}
RETURNING t.status
"""


def reap_sql() -> str:
    return f"""
WITH cur AS (
    SELECT t.id, t.pool,
           ({_ATTEMPT} >= {_MAX_ATTEMPTS}) AS dead
      FROM pgdef.task t
     WHERE t.pool = %(pool)s AND t.status = 'running'
       AND (t.logistics->'lease'->>'until')::float8 < {NOW}
       FOR UPDATE SKIP LOCKED)
{_settle_sql('lease_expired')}
RETURNING t.id, t.status
"""


def release_sql() -> str:
    """Graceful release (shutdown): back to pending, attempt not consumed."""
    return f"""
UPDATE pgdef.task t SET
    status = 'pending',
    logistics = jsonb_set(t.logistics - 'lease', '{{attempt}}',
                          to_jsonb(GREATEST({_ATTEMPT} - 1, 0)))
 WHERE t.id = %(id)s AND t.pool = %(pool)s AND t.status = 'running'
"""


def complete_sql() -> str:
    return f"""
UPDATE pgdef.task t SET
    status = 'done',
    logistics = (t.logistics - 'lease') || jsonb_build_object('finished', {NOW})
 WHERE t.id = %(id)s AND t.pool = %(pool)s AND t.status = 'running'
"""


def reset_sql(where: str, *, purge_reports: bool, keep_appcookie: bool) -> str:
    """Reset the rows matched by `where` (a WHERE body over alias `t`, with
    positional %s parameters) to pending, as a fresh task.

    Attempts, lease, timings and error history are dropped; the retry policy
    is kept. appcookie is cleared unless `keep_appcookie`. With
    `purge_reports`, the tasks' pgdef.report rows are deleted in the SAME
    statement, so a crash cannot leave done-looking tasks without results;
    application tables that derive from reports cascade (ON DELETE CASCADE).
    """
    purge = ("DELETE FROM pgdef.report r USING sel "
             "WHERE r.task_id = sel.id AND r.pool = sel.pool") if purge_reports else None
    cookie = '' if keep_appcookie else ', appcookie = NULL'
    sql = f"WITH sel AS (SELECT t.id, t.pool FROM pgdef.task t WHERE {where} FOR UPDATE)"
    if purge:
        sql += f",\n     purged AS ({purge} RETURNING r.id)"
    sql += f"""
UPDATE pgdef.task t SET
    status = 'pending',
    logistics = jsonb_build_object('v', {VERSION})
                || CASE WHEN (t.logistics->'retry') IS NOT NULL
                        THEN jsonb_build_object('retry', t.logistics->'retry')
                        ELSE '{{}}'::jsonb END
    {cookie}
  FROM sel
 WHERE t.id = sel.id AND t.pool = sel.pool
"""
    return sql


def cancel_sql(where: str) -> str:
    return f"""
UPDATE pgdef.task t SET
    status = 'cancelled',
    logistics = (t.logistics - 'lease') || jsonb_build_object('finished', {NOW})
 WHERE t.status IN ('pending', 'running') AND {where}
"""


# ---------------------------------------------------------------------------
# Statements that run (called by pq.tasks)
# ---------------------------------------------------------------------------

def claim(conn, task_cls, *, pool: str, category, subpool: str | None,
          worker: str, lease_s: float = DEFAULT_LEASE_SECONDS, limit: int = 1) -> list:
    params = {'pool': pool, 'subpool': subpool, 'worker': worker,
              'lease_s': float(lease_s), 'limit': int(limit),
              **_category.to_columns(category)}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(claim_sql(), params)
        rows = cur.fetchall()
    # RETURNING order is unspecified; present the claim in dispatch order.
    rows.sort(key=lambda r: (-r['priority'], r['id']))
    return [task_cls.from_row(r) for r in rows]


def renew(conn, task_ref, *, worker: str, lease_s: float = DEFAULT_LEASE_SECONDS) -> bool:
    """Extend the lease. False means this worker no longer holds the task
    (reaped, reset, cancelled): the caller should stop working on it."""
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.execute(renew_sql(), {'id': task_id, 'pool': pool, 'worker': worker,
                                  'lease_s': float(lease_s)})
        return cur.rowcount == 1


def complete(conn, task_ref) -> bool:
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.execute(complete_sql(), {'id': task_id, 'pool': pool})
        return cur.rowcount == 1


def fail(conn, task_ref, error: str, *, permanent: bool = False) -> str | None:
    """Fail one attempt. Returns the task's new status: 'pending' if it will be
    retried, 'failed' if the attempts are used up or `permanent`; None if the
    task was not running."""
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.execute(fail_sql(), {'id': task_id, 'pool': pool, 'permanent': bool(permanent),
                                 'reason': 'error', 'detail': str(error)[:2000]})
        row = cur.fetchone()
    return row[0] if row else None


def release(conn, task_ref) -> bool:
    task_id, pool = task_ref
    with conn.cursor() as cur:
        cur.execute(release_sql(), {'id': task_id, 'pool': pool})
        return cur.rowcount == 1


def reap_expired(conn, pool: str) -> list[tuple[int, str]]:
    """Move running tasks whose lease lapsed back to pending (or to failed when
    out of attempts). Returns [(task_id, new_status), ...]."""
    with conn.cursor() as cur:
        cur.execute(reap_sql(), {'pool': pool, 'reason': 'lease_expired', 'detail': ''})
        return [(r[0], r[1]) for r in cur.fetchall()]


def reset(conn, where: str, params: list, *, purge_reports: bool, keep_appcookie: bool) -> int:
    with conn.cursor() as cur:
        cur.execute(reset_sql(where, purge_reports=purge_reports, keep_appcookie=keep_appcookie),
                    params)
        return cur.rowcount


def cancel(conn, where: str, params: list) -> int:
    with conn.cursor() as cur:
        cur.execute(cancel_sql(where), params)
        return cur.rowcount
