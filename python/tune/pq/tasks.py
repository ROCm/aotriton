# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
tune's view of the queue: find, reset and report progress on tune tasks.

Writes to engine tables go through aotriton.pgdef.pq. This module reads them
directly where the engine has no function for the question (it may: only
writes are restricted), and keeps tune's vocabulary:

    arch          the GPU architecture; it is the task's `pool`
    tuning_level  'kernel' | 'op'; stored as the task's `cat_level1`

A tune task's payload is {"module": <family>, "entry": {...}, "max_hsaco"?: {...}};
the arch and the level are the task's columns, not repeated in the payload.
"""

from dataclasses import asdict, is_dataclass

from psycopg.rows import dict_row

from aotriton.pgdef.pq import category as pgcat
from aotriton.pgdef.pq import progress as pgprogress
from aotriton.pgdef.pq import tasks as pgtasks
from aotriton.pgdef.pq.logistics import NOW
from aotriton.pgdef.pq.tasks import TaskSelector

APP = 'tune'
LEVELS = ('kernel', 'op')

# A task still 'running' after this long is assumed abandoned rather than slow,
# by `reset_task --select stale`. (Leases make this largely automatic now: see
# pgdef.pq.tasks.reap_expired. This is the operator's override.)
STALE_TIMEOUT_SECONDS = 7200


def category(level: str) -> pgcat.Category:
    """('tune', level), validated."""
    return (APP, check_level(level))


def check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown tuning_level {level!r}; expected one of {list(LEVELS)}")
    return level


def level_of(text: str | pgcat.Category) -> str:
    """The tune level named by a `--category` value: 'tune,kernel' or a
    parsed ('tune', 'kernel'). Anything outside tune, or without a level, is an
    error: the tools that call this are tune tools."""
    cat = pgcat.parse_category(text) if isinstance(text, str) else tuple(text)
    if cat[0] != APP or cat[1] is None:
        raise ValueError(f"expected a category like 'tune,kernel' or 'tune,op', got {pgcat.format_category(cat)!r}")
    return check_level(cat[1])


def entry_selector(entry, *, arch=None, tuning_level=None, module=None, **selector) -> TaskSelector:
    """TaskSelector for the tasks of one tuning entry.

    `entry` is a mapping of entry fields, or any dataclass instance (a family's
    Entry class): the field names are the family's, the shape of the query is
    not. Scalars match by jsonb containment, which compares numbers and
    booleans by value (no text casts). Tuples and lists are compared as the
    ORDERED json arrays they are, by exact path equality: containment would
    treat `[64, 128]` and `[128, 64]` alike, which for an asymmetric hdim is
    two different entries.

    Extra keyword arguments (status=, ids=, ...) go to the TaskSelector.
    """
    if is_dataclass(entry) and not isinstance(entry, type):
        entry = asdict(entry)
    scalars: dict = {}
    exact: dict = {}
    for name, value in entry.items():
        if isinstance(value, (tuple, list)):
            exact[('entry', name)] = list(value)
        else:
            scalars[name] = value
    contains: dict = {}
    if module is not None:
        contains['module'] = module
    if scalars:
        contains['entry'] = scalars
    return TaskSelector(
        pool=arch,
        category=(APP, check_level(tuning_level) if tuning_level is not None else None),
        payload_contains=contains or None,
        payload_exact=exact or None,
        **selector)


def _row(t: pgtasks.Task) -> dict:
    return {'id': t.id, 'arch': t.pool, 'module': t.payload.get('module'),
            'tuning_level': t.cat_level1, 'status': t.status}


def find_by_entry(conn, entry, *, arch=None, tuning_level=None, module=None,
                  limit: int | None = None) -> list[dict]:
    """Tune tasks matching one tuning entry, as {id, arch, module, tuning_level, status}.

    Filters are optional so a caller can be as specific as it can be: a pytest
    node ID carries no arch, while a TUNE_V3BIS line does. Supply tuning_level
    whenever it is known -- both levels share arch and every payload field, so
    omitting it matches an entry's rows at both.
    """
    found = pgtasks.find(conn, entry_selector(entry, arch=arch, tuning_level=tuning_level, module=module),
                         limit=limit)
    return [_row(t) for t in found]


def find_by_ids(conn, task_ids: list[int]) -> list[dict]:
    """Tune tasks for the given ids, in (arch, id) order."""
    if not task_ids:
        return []
    found = pgtasks.find(conn, TaskSelector(category=(APP, None), ids=tuple(task_ids)))
    return [_row(t) for t in found]


# Shared by both requeue lookups so their callers can treat the rows
# interchangeably. tuning_level is in here because reset() is scoped by it; age_s
# because a caller deciding whether a claim is abandoned needs to see how old it is.
_RESET_ROWS = """
SELECT t.id, t.pool AS arch, t.cat_level1 AS tuning_level, t.status,
       t.logistics->'lease'->>'owner' AS lease_owner,
       (%(now)s - (t.logistics->>'started')::float8)::bigint AS age_s
  FROM pgdef.task t
 WHERE t.cat_level0 = 'tune' AND {where}
 ORDER BY t.pool, t.id
"""


def find_resettable(conn, what: str, *, timeout_seconds: int | None = None) -> list[dict]:
    """Rows that will not be retried without intervention.

    what='stale'   status='running' for longer than timeout_seconds. Readers
                   reap expired leases themselves; this finds claims that are
                   still leased (a live but wedged reader) and are therefore
                   not reaped.
    what='failed'  status='failed', terminal until something moves it.

    Returns id, arch, tuning_level, status, lease_owner and the age in seconds.
    tuning_level is included because reset() is scoped by it, so a caller
    resetting these has to group by it first.
    """
    if what == 'stale':
        where = "t.status = 'running' AND %(now)s - (t.logistics->>'started')::float8 > %(timeout)s"
        params = {'timeout': timeout_seconds if timeout_seconds is not None else STALE_TIMEOUT_SECONDS}
    elif what == 'failed':
        where = "t.status = 'failed'"
        params = {}
    else:
        raise ValueError(f"unknown selection {what!r}; expected 'stale' or 'failed'")
    return _fetch_reset_rows(conn, where, params)


def find_resettable_by_ids(conn, task_ids: list[int]) -> list[dict]:
    """The same rows find_resettable() returns, for explicitly named ids.

    Deliberately unfiltered by status: a caller naming ids has already decided,
    and the statuses worth requeuing by hand are exactly the ones no rule
    covers. Ids that do not exist are simply absent from the result, so a
    caller can diff against what it asked for and say so.
    """
    if not task_ids:
        return []
    return _fetch_reset_rows(conn, 't.id = ANY(%(ids)s)', {'ids': [int(i) for i in task_ids]})


def _fetch_reset_rows(conn, where: str, params: dict) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_RESET_ROWS.format(where=where).replace('%(now)s', NOW), params)
        return cur.fetchall()


def reset(conn, row_ids: list[int], tuning_level: str, *, delete_results: bool) -> int:
    """Reset the given tune tasks to pending, for re-running.

    delete_results is keyword-only and REQUIRED because it is destructive and
    the destruction is not implied by the method name. With it set, the tasks'
    reports are dropped as well -- GPU-hours of measurements -- so that a re-run
    starts clean instead of mixing new results with stale ones; the derived
    most_accurate_tuning_results rows go with them (ON DELETE CASCADE). Pass
    False to requeue while keeping the existing rows.

    The selection is scoped by tuning_level as well as id: callers select ids by
    arch/entry, which both levels share, so an id list can span levels. Returns
    the number of tasks actually reset.
    """
    if not row_ids:
        return 0
    return pgtasks.reset(conn, TaskSelector(category=category(tuning_level), ids=tuple(row_ids)),
                         purge_reports=delete_results)


def get_progress(conn, tuning_level: str, *, recent_window_s: float = 300.0) -> dict:
    """Per-arch queue progress for one level.

    Returns {'progress': [...], 'speed': [...], 'stale': [...]}: the engine's
    queue_progress rows for ('tune', level) summed over subpools, recent
    completion counts, and counts of running tasks whose lease has lapsed.
    Callers merge and format these.
    """
    cat = category(tuning_level)
    by_pool: dict[str, dict] = {}
    for row in pgprogress.queue_progress(conn, cat):
        acc = by_pool.setdefault(row['pool'], {'arch': row['pool'], 'pending': 0, 'running': 0,
                                               'done': 0, 'failed': 0, 'cancelled': 0, 'total': 0})
        for key in ('pending', 'running', 'done', 'failed', 'cancelled', 'total'):
            acc[key] += row[key]
    for acc in by_pool.values():
        acc['pct_complete'] = round(100.0 * acc['done'] / acc['total'], 2) if acc['total'] else None
    speed = [{'arch': r['pool'], 'recent_completions': r['recent_completions']}
             for r in pgprogress.recent_completions(conn, cat, window_s=recent_window_s)]
    stale = [{'arch': r['pool'], 'stale_count': r['expired_count']}
             for r in pgprogress.expired_leases(conn, cat)]
    return {'progress': sorted(by_pool.values(), key=lambda r: r['arch']), 'speed': speed, 'stale': stale}


def completed_payloads(conn, module: str, tuning_level: str, *, arch: str | None = None) -> list[tuple[str, dict]]:
    """(arch, payload) of every done task of `module` at one level, for
    dispatch's --skip_completed."""
    sel = TaskSelector(pool=arch, category=category(tuning_level), status='done',
                       payload_contains={'module': module})
    return [(t.pool, t.payload) for t in pgtasks.find(conn, sel)]
