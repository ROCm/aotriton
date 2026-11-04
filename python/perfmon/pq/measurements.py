# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
perfmon's measurements: written as pgdef.report rows, read through the
`perfmon.measurement` and `perfmon.latest_measurement` views.

Storage mapping: the task's pool is the arch and its subpool the variant
(NULL reads as 'any'); a report's rkey is {"iface", "backend"}, backend NULL
for the unforced operator. Code outside this package says arch, variant,
subject, iface and backend.
"""

from psycopg.rows import dict_row

from aotriton.pgdef.pq import reports as pgreports
from aotriton.pgdef.pq.tasks import TaskRef

APP = 'perfmon'

#: Report statuses. `excluded`: the subject, arch or GPU cannot run the point
#: (unsupported, unavailable backend, does not fit); nothing was timed.
STATUSES = ('OK', 'NotOK', 'crash', 'excluded')

#: The variant of tasks dispatched without a subpool.
ANY_VARIANT = 'any'


def save_measurement(conn, task_ref: TaskRef, report: dict, *, attempt: int) -> int:
    """Write one measurement as a pgdef.report row and return its id.

    `report` holds `iface`, `backend` (None: unforced), `status` (STATUSES),
    `payload` (the measurement: stats, platform, thermal, ...; may be None),
    `error` (None for OK) and `gpu`.

    A retried attempt measures again, so a result replaces any earlier report
    of the same (iface, backend) of this task.
    """
    status = report['status']
    if status not in STATUSES:
        raise ValueError(f'unknown measurement status {status!r}; expected one of {STATUSES}')
    gpu = report.get('gpu')
    return pgreports.insert(conn,
                            task_ref,
                            cat_level0=APP,
                            attempt=attempt,
                            rkey={'iface': report['iface'], 'backend': report['backend']},
                            status=status,
                            payload=report.get('payload') or None,
                            error=report.get('error') or None,
                            worker=None if gpu is None else f'gpu-{gpu}',
                            replace=True)


def _where(*, module=None, arch=None, variant=None, subjects=None) -> tuple[str, list]:
    clauses, params = [], []
    if module is not None:
        clauses.append('module = %s')
        params.append(module)
    if arch is not None:
        clauses.append('arch = %s')
        params.append(arch)
    if variant is not None:
        clauses.append('variant = %s')
        params.append(variant)
    if subjects:
        clauses.append('subject = ANY(%s)')
        params.append(list(subjects))
    return (' AND '.join(clauses) or 'TRUE'), params


LATEST_COLUMNS = ('id, task_id, arch, variant, module, subject, iface, backend, entry, status, '
                  'median_us, mean_us, stddev_us, min_us, p05_us, p95_us, timing_method, iterations, '
                  'l2_flush, platform, thermal, harness, error, worker, created_at')


def iter_latest(conn, *, arch: str, variant: str | None = None, subjects=None, module: str | None = None):
    """The newest measurement of every point of `arch`, as dicts with the
    columns of `perfmon.latest_measurement` (LATEST_COLUMNS), ordered by
    variant, subject, iface, backend."""
    where, params = _where(module=module, arch=arch, variant=variant, subjects=subjects)
    query = (f'SELECT {LATEST_COLUMNS} FROM perfmon.latest_measurement WHERE {where} '
             'ORDER BY variant, subject, iface, backend NULLS FIRST, id')
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        yield from cur.fetchall()


def count_superseded(conn, *, arch: str, variant: str | None = None, subjects=None,
                     module: str | None = None) -> int:
    """How many measurements of the selection a newer one of the same point
    supersedes (the export reports it)."""
    where, params = _where(module=module, arch=arch, variant=variant, subjects=subjects)
    query = (f'SELECT (SELECT count(*) FROM perfmon.measurement WHERE {where}) '
             f'- (SELECT count(*) FROM perfmon.latest_measurement WHERE {where}) AS superseded')
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params + params)
        row = cur.fetchone()
    return int(row['superseded']) if row else 0


def list_selections(conn, *, module: str | None = None, arch: str | None = None,
                    variant: str | None = None, subjects=None) -> list[dict]:
    """The (module, arch, variant, subject) combinations that have
    measurements, filtered by whatever is given: per combination the number
    of latest measurements, the count of each status, and the newest date.
    Rows are dicts ordered module, arch, variant, subject."""
    where, params = _where(module=module, arch=arch, variant=variant, subjects=subjects)
    query = ('SELECT module, arch, variant, subject, count(*) AS measured, '
             "count(*) FILTER (WHERE status = 'OK') AS ok, "
             "count(*) FILTER (WHERE status = 'NotOK') AS notok, "
             "count(*) FILTER (WHERE status = 'crash') AS crash, "
             "count(*) FILTER (WHERE status = 'excluded') AS excluded, "
             'max(created_at) AS newest '
             f'FROM perfmon.latest_measurement WHERE {where} '
             'GROUP BY module, arch, variant, subject '
             'ORDER BY module, arch, variant, subject')
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def subject_progress(conn, *, arch: str | None = None) -> list[dict]:
    """Task counts per (arch, variant, subject) and status, for the webui:
    rows {arch, variant, subject, pending, running, done, failed, cancelled,
    total}."""
    clauses = ["t.cat_level0 = 'perfmon'"]
    params: list = []
    if arch is not None:
        clauses.append('t.pool = %s')
        params.append(arch)
    query = ("SELECT t.pool AS arch, COALESCE(t.subpool, 'any') AS variant, "
             "t.payload->>'subject' AS subject, "
             "count(*) FILTER (WHERE t.status = 'pending') AS pending, "
             "count(*) FILTER (WHERE t.status = 'running') AS running, "
             "count(*) FILTER (WHERE t.status = 'done') AS done, "
             "count(*) FILTER (WHERE t.status = 'failed') AS failed, "
             "count(*) FILTER (WHERE t.status = 'cancelled') AS cancelled, "
             'count(*) AS total '
             f"FROM pgdef.task t WHERE {' AND '.join(clauses)} "
             'GROUP BY 1, 2, 3 ORDER BY 1, 2, 3')
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def _task_where(*, module=None, arch=None, variant=None, subjects=None) -> tuple[str, list]:
    clauses, params = ["t.cat_level0 = 'perfmon'"], []
    if module is not None:
        clauses.append("t.payload->>'module' = %s")
        params.append(module)
    if arch is not None:
        clauses.append('t.pool = %s')
        params.append(arch)
    if variant is not None:
        clauses.append("COALESCE(t.subpool, 'any') = %s")
        params.append(variant)
    if subjects:
        clauses.append("t.payload->>'subject' = ANY(%s)")
        params.append(list(subjects))
    return ' AND '.join(clauses), params


def _pairs(value) -> set[tuple]:
    return {(iface, backend) for iface, backend in (value or [])}


def task_shortfalls(conn, *, module: str | None = None, arch: str | None = None,
                    variant: str | None = None, subjects=None) -> list[dict]:
    """The perfmon tasks of the selection whose results may be incomplete.

    A task is marked `done` before all its result rows are written (by design,
    see "Completion does not wait for result writes" in docs/Distributed
    Execution Framework.md). The (iface, backend) pairs recorded done in the
    task's appcookie are compared with the report rows present. A task is
    returned, as {task_id, arch, variant, subject, module, entry, status,
    missing, reported, reason}, when its status is not `done`, when it
    recorded pairs that have no report row (`missing`, sorted), or when it is
    done with no report at all. `reported` counts the task's report rows. A
    task whose every report a newer dispatch superseded is left out.
    Only perfmon tasks are looked at; the tuner's tasks in the same database
    never match.
    """
    where, params = _task_where(module=module, arch=arch, variant=variant, subjects=subjects)
    query = ("SELECT t.id AS task_id, t.pool AS arch, COALESCE(t.subpool, 'any') AS variant, "
             "t.payload->>'subject' AS subject, t.payload->>'module' AS module, "
             "t.payload->'entry' AS entry, t.status, "
             "COALESCE(t.appcookie->'done', '[]'::jsonb) AS done, "
             "COALESCE((SELECT jsonb_agg(jsonb_build_array(r.rkey->>'iface', r.rkey->>'backend')) "
             "FROM pgdef.report r WHERE r.task_id = t.id AND r.pool = t.pool "
             "AND r.cat_level0 = 'perfmon'), '[]'::jsonb) AS reported, "
             'EXISTS (SELECT 1 FROM perfmon.latest_measurement l '
             'WHERE l.task_id = t.id AND l.arch = t.pool) AS current '
             f'FROM pgdef.task t WHERE {where} ORDER BY t.pool, t.id')
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        tasks = cur.fetchall()
    out = []
    for task in tasks:
        done, reported = _pairs(task.pop('done')), _pairs(task.pop('reported'))
        if reported and not task.pop('current'):
            continue            # fully superseded by a newer dispatch: no export depends on it
        task.pop('current', None)
        missing = sorted(done - reported, key=lambda p: (p[0], p[1] or ''))
        if task['status'] != 'done':
            reason = f"task is {task['status']}"
        elif missing:
            reason = f'{len(missing)} of {len(done)} recorded results have no report row'
        elif not reported:
            reason = 'task is done but has no report row'
        else:
            continue
        out.append({**task, 'missing': [list(p) for p in missing], 'reported': len(reported),
                    'reason': reason})
    return out
