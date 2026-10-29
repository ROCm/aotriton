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
