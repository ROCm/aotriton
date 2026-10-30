# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.perfmon.pq against a real, disposable PostgreSQL server.

Marker `pg`: skipped unless AOTRITON_PG_TEST_DISPOSABLE=1 and a server answers
(see aotriton.pgdef.pq.testkit). These tests have NOT been run by the author:
the development sandbox had no PostgreSQL server. Expect to find a SQL typo.
"""

import time

import pytest

from aotriton.pgdef.pq import tasks
from aotriton.pgdef.pq.tasks import TaskSpec
from aotriton.perfmon import pq

pytestmark = pytest.mark.pg

ARCH = 'gfx950'
CATEGORY = ('perfmon', None)
S13 = 'aotriton-0.13b+rocm10.1.0'
S14 = 'aotriton-0.14.2b+rocm10.1.0'


def _entry(seqlen=1024):
    return {'T_io': 'bf16', 'hdim_qk': 64, 'hdim_v': 64, 'causal': 0, 'seqlen_q': seqlen,
            'seqlen_k': seqlen, 'batch': 2, 'n_heads_q': 4}


def _task(conn, subject=S13, entry=None, subpool=None):
    spec = TaskSpec({'module': 'flash', 'subject': subject, 'entry': entry or _entry()},
                    CATEGORY,
                    subpool=subpool)
    tasks.ensure_pool(conn, ARCH)
    tasks.insert(conn, ARCH, [spec])
    got = tasks.claim(conn, pool=ARCH, category=CATEGORY, subpool=subpool, worker='w0', limit=1)
    return got[0]


def _report(median=10.0, **patch):
    out = {'iface': 'op_attn_fwd', 'backend': 'triton', 'status': 'OK', 'gpu': 0, 'error': None,
           'payload': {'stats': {'median': median, 'mean': median + 0.1, 'stddev': 0.05, 'min': median - 0.5,
                                 'p05': median - 0.3, 'p95': median + 0.4},
                       'timing_method': 'hipgraph_ev100', 'n': 100, 'l2_flush': 1,
                       'platform': {'pci_device_id': '0x75a0'}, 'harness': {'commit': 'abc'}}}
    out.update(patch)
    return out


def _all(conn, **kw):
    return list(pq.iter_latest(conn, arch=ARCH, **kw))


def test_a_measurement_appears_in_the_view_with_typed_columns(pg_conn):
    task = _task(pg_conn)
    pq.save_measurement(pg_conn, task.ref, _report(), attempt=1)
    (row,) = _all(pg_conn)
    assert (row['arch'], row['variant'], row['subject'], row['module']) == (ARCH, 'any', S13, 'flash')
    assert (row['iface'], row['backend'], row['status']) == ('op_attn_fwd', 'triton', 'OK')
    assert row['median_us'] == 10.0 and row['p95_us'] == pytest.approx(10.4)
    assert row['entry']['seqlen_q'] == 1024
    assert (row['timing_method'], row['iterations'], row['l2_flush']) == ('hipgraph_ev100', 100, 1)
    assert row['platform']['pci_device_id'] == '0x75a0' and row['harness']['commit'] == 'abc'


def test_the_unforced_operator_has_a_null_backend(pg_conn):
    task = _task(pg_conn)
    pq.save_measurement(pg_conn, task.ref, _report(backend=None), attempt=1)
    (row,) = _all(pg_conn)
    assert row['backend'] is None


def test_a_retry_replaces_the_report_of_the_same_iface_and_backend(pg_conn):
    task = _task(pg_conn)
    pq.save_measurement(pg_conn, task.ref, _report(10.0), attempt=1)
    pq.save_measurement(pg_conn, task.ref, _report(11.0), attempt=2)
    (row,) = _all(pg_conn)
    assert row['median_us'] == 11.0
    assert pq.count_superseded(pg_conn, arch=ARCH) == 0


def test_the_newest_of_two_dispatches_wins_and_the_older_is_counted_superseded(pg_conn):
    first = _task(pg_conn)
    pq.save_measurement(pg_conn, first.ref, _report(10.0), attempt=1)
    time.sleep(0.01)
    second = _task(pg_conn)
    pq.save_measurement(pg_conn, second.ref, _report(20.0), attempt=1)
    (row,) = _all(pg_conn)
    assert row['median_us'] == 20.0                 # never averaged
    assert pq.count_superseded(pg_conn, arch=ARCH) == 1


def test_statuses_without_statistics_have_null_times(pg_conn):
    task = _task(pg_conn)
    notok = _report(backend='aiter', status='NotOK', payload=None, error={'stdout': 'x'})
    crash = _report(backend='triton_split', status='crash', payload=None, error={'errno': 5})
    pq.save_measurement(pg_conn, task.ref, notok, attempt=1)
    pq.save_measurement(pg_conn, task.ref, crash, attempt=1)
    rows = {r['backend']: r for r in _all(pg_conn)}
    assert rows['aiter']['status'] == 'NotOK' and rows['aiter']['median_us'] is None
    assert rows['triton_split']['status'] == 'crash' and rows['triton_split']['error'] == {'errno': 5}


def test_the_variant_is_the_subpool_and_any_when_there_is_none(pg_conn):
    plain = _task(pg_conn)
    pq.save_measurement(pg_conn, plain.ref, _report(), attempt=1)
    named = _task(pg_conn, subpool='pool_a')
    pq.save_measurement(pg_conn, named.ref, _report(12.0), attempt=1)
    assert {r['variant'] for r in _all(pg_conn)} == {'any', 'pool_a'}
    (only,) = _all(pg_conn, variant='pool_a')
    assert only['median_us'] == 12.0
    # one row per variant: they never supersede each other
    assert pq.count_superseded(pg_conn, arch=ARCH) == 0


def test_list_selections_counts_by_status(pg_conn):
    task = _task(pg_conn)
    pq.save_measurement(pg_conn, task.ref, _report(), attempt=1)
    pq.save_measurement(pg_conn, task.ref, _report(backend='aiter', status='NotOK', payload=None), attempt=1)
    pq.save_measurement(pg_conn, task.ref, _report(backend='x', status='crash', payload=None), attempt=1)
    other = _task(pg_conn, subject=S14)
    pq.save_measurement(pg_conn, other.ref, _report(), attempt=1)
    rows = pq.list_selections(pg_conn, module='flash', arch=ARCH)
    assert [(r['subject'], r['measured'], r['ok'], r['notok'], r['crash']) for r in rows] == [
        (S13, 3, 1, 1, 1), (S14, 1, 1, 0, 0)]
    assert rows[0]['newest'] is not None
    assert [r['subject'] for r in pq.list_selections(pg_conn, subjects=[S14])] == [S14]
    assert pq.list_selections(pg_conn, arch='gfx000') == []


def test_iter_latest_filters_by_subject(pg_conn):
    pq.save_measurement(pg_conn, _task(pg_conn).ref, _report(), attempt=1)
    pq.save_measurement(pg_conn, _task(pg_conn, subject=S14).ref, _report(), attempt=1)
    assert {r['subject'] for r in _all(pg_conn, subjects=[S14])} == {S14}


def test_subject_progress_counts_tasks(pg_conn):
    task = _task(pg_conn)
    tasks.complete(pg_conn, task.ref, owner='w0')
    tasks.insert(pg_conn, ARCH, [TaskSpec({'module': 'flash', 'subject': S13, 'entry': _entry(16)}, CATEGORY)])
    (row,) = pq.subject_progress(pg_conn, arch=ARCH)
    assert (row['subject'], row['done'], row['pending'], row['total']) == (S13, 1, 1, 2)
