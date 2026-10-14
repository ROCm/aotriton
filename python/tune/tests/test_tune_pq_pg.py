# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.tune.pq against a real, disposable PostgreSQL server.

Marker `pg`: skipped unless AOTRITON_PG_TEST_DISPOSABLE=1 and a server answers
(see aotriton.pgdef.pq.testkit). NOT run by the author: the development sandbox
had no PostgreSQL server.
"""

import pytest

pytest.importorskip('psycopg')

from psycopg.rows import dict_row

from aotriton.pgdef.pq import reports as pgreports
from aotriton.pgdef.pq import tasks as pgtasks
from aotriton.pgdef.pq.tasks import TaskSelector, TaskSpec
from aotriton.tune.pq import extra_uts, results
from aotriton.tune.pq import tasks as tune

pytestmark = pytest.mark.pg

ARCH = 'gfx950'


def _entry(hdim=(64, 128), seq=128, causal=True):
    return {'dtype': 'float16', 'hdim': list(hdim), 'seqlen_q': seq, 'causal': causal}


def _dispatch(conn, entries, level='kernel', arch=ARCH):
    # tune's schema.sql creates the partitions of every pool tune anticipates.
    return pgtasks.insert(conn, arch, [
        TaskSpec({'module': 'flash', 'entry': e}, tune.category(level)) for e in entries])


def _report(task_ref, iface='attn_fwd', impl=0, result='OK', level='kernel', data=None):
    return {'tuning_level': level, 'iface_name': iface, 'impl_index': impl, 'result': result,
            'result_data': data if data is not None else {'times': [1.0], 'adiffs': {'t': {'dq': [1e-3, 2e-3]}}},
            'error': None, 'complete_on_gpu': 3}


def _claim_all(conn, level='kernel', n=100):
    return pgtasks.claim(conn, pool=ARCH, category=tune.category(level), subpool=None,
                         worker='h/r0', limit=n)


def test_find_by_entry_distinguishes_tuple_order_and_level(pg_conn):
    _dispatch(pg_conn, [_entry((64, 128)), _entry((128, 64))])
    _dispatch(pg_conn, [_entry((64, 128))], level='op')
    rows = tune.find_by_entry(pg_conn, _entry((64, 128)), arch=ARCH, tuning_level='kernel', module='flash')
    assert len(rows) == 1 and rows[0]['arch'] == ARCH and rows[0]['tuning_level'] == 'kernel'
    both = tune.find_by_entry(pg_conn, _entry((64, 128)))
    assert sorted(r['tuning_level'] for r in both) == ['kernel', 'op']


def test_results_roundtrip_through_the_typed_view(pg_conn):
    _dispatch(pg_conn, [_entry()])
    (t,) = _claim_all(pg_conn)
    results.save_tuning_result(pg_conn, t.ref, _report(t.ref, impl=2), attempt=1)
    results.save_tuning_result(pg_conn, t.ref, _report(t.ref, impl=0, result='crash', data={}), attempt=1)
    got = results.get_task_results(pg_conn, t.id)
    assert [(r['iface_name'], r['impl_index'], r['result']) for r in got] == [
        ('attn_fwd', 0, 'crash'), ('attn_fwd', 2, 'OK')]
    assert got[0]['worker'] == 'gpu-3'
    assert results.get_task_results(pg_conn, t.id, tuning_level='op') == []
    with pg_conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT arch, cat_level1, iface_name, impl_index FROM tune.tuning_results WHERE task_id = %s',
                    (t.id,))
        assert {(r['arch'], r['cat_level1']) for r in cur.fetchall()} == {(ARCH, 'kernel')}


def test_reset_with_delete_results_cascades_into_most_accurate(pg_conn):
    _dispatch(pg_conn, [_entry()])
    (t,) = _claim_all(pg_conn)
    rid = results.save_tuning_result(pg_conn, t.ref, _report(t.ref), attempt=1)
    pgtasks.complete(pg_conn, t.ref, owner='h/r0')
    with pg_conn.cursor() as cur:
        cur.execute(
            'INSERT INTO tune.most_accurate_tuning_results '
            '(report_id, task_id, arch, cat_level1, payload, iface_name, test_case, tensor_name) '
            "VALUES (%s, %s, %s, 'kernel', '{}'::jsonb, 'attn_fwd', '00_basic', 'dq')", (rid, t.id, ARCH))
    assert tune.reset(pg_conn, [t.id], 'kernel', delete_results=True) == 1
    with pg_conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM tune.most_accurate_tuning_results')
        assert cur.fetchone()[0] == 0
        cur.execute('SELECT count(*) FROM tune.tuning_results')
        assert cur.fetchone()[0] == 0


def test_reset_is_scoped_by_level(pg_conn):
    _dispatch(pg_conn, [_entry()], level='kernel')
    _dispatch(pg_conn, [_entry()], level='op')
    ids = [r['id'] for r in tune.find_by_entry(pg_conn, _entry())]
    assert tune.reset(pg_conn, ids, 'op', delete_results=False) == 1


def test_extra_uts_survive_reset_and_die_with_the_task(pg_conn):
    _dispatch(pg_conn, [_entry()])
    (t,) = _claim_all(pg_conn)
    extra_uts.insert_extra_uts(pg_conn, t.ref, ['a=1', 'b=2'])
    extra_uts.insert_extra_uts(pg_conn, t.ref, ['a=1'])          # upsert, no duplicate
    pgtasks.complete(pg_conn, t.ref, owner='h/r0')
    tune.reset(pg_conn, [t.id], 'kernel', delete_results=True)
    assert extra_uts.get_extra_uts(pg_conn, t.id) == ['a=1', 'b=2']
    pgtasks.delete(pg_conn, TaskSelector(ids=(t.id,)))
    assert extra_uts.get_extra_uts(pg_conn, t.id) == []


def test_find_resettable_and_progress(pg_conn):
    _dispatch(pg_conn, [_entry(seq=s) for s in (1, 2, 3)])
    t1, t2, t3 = _claim_all(pg_conn)
    pgtasks.complete(pg_conn, t1.ref, owner='h/r0')
    pgtasks.fail(pg_conn, t2.ref, 'x', owner='h/r0', permanent=True)
    failed = tune.find_resettable(pg_conn, 'failed')
    assert [r['id'] for r in failed] == [t2.id]
    assert tune.find_resettable(pg_conn, 'stale', timeout_seconds=3600) == []
    stale = tune.find_resettable(pg_conn, 'stale', timeout_seconds=-1)
    assert [r['id'] for r in stale] == [t3.id] and stale[0]['lease_owner'] == 'h/r0'
    assert [r['id'] for r in tune.find_resettable_by_ids(pg_conn, [t1.id, 10**9])] == [t1.id]
    prog = tune.get_progress(pg_conn, 'kernel')
    (row,) = prog['progress']
    assert (row['arch'], row['done'], row['failed'], row['running'], row['total']) == (ARCH, 1, 1, 1, 3)
    assert prog['speed'][0]['recent_completions'] == 1


def test_completed_payloads_for_skip_completed(pg_conn):
    _dispatch(pg_conn, [_entry(seq=1), _entry(seq=2)])
    t1, t2 = _claim_all(pg_conn)
    pgtasks.complete(pg_conn, t1.ref, owner='h/r0')
    done = tune.completed_payloads(pg_conn, 'flash', 'kernel')
    assert [(a, p['entry']['seqlen_q']) for a, p in done] == [(ARCH, 1)]


def test_debug_snapshot(pg_conn):
    _dispatch(pg_conn, [_entry()])
    (t,) = _claim_all(pg_conn)
    results.save_tuning_result(pg_conn, t.ref, _report(t.ref), attempt=1)
    snap = results.get_task_debug_snapshot(pg_conn, t.id)
    assert snap['task']['status'] == 'running' and snap['task']['payload']['module'] == 'flash'
    assert len(snap['tuning_results']) == 1 and snap['optune_results'] == []
