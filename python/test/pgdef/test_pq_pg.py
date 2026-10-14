# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.pgdef.pq against a real, disposable PostgreSQL server.

Marker `pg`: skipped unless AOTRITON_PG_TEST_DISPOSABLE=1 and a server answers
(see aotriton.pgdef.pq.testkit). These tests have NOT been run by the author:
the development sandbox had no PostgreSQL server. Expect to find a SQL typo.
"""

import threading
import time

import psycopg
import pytest
from psycopg.rows import dict_row

from aotriton.pgdef.pq import logistics as lg
from aotriton.pgdef.pq import progress, reports, tasks, testkit, workers
from aotriton.pgdef.pq.tasks import TaskSelector, TaskSpec

pytestmark = pytest.mark.pg

POOL = 'gfx950'


def _spec(n=0, category=('tune', 'kernel'), **kw):
    return TaskSpec({'n': n, 'module': 'flash'}, category, **kw)


def _row(conn, ref):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM pgdef.task WHERE id = %s AND pool = %s', ref)
        return cur.fetchone()


def _claim(conn, **kw):
    kw.setdefault('pool', POOL)
    kw.setdefault('category', ('tune', 'kernel'))
    kw.setdefault('subpool', None)
    kw.setdefault('worker', 'w0')
    return tasks.claim(conn, **kw)


@pytest.fixture
def pool(pg_conn):
    testkit.ensure_pool(pg_conn, POOL)
    return POOL


# --- partitions -------------------------------------------------------------

def test_ensure_pool_is_idempotent_and_listed(pg_conn):
    # tune's schema.sql (applied by the fixture) already creates its pools.
    assert {'gfx942', 'gfx950'} <= set(tasks.list_pools(pg_conn))
    testkit.ensure_pool(pg_conn, 'test_pool')
    testkit.ensure_pool(pg_conn, 'test_pool')
    assert 'test_pool' in tasks.list_pools(pg_conn)


def test_unknown_pool_fails_loudly_at_insert(pg_conn):
    with pytest.raises(psycopg.errors.CheckViolation):     # "no partition of relation found for row"
        tasks.insert(pg_conn, 'gfx000', [_spec()])


# --- claim ------------------------------------------------------------------

def test_claim_order_is_priority_then_id(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(0, priority=5), _spec(1, priority=9), _spec(2, priority=5)])
    got = _claim(pg_conn, limit=3)
    assert [t.payload['n'] for t in got] == [1, 0, 2]
    assert all(t.status == 'running' and t.logistics.attempt == 1 for t in got)
    assert got[0].logistics.lease.owner == 'w0'


def test_pair_rule_for_subpool(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(0), _spec(1, subpool='MI355'), _spec(2, subpool='MI350')])
    generic = _claim(pg_conn, subpool=None, limit=10)
    assert [t.payload['n'] for t in generic] == [0]
    pg_conn.execute('UPDATE pgdef.task SET status = %s', ('pending',))
    mi355 = _claim(pg_conn, subpool='MI355', limit=10)
    assert sorted(t.payload['n'] for t in mi355) == [0, 1]


def test_pair_rule_for_category_level1(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(0, ('tune', 'kernel')), _spec(1, ('tune', 'op')),
                                 _spec(2, ('tune', None)), _spec(3, ('other', None))])
    kernel = _claim(pg_conn, category=('tune', 'kernel'), limit=10)
    assert sorted(t.payload['n'] for t in kernel) == [0, 2]
    pg_conn.execute('UPDATE pgdef.task SET status = %s', ('pending',))
    generic = _claim(pg_conn, category=('tune', None), limit=10)
    assert [t.payload['n'] for t in generic] == [2]
    other = _claim(pg_conn, category=('other', None), limit=10)
    assert [t.payload['n'] for t in other] == [3]


def test_two_readers_never_claim_the_same_task(pg_conn, pg_conn2, pool):
    tasks.insert(pg_conn, pool, [_spec(i) for i in range(20)])
    a = _claim(pg_conn, worker='a', limit=7)
    b = _claim(pg_conn2, worker='b', limit=20)
    assert not ({t.id for t in a} & {t.id for t in b})
    assert len(a) + len(b) == 20


def test_identical_tasks_are_legal(pg_conn, pool):
    assert tasks.insert(pg_conn, pool, [_spec(0), _spec(0)]) == 2


# --- completion, failure, lease ---------------------------------------------

def test_complete(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn)
    assert tasks.complete(pg_conn, t.ref, owner='w0')
    row = _row(pg_conn, t.ref)
    assert row['status'] == 'done' and 'lease' not in row['logistics']
    assert row['logistics']['finished'] > 0
    assert not tasks.complete(pg_conn, t.ref, owner='w0')          # not running any more


def test_fail_retries_until_attempts_are_used_up(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(max_attempts=2)])
    (t,) = _claim(pg_conn)
    assert tasks.fail(pg_conn, t.ref, 'boom 1', owner='w0') == 'pending'
    (t,) = _claim(pg_conn)
    assert t.logistics.attempt == 2
    assert tasks.fail(pg_conn, t.ref, 'boom 2', owner='w0') == 'failed'
    row = _row(pg_conn, t.ref)
    assert [e['detail'] for e in row['logistics']['errors']] == ['boom 1', 'boom 2']
    assert _claim(pg_conn) == []


def test_permanent_failure_does_not_retry(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn)
    assert tasks.fail(pg_conn, t.ref, 'bad input', owner='w0', permanent=True) == 'failed'


def test_error_history_is_capped(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(max_attempts=100)])
    for i in range(lg.ERROR_HISTORY_CAP + 4):
        (t,) = _claim(pg_conn)
        tasks.fail(pg_conn, t.ref, f'e{i}', owner='w0')
    errors = _row(pg_conn, (t.id, t.pool))['logistics']['errors']
    assert len(errors) == lg.ERROR_HISTORY_CAP
    assert errors[-1]['detail'] == f'e{lg.ERROR_HISTORY_CAP + 3}'      # newest last


def test_renew_extends_the_lease_for_the_owner_only(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn, lease_s=10)
    before = _row(pg_conn, t.ref)['logistics']['lease']['until']
    assert not tasks.renew(pg_conn, t.ref, worker='someone-else', lease_s=1000)
    assert tasks.renew(pg_conn, t.ref, worker='w0', lease_s=1000)
    assert _row(pg_conn, t.ref)['logistics']['lease']['until'] > before + 900


def test_expired_lease_is_reaped_back_to_pending_and_keeps_the_appcookie(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn, lease_s=-1)                       # already expired
    tasks.append_cookie(pg_conn, t.ref, 'done', [1])
    assert progress.expired_leases(pg_conn)[0]['expired_count'] == 1
    assert tasks.reap_expired(pg_conn, pool) == [(t.id, 'pending')]
    row = _row(pg_conn, t.ref)
    assert row['status'] == 'pending' and row['appcookie'] == {'done': [1]}
    assert row['logistics']['errors'][0]['reason'] == 'lease_expired'
    (again,) = _claim(pg_conn, worker='w1')
    assert again.logistics.attempt == 2 and again.appcookie == {'done': [1]}


def test_a_stolen_lease_cannot_be_settled_by_its_old_owner(pg_conn, pool):
    # w0's lease expired and w1 claimed the task again: w0's graph, still
    # running, must not settle w1's claim. owner=None means "don't care".
    tasks.insert(pg_conn, pool, [_spec(max_attempts=2)])
    (t,) = _claim(pg_conn, lease_s=-1)
    tasks.reap_expired(pg_conn, pool)
    (t,) = _claim(pg_conn, worker='w1')
    assert not tasks.complete(pg_conn, t.ref, owner='w0')
    assert tasks.fail(pg_conn, t.ref, 'late', owner='w0') is None
    assert not tasks.release(pg_conn, t.ref, owner='w0')
    assert _row(pg_conn, t.ref)['status'] == 'running'
    assert tasks.complete(pg_conn, t.ref, owner=None)


def test_reaping_fails_a_task_that_is_out_of_attempts(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(max_attempts=1)])
    (t,) = _claim(pg_conn, lease_s=-1)
    assert tasks.reap_expired(pg_conn, pool) == [(t.id, 'failed')]


def test_reap_leaves_live_leases_alone(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    _claim(pg_conn, lease_s=600)
    assert tasks.reap_expired(pg_conn, pool) == []


def test_release_does_not_consume_the_attempt(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(max_attempts=1)])
    (t,) = _claim(pg_conn)
    assert tasks.release(pg_conn, t.ref, owner='w0')
    (t,) = _claim(pg_conn)
    assert t.logistics.attempt == 1


# --- appcookie --------------------------------------------------------------

def test_append_cookie_creates_and_appends(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn)
    assert _row(pg_conn, t.ref)['appcookie'] is None
    tasks.append_cookie(pg_conn, t.ref, 'impls', [1])
    tasks.append_cookie(pg_conn, t.ref, 'impls', [2, 3])
    tasks.append_cookie(pg_conn, t.ref, 'other', ['x'])
    assert _row(pg_conn, t.ref)['appcookie'] == {'impls': [1, 2, 3], 'other': ['x']}


def test_concurrent_appends_lose_nothing(pg_conn, pg_params, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn)

    def worker(base):
        conn = psycopg.connect(**pg_params, autocommit=True)
        for i in range(25):
            tasks.append_cookie(conn, t.ref, 'done', [base + i])
        conn.close()

    threads = [threading.Thread(target=worker, args=(b * 100,)) for b in range(4)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    done = _row(pg_conn, t.ref)['appcookie']['done']
    assert sorted(done) == sorted(b * 100 + i for b in range(4) for i in range(25))


def test_replace_cookie(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec()])
    (t,) = _claim(pg_conn)
    tasks.append_cookie(pg_conn, t.ref, 'impls', [1])
    tasks.replace_cookie(pg_conn, t.ref, {'compact': True})
    assert _row(pg_conn, t.ref)['appcookie'] == {'compact': True}
    tasks.replace_cookie(pg_conn, t.ref, None)
    assert _row(pg_conn, t.ref)['appcookie'] is None


# --- reports, reset, cascades -----------------------------------------------

def _finish_with_report(conn, pool, **kw):
    tasks.insert(conn, pool, [_spec(**kw)])
    (t,) = _claim(conn)
    rid = reports.insert(conn, t.ref, cat_level0='tune', attempt=1,
                         rkey={'iface': 'attn_fwd', 'impl': 0}, status='OK', payload={'t': 1.5})
    tasks.complete(conn, t.ref, owner='w0')
    return t, rid


def test_reports_roundtrip(pg_conn, pool):
    t, rid = _finish_with_report(pg_conn, pool)
    (r,) = reports.for_task(pg_conn, t.ref)
    assert r['id'] == rid and r['payload'] == {'t': 1.5} and r['rkey']['iface'] == 'attn_fwd'
    assert reports.for_task(pg_conn, t.ref, rkey_contains={'iface': 'other'}) == []


def test_reset_clears_appcookie_and_purges_reports(pg_conn, pool):
    t, _ = _finish_with_report(pg_conn, pool)
    tasks.append_cookie(pg_conn, t.ref, 'k', [1])
    n = tasks.reset(pg_conn, TaskSelector(pool=pool, ids=(t.id,)), purge_reports=True)
    assert n == 1
    row = _row(pg_conn, t.ref)
    assert row['status'] == 'pending' and row['appcookie'] is None
    assert row['logistics'] == {'v': 1}
    assert reports.for_task(pg_conn, t.ref) == []


def test_reset_can_keep_reports_and_appcookie_and_the_retry_policy(pg_conn, pool):
    t, _ = _finish_with_report(pg_conn, pool, max_attempts=7)
    tasks.append_cookie(pg_conn, t.ref, 'k', [1])
    tasks.reset(pg_conn, TaskSelector(ids=(t.id,)), purge_reports=False, keep_appcookie=True)
    row = _row(pg_conn, t.ref)
    assert row['appcookie'] == {'k': [1]} and row['logistics'] == {'v': 1, 'retry': {'max_attempts': 7}}
    assert len(reports.for_task(pg_conn, t.ref)) == 1


def test_deleting_a_task_cascades_to_its_reports(pg_conn, pool):
    t, _ = _finish_with_report(pg_conn, pool)
    assert tasks.delete(pg_conn, TaskSelector(ids=(t.id,))) == 1
    assert reports.for_task(pg_conn, t.ref) == []


def test_cancel(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(0), _spec(1)])
    (t,) = _claim(pg_conn)
    assert tasks.cancel(pg_conn, TaskSelector(pool=pool)) == 2
    assert _row(pg_conn, t.ref)['status'] == 'cancelled' and _claim(pg_conn) == []


def test_selectors_find_by_payload(pg_conn, pool):
    tasks.insert(pg_conn, pool, [
        TaskSpec({'module': 'flash', 'entry': {'hdim': [64, 128], 'bf': True, 'seq': 128}}, ('tune', 'kernel')),
        TaskSpec({'module': 'flash', 'entry': {'hdim': [128, 64], 'bf': True, 'seq': 128}}, ('tune', 'kernel')),
    ])
    contains = tasks.find(pg_conn, TaskSelector(payload_contains={'entry': {'bf': True, 'seq': 128}}))
    assert len(contains) == 2
    exact = tasks.find(pg_conn, TaskSelector(payload_exact={('entry', 'hdim'): [64, 128]}))
    assert [t.payload['entry']['hdim'] for t in exact] == [[64, 128]]     # order matters


# --- views and workers ------------------------------------------------------

def test_views(pg_conn, pool):
    tasks.insert(pg_conn, pool, [_spec(0), _spec(1, ('tune', 'op')), _spec(2, ('other', None))])
    (t,) = _claim(pg_conn)
    tasks.complete(pg_conn, t.ref, owner='w0')
    rows = progress.queue_progress(pg_conn, ('tune', 'kernel'))
    assert rows[0]['done'] == 1 and rows[0]['total'] == 1
    assert len(progress.queue_progress(pg_conn)) == 3
    assert progress.recent_completions(pg_conn, ('tune', 'kernel'))[0]['recent_completions'] == 1
    with pg_conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM pgdef.task_status WHERE id = %s', (t.id,))
        s = cur.fetchone()
        assert s['attempt'] == 1 and s['finished_at'] is not None
        for view in ('task_timing_stats', 'completion_eta', 'expired_leases'):
            cur.execute(f'SELECT * FROM pgdef.{view}')
            cur.fetchall()


def test_worker_upsert_and_health(pg_conn, pool):
    workers.upsert(pg_conn, worker_id='h/r0', host='h', pool=pool, subpool='MI355',
                   category=('tune', 'kernel'))
    workers.upsert(pg_conn, worker_id='h/r0', host='h', pool=pool, subpool='MI355',
                   category=('tune', 'kernel'), current_task=5)
    (w,) = workers.list_health(pg_conn)
    assert w['current_task'] == 5 and w['health_status'] == 'healthy'
    workers.remove(pg_conn, 'h/r0')
    assert workers.list_health(pg_conn) == []


def test_schema_is_idempotent(pg_conn):
    from aotriton.pgdef.pq import admin
    for path in admin.all_schema_files():
        admin.apply_schema(pg_conn, path)


def test_nothing_in_public(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_tables WHERE schemaname = 'public'")
        assert cur.fetchone()[0] == 0
