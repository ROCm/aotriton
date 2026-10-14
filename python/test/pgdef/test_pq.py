# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-only tests of aotriton.pgdef.pq: pure logic and SQL generation.

Nothing here talks to PostgreSQL. test_pq_pg.py (marker `pg`) does.
"""

import pathlib

import pytest

from aotriton.pgdef.pq import category as cat
from aotriton.pgdef.pq import conn as pqconn
from aotriton.pgdef.pq import logistics as lg
from aotriton.pgdef.pq import progress, reports, tasks, workers
from aotriton.pgdef.pq.tasks import TaskSelector, TaskSpec

from pgfake import FakeConn, check_statement


# ---------------------------------------------------------------------------
# category
# ---------------------------------------------------------------------------

def test_category_roundtrip():
    assert cat.parse_category('tune,kernel') == ('tune', 'kernel')
    assert cat.parse_category('other') == ('other', None)
    assert cat.format_category(('tune', 'op')) == 'tune,op'
    assert cat.format_category(('other', None)) == 'other'
    assert cat.to_columns(('tune', 'op')) == {'cat_level0': 'tune', 'cat_level1': 'op'}
    assert cat.to_columns(('other', None)) == {'cat_level0': 'other', 'cat_level1': None}


@pytest.mark.parametrize('bad', ['', ',kernel', 'tune,', 'a,b,c', 'tu ne', 'tune;drop', 'tune,,x'])
def test_category_rejects(bad):
    with pytest.raises(ValueError):
        cat.parse_category(bad)


def test_claim_terms_one_term_per_level():
    two = cat.claim_terms(2)
    assert two == 'cat_level0 = %(cat_level0)s AND (cat_level1 IS NULL OR cat_level1 = %(cat_level1)s)'
    three = cat.claim_terms(3)
    assert '(cat_level2 IS NULL OR cat_level2 = %(cat_level2)s)' in three
    assert cat.claim_terms(1) == 'cat_level0 = %(cat_level0)s'


# ---------------------------------------------------------------------------
# logistics accessor
# ---------------------------------------------------------------------------

def test_logistics_defaults_for_absent_keys():
    lgx = lg.Logistics.from_json(None)
    assert (lgx.attempt, lgx.max_attempts, lgx.errors) == (0, lg.DEFAULT_MAX_ATTEMPTS, ())
    assert lgx.lease.owner is None and not lgx.lease_expired(1e12)
    assert lgx.attempts_left == lg.DEFAULT_MAX_ATTEMPTS
    assert lg.Logistics.from_json({'v': 1}) == lgx


def test_logistics_reads_a_full_document_and_ignores_unknown_keys():
    doc = {'v': 1, 'attempt': 2, 'retry': {'max_attempts': 5},
           'lease': {'owner': 'n/r0', 'until': 100.0, 'renewed': 90.0},
           'started': 50.0, 'finished': None,
           'errors': [{'attempt': 1, 'reason': 'lease_expired'}],
           'not_before': 1.0, 'future_field': {'x': 1}}
    x = lg.Logistics.from_json(doc)
    assert (x.attempt, x.max_attempts, x.attempts_left) == (2, 5, 3)
    assert x.lease.owner == 'n/r0' and x.lease_expired(100.5) and not x.lease_expired(99.0)
    assert x.errors[0]['reason'] == 'lease_expired'


def test_logistics_upgrades_on_read():
    assert lg.Logistics.from_json({'v': 0, 'attempt': 1}).v == lg.VERSION


def test_initial_json():
    assert lg.initial_json() == {'v': 1}
    assert lg.initial_json(4) == {'v': 1, 'retry': {'max_attempts': 4}}


# ---------------------------------------------------------------------------
# SQL text
# ---------------------------------------------------------------------------

def test_claim_sql_shape():
    q = lg.claim_sql()
    check_statement(q, {'worker': 1, 'lease_s': 1, 'pool': 1, 'limit': 1, 'subpool': 1,
                        'cat_level0': 1, 'cat_level1': 1})
    # targets the PARENT table and prunes by pool
    assert 'UPDATE pgdef.task t' in q
    assert q.count('pool = %(pool)s') == 2
    assert 'FOR UPDATE SKIP LOCKED' in q
    assert 'ORDER BY priority DESC, id' in q
    assert "status = 'pending'" in q
    assert '(subpool IS NULL OR subpool = %(subpool)s)' in q
    assert '(cat_level1 IS NULL OR cat_level1 = %(cat_level1)s)' in q
    # no leaf table name is ever built
    assert 'task_' not in q.replace('task_claim', '')


def test_claim_sql_grows_with_levels():
    assert '(cat_level2 IS NULL OR cat_level2 = %(cat_level2)s)' in lg.claim_sql(3)


def test_every_statement_is_well_formed():
    sels = TaskSelector(pool='gfx950', status='failed').where('t')
    statements = {
        'renew': lg.renew_sql(), 'fail': lg.fail_sql(), 'reap': lg.reap_sql(),
        'release': lg.release_sql(), 'complete': lg.complete_sql(),
        'reset_purge': lg.reset_sql(sels[0], purge_reports=True, keep_appcookie=False),
        'reset_keep': lg.reset_sql(sels[0], purge_reports=False, keep_appcookie=True),
        'cancel': lg.cancel_sql(sels[0]),
    }
    for name, q in statements.items():
        stripped = q.replace("'{}'", '')
        assert stripped.count('(') == stripped.count(')'), name
        assert q.count("'") % 2 == 0, name


def test_logistics_is_the_only_writer_of_the_column():
    """No statement outside logistics.py may assign `logistics =`."""
    import pathlib
    pq = pathlib.Path(tasks.__file__).parent
    for path in pq.glob('*.py'):
        if path.name == 'logistics.py':
            continue
        text = path.read_text()
        assert 'logistics =' not in text.replace('self.logistics', ''), path.name
        assert 'SET logistics' not in text, path.name


def test_reap_and_fail_cap_the_error_history():
    assert f'LIMIT {lg.ERROR_HISTORY_CAP - 1}' in lg.fail_sql()
    assert f'LIMIT {lg.ERROR_HISTORY_CAP - 1}' in lg.reap_sql()


def test_reset_purges_in_the_same_statement():
    q = lg.reset_sql('TRUE', purge_reports=True, keep_appcookie=False)
    assert 'DELETE FROM pgdef.report' in q and 'appcookie = NULL' in q
    assert q.index('DELETE FROM pgdef.report') < q.index('UPDATE pgdef.task')
    q = lg.reset_sql('TRUE', purge_reports=False, keep_appcookie=True)
    assert 'DELETE FROM pgdef.report' not in q and 'appcookie' not in q


def test_reset_keeps_the_retry_policy_only():
    q = lg.reset_sql('TRUE', purge_reports=False, keep_appcookie=False)
    assert "t.logistics->'retry'" in q
    assert "'attempt'" not in q and "'lease'" not in q


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------

def test_ensure_pool_is_the_engines_partition_api():
    # Partitions are created by the SQL function pgdef.ensure_pool(), which
    # applications call from their own schema.sql. It validates the name and
    # quotes the leaf with %I, so a pool name can never inject SQL.
    sql_text = (pathlib.Path(tasks.__file__).parent / 'schema.sql').read_text()
    fn = sql_text[sql_text.index('CREATE OR REPLACE FUNCTION pgdef.ensure_pool'):]
    assert "pool !~ '^[a-z0-9_]+$'" in fn
    assert "'task_' || pool" in fn and "'report_' || pool" in fn
    assert fn.count('CREATE TABLE IF NOT EXISTS pgdef.%I PARTITION OF') == 2


@pytest.mark.parametrize('bad', ['', 'GFX950', 'gfx950; DROP TABLE pgdef.task', 'a b', "x'y", 'gfx-950', None])
def test_unsafe_pool_names_are_rejected(bad):
    with pytest.raises(ValueError):
        tasks.validate_pool(bad)


# ---------------------------------------------------------------------------
# selector
# ---------------------------------------------------------------------------

def test_selector_where():
    sel = TaskSelector(pool='gfx950', category=('tune', 'op'), status=('failed', 'running'),
                       ids=(3, 4), payload_contains={'module': 'flash'},
                       payload_exact={('entry', 'hdim'): [64, 128]})
    sql, params = sel.where('t')
    check_statement(sql, params)
    assert sql == ('t.pool = %s AND t.cat_level0 = %s AND t.cat_level1 = %s AND t.status = ANY(%s) '
                   'AND t.id = ANY(%s) AND t.payload @> %s AND t.payload #> %s = %s')
    assert params[:3] == ['gfx950', 'tune', 'op']
    assert params[3] == ['failed', 'running'] and params[4] == [3, 4]
    assert params[5].obj == {'module': 'flash'}
    assert params[6] == ['entry', 'hdim'] and params[7].obj == [64, 128]


def test_selector_none_level_is_not_filtered():
    sql, params = TaskSelector(category=('tune', None)).where('t')
    assert sql == 't.cat_level0 = %s' and params == ['tune']


def test_empty_selector():
    assert TaskSelector().is_empty()
    assert TaskSelector(category=(None, None)).is_empty()
    assert TaskSelector(payload_contains={}).is_empty()
    assert not TaskSelector(ids=(1,)).is_empty()
    assert TaskSelector().where('t') == ('TRUE', [])


@pytest.mark.parametrize('op', [
    lambda c, s: tasks.reset(c, s, purge_reports=True),
    lambda c, s: tasks.cancel(c, s),
    lambda c, s: tasks.delete(c, s),
])
def test_destructive_operations_refuse_an_empty_selector(op):
    conn = FakeConn()
    with pytest.raises(ValueError):
        op(conn, TaskSelector())
    assert conn.calls == []


def test_pool_in_selector_is_validated():
    with pytest.raises(ValueError):
        TaskSelector(pool='x; drop').where('t')


# ---------------------------------------------------------------------------
# functions against a recording connection
# ---------------------------------------------------------------------------

def _task_row(**kw):
    row = dict(id=7, pool='gfx950', subpool=None, cat_level0='tune', cat_level1='kernel',
               payload={'module': 'flash'}, appcookie=None, priority=5, status='running',
               logistics={'v': 1, 'attempt': 1}, created_at=None)
    row.update(kw)
    return row


def test_claim_passes_every_placeholder_and_orders_the_result():
    conn = FakeConn(rows=[[_task_row(id=9, priority=5), _task_row(id=8, priority=7),
                           _task_row(id=3, priority=5)]])
    got = tasks.claim(conn, pool='gfx950', category=('tune', 'kernel'), subpool='MI355',
                      worker='n/r0', lease_s=120, limit=3)
    (text, params), = conn.calls
    check_statement(text, params)
    assert params == {'pool': 'gfx950', 'subpool': 'MI355', 'worker': 'n/r0', 'lease_s': 120.0,
                      'limit': 3, 'cat_level0': 'tune', 'cat_level1': 'kernel'}
    assert [t.id for t in got] == [8, 3, 9]          # priority DESC, id
    assert got[0].ref == (8, 'gfx950') and got[0].category == ('tune', 'kernel')
    assert got[0].logistics.attempt == 1


def test_claim_with_a_generic_reader():
    conn = FakeConn(rows=[[]])
    assert tasks.claim(conn, pool='gfx950', category=('other', None), subpool=None, worker='w') == []
    assert conn.calls[0][1]['subpool'] is None and conn.calls[0][1]['cat_level1'] is None


def test_complete_fail_release_renew_reap_use_their_statements():
    conn = FakeConn(rows=[[], [], [], [('pending',)], [(5, 'failed'), (6, 'pending')]])
    assert tasks.complete(conn, (1, 'gfx950'), owner='w') is True
    assert tasks.renew(conn, (1, 'gfx950'), worker='w') is True
    assert tasks.release(conn, (1, 'gfx950'), owner='w') is True
    assert tasks.fail(conn, (1, 'gfx950'), 'boom', owner='w', permanent=True) == 'pending'
    assert tasks.reap_expired(conn, 'gfx950') == [(5, 'failed'), (6, 'pending')]
    for text, params in conn.calls:
        check_statement(text, params)
    fail_params = conn.calls[3][1]
    assert fail_params['permanent'] is True and fail_params['detail'] == 'boom'
    assert fail_params['id'] == 1 and fail_params['pool'] == 'gfx950'


def test_fail_of_a_task_that_is_not_running_returns_none():
    conn = FakeConn(rows=[[]])
    assert tasks.fail(conn, (1, 'gfx950'), 'x', owner=None) is None


def test_renew_reports_a_lost_lease():
    conn = FakeConn(rowcount=0)
    assert tasks.renew(conn, (1, 'gfx950'), worker='w') is False


def test_append_cookie_is_one_sql_append():
    conn = FakeConn()
    tasks.append_cookie(conn, (4, 'gfx950'), 'impls_done', [1, 2])
    (text, params), = conn.calls
    check_statement(text, params)
    assert params['key'] == 'impls_done' and params['values'].obj == [1, 2]
    assert "COALESCE(appcookie, '{}'::jsonb)" in text
    assert "|| %(values)s::jsonb" in text
    assert '||' in text and "appcookie ||" not in text      # never a merge


def test_replace_cookie():
    conn = FakeConn()
    tasks.replace_cookie(conn, (4, 'gfx950'), {'a': 1})
    tasks.replace_cookie(conn, (4, 'gfx950'), None)
    assert conn.calls[0][1][0].obj == {'a': 1} and conn.calls[1][1][0] is None
    for text, params in conn.calls:
        check_statement(text, list(params))


def test_insert_stamps_logistics_and_category():
    conn = FakeConn()
    n = tasks.insert(conn, 'gfx950', [
        TaskSpec({'module': 'flash'}, ('tune', 'kernel'), max_attempts=4),
        TaskSpec({'module': 'flash'}, ('other', None), subpool='MI355', priority=9),
    ], batch_size=10)
    assert n == 2
    (text, batch), = conn.calls
    check_statement(text, batch)
    a, b = batch
    assert a['logistics'].obj == {'v': 1, 'retry': {'max_attempts': 4}}
    assert (a['cat_level0'], a['cat_level1'], a['subpool']) == ('tune', 'kernel', None)
    assert (b['cat_level0'], b['cat_level1'], b['subpool'], b['priority']) == ('other', None, 'MI355', 9)


def test_insert_validates_pool_and_category():
    with pytest.raises(ValueError):
        tasks.insert(FakeConn(), 'Bad Pool', [])
    with pytest.raises(ValueError):
        tasks.insert(FakeConn(), 'gfx950', [TaskSpec({}, ('tune', None, 'x'))])


def test_identical_tasks_are_legal():
    conn = FakeConn()
    spec = TaskSpec({'module': 'flash', 'entry': {'a': 1}}, ('tune', 'kernel'))
    assert tasks.insert(conn, 'gfx950', [spec, spec]) == 2


def test_reset_statement_and_params():
    conn = FakeConn(rowcount=3)
    n = tasks.reset(conn, TaskSelector(pool='gfx950', ids=(1, 2, 3)), purge_reports=True)
    assert n == 3
    (text, params), = conn.calls
    check_statement(text, params)
    assert params == ['gfx950', [1, 2, 3]]
    assert 'DELETE FROM pgdef.report' in text


def test_find_and_count_build_where():
    conn = FakeConn(rows=[[_task_row()], [(2,)]])
    found = tasks.find(conn, TaskSelector(pool='gfx950', status='failed'), limit=5)
    assert [t.id for t in found] == [7]
    assert tasks.count(conn, TaskSelector(pool='gfx950')) == 2
    for text, params in conn.calls:
        check_statement(text, params)
    assert conn.calls[0][1] == ['gfx950', ['failed'], 5]


def test_report_insert_and_workers_and_progress_statements():
    conn = FakeConn(rows=[[(11,)]])
    rid = reports.insert(conn, (1, 'gfx950'), cat_level0='tune', attempt=2,
                         rkey={'iface': 'attn_fwd', 'impl': 3}, status='OK', payload={'t': 1.0})
    assert rid == 11
    workers.upsert(conn, worker_id='h/r0', host='h', pool='gfx950', subpool=None,
                   category=('tune', 'kernel'), current_task=1)
    workers.remove(conn, 'h/r0')
    progress.queue_progress(conn, ('tune', None))
    progress.recent_completions(conn, ('tune', 'op'), window_s=60)
    progress.expired_leases(conn)
    for text, params in conn.calls:
        check_statement(text, params if not isinstance(params, tuple) else list(params))
    assert 'cat_level0 = %s' in conn.calls[3][0]


# ---------------------------------------------------------------------------
# connection parameters
# ---------------------------------------------------------------------------

def test_params_from_env_has_no_dbname_ever():
    p = pqconn.params_from_env({'PGHOST': 'db', 'PGPORT': '6543', 'PGUSER': 'u', 'PGPASSWORD': 'pw',
                                'PGDATABASE': 'oops'})
    assert p == {'host': 'db', 'port': 6543, 'user': 'u', 'password': 'pw'}
    assert 'dbname' not in p and 'database' not in p


def test_params_from_env_falls_back_to_config_rc_names():
    p = pqconn.params_from_env({'CELERY_SERVICE_HOST': 'h', 'POSTGRES_PORT': '5000',
                                'POSTGRES_USER': 'tuner', 'POSTGRES_PASSWORD': 'x'})
    assert p == {'host': 'h', 'port': 5000, 'user': 'tuner', 'password': 'x'}
    assert pqconn.params_from_env({}) == {'host': 'localhost', 'port': 5432,
                                          'user': 'aotriton', 'password': None}


def test_schema_sql_is_plain_and_idempotent():
    from aotriton.pgdef.pq import admin
    import re
    text = re.sub(r'--[^\n]*', '', admin.schema_path().read_text())
    assert '\\' not in text.replace('\\n', ''), 'psql meta-command in schema.sql'
    # The engine creates no partition itself: no DEFAULT one, and pool leaves
    # only through pgdef.ensure_pool(), which applications call.
    static = re.sub(r'CREATE OR REPLACE FUNCTION pgdef\.ensure_pool.*?\$\$;', '', text, flags=re.S)
    assert 'DEFAULT PARTITION' not in static.upper() and 'FOR VALUES' not in static.upper()
    assert 'CREATE OR REPLACE FUNCTION pgdef.ensure_pool' in text
    for stmt in re.findall(r'^CREATE (?:TABLE|INDEX|SCHEMA)[^;]*;', text, flags=re.M):
        assert 'IF NOT EXISTS' in stmt, stmt
    assert 'public.' not in text
    assert 'PARTITION BY LIST (pool)' in text
    assert 'ON DELETE CASCADE' in text


# ---------------------------------------------------------------------------
# reports.insert(replace=True)
# ---------------------------------------------------------------------------

def test_report_replace_deletes_the_same_rkey_in_the_same_statement():
    conn = FakeConn(rows=[[(1,)], [(5,)], [(6,)]])
    reports.insert(conn, (1, 'gfx950'), cat_level0='tune', attempt=2,
                   rkey={'iface': 'a', 'impl': 0}, status='OK', replace=True)
    reports.insert(conn, (1, 'gfx950'), cat_level0='tune', attempt=1,
                   rkey={'iface': 'a', 'impl': 0}, status='OK')
    (lock, lock_params), (text, params), (plain, plain_params) = conn.calls
    # Replacements are serialized by locking the task row, in one transaction.
    assert conn.transactions == 1
    assert 'FROM pgdef.task WHERE id = %s AND pool = %s FOR UPDATE' in lock
    assert lock_params == (1, 'gfx950')
    check_statement(text, list(params))
    assert text.startswith('WITH replaced AS (DELETE FROM pgdef.report') and 'rkey = %s' in text
    assert 'DELETE' not in plain
    assert params[3].obj == {'iface': 'a', 'impl': 0} and params[0:3] == (1, 'gfx950', 'tune')


# ---------------------------------------------------------------------------
# dispatch driver
# ---------------------------------------------------------------------------

class _Source:
    default_category = ('toy', None)

    def __init__(self, n=3, **attrs):
        self.n = n
        self.configured = None
        self.__dict__.update(attrs)

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=self.n)

    def configure(self, args):
        self.configured = args.count

    def tasks(self, args, pool):
        for i in range(args.count):
            yield TaskSpec({'i': i, 'pool': pool}, None)


def _drive(source, argv, conn=None, **kw):
    from aotriton.pgdef.pq import dispatch
    return dispatch.drive(source, argv, conn=conn, stdin_is_tty=False, **kw)


def test_drive_inserts_into_each_pool(capsys):
    conn = FakeConn()
    src = _Source()
    assert _drive(src, ['--pool', 'gfx942', 'gfx950', '--subpool', 'MI355', '--count', '2'], conn) == 0
    assert src.configured == 2
    assert not any('PARTITION OF' in t for t, _ in conn.calls)   # partitions come from schema.sql
    inserts = [p for t, p in conn.calls if t.startswith('INSERT INTO pgdef.task')]
    assert len(inserts) == 2 and all(len(batch) == 2 for batch in inserts)
    spec = inserts[0][0]
    assert (spec['cat_level0'], spec['cat_level1'], spec['subpool']) == ('toy', None, 'MI355')
    assert 'Dispatched 4 task(s)' in capsys.readouterr().out


def test_drive_names_the_fix_for_a_pool_without_partitions(monkeypatch):
    import psycopg
    from aotriton.pgdef.pq import dispatch

    def no_partition(*a, **kw):
        raise psycopg.errors.CheckViolation('no partition of relation "task" found for row')
    monkeypatch.setattr(dispatch._tasks, 'insert', no_partition)
    with pytest.raises(SystemExit, match=r"ensure_pool\('gfx9999'\).*rerun initdb"):
        _drive(_Source(), ['--pool', 'gfx9999', '--count', '1'], FakeConn())


def test_drive_dry_run_touches_nothing(capsys):
    conn = FakeConn()
    assert _drive(_Source(), ['--pool', 'gfx950', '--dry_run'], conn) == 0
    assert conn.calls == []
    assert 'Dry run' in capsys.readouterr().out


def test_drive_skip_completed_compares_payloads_with_done_tasks(capsys):
    done = [{'payload': {'i': 1, 'pool': 'gfx950'}}]
    conn = FakeConn(rows=[done])
    assert _drive(_Source(), ['--pool', 'gfx950', '--skip_completed', '--count', '3'], conn) == 0
    text, params = conn.calls[0]
    assert 'status = ANY(%s)' in text and params[0] == 'gfx950'
    batch = [p for t, p in conn.calls if t.startswith('INSERT')][0]
    assert sorted(b['payload'].obj['i'] for b in batch) == [0, 2]
    assert '1 already done' in capsys.readouterr().out


def test_drive_uses_the_applications_skip_key_and_retry_policy():
    src = _Source(skip_key=lambda payload: payload['i'], default_max_attempts=5)
    conn = FakeConn(rows=[[{'payload': {'i': 0, 'extra': 'ignored'}}]])
    _drive(src, ['--pool', 'p', '--skip_completed', '--count', '2'], conn)
    batch = [p for t, p in conn.calls if t.startswith('INSERT')][0]
    assert [b['payload'].obj['i'] for b in batch] == [1]
    assert batch[0]['logistics'].obj['retry'] == {'max_attempts': 5}


def test_drive_command_line_max_attempts_wins():
    conn = FakeConn()
    _drive(_Source(default_max_attempts=5), ['--pool', 'p', '--max_attempts', '2', '--count', '1'], conn)
    batch = [p for t, p in conn.calls if t.startswith('INSERT')][0]
    assert batch[0]['logistics'].obj['retry'] == {'max_attempts': 2}


def test_drive_needs_pools_from_somewhere():
    with pytest.raises(SystemExit):
        _drive(_Source(), [], FakeConn())
    conn = FakeConn()
    src = _Source(default_pools=lambda args: ['gfx950'])
    assert _drive(src, ['--count', '1'], conn) == 0


def test_drive_requires_a_category_unless_the_source_has_a_default():
    class NoDefault(_Source):
        default_category = None
    with pytest.raises(SystemExit):
        _drive(NoDefault(), ['--pool', 'p'], FakeConn())
    assert _drive(NoDefault(), ['--pool', 'p', '--category', 'toy,x', '--count', '1'], FakeConn()) == 0


def test_drive_rejects_a_bad_category():
    with pytest.raises(SystemExit):
        _drive(_Source(), ['--pool', 'p', '--category', 'a,b,c'], FakeConn())


# ---------------------------------------------------------------------------
# helpers that read rows by position work on dict_row connections too
# ---------------------------------------------------------------------------

def test_positional_readers_ask_for_tuple_rows():
    # CPU workers and the webui use dict_row connections; a helper that reads
    # `row[0]` must not inherit that factory.
    from psycopg.rows import tuple_row
    conn = FakeConn(rows=[[(7,)], [(3,)], [('pending',)], [(9, 'pending')]])
    reports.insert(conn, (1, 'gfx950'), cat_level0='tune', attempt=1, rkey={}, status='OK')
    tasks.count(conn, tasks.TaskSelector(pool='gfx950'))
    lg.fail(conn, (1, 'gfx950'), 'boom', owner='w')
    lg.reap_expired(conn, 'gfx950')
    assert conn.row_factories == [tuple_row] * 4


def test_settling_a_task_requires_an_owner_or_an_explicit_none():
    # The owner guards the lease-expiry race; it is a required keyword, and
    # None means "don't care" (the SQL then skips the lease check).
    conn = FakeConn(rows=[[], [], []])
    with pytest.raises(TypeError):
        tasks.complete(conn, (1, 'gfx950'))
    with pytest.raises(ValueError):
        tasks.release(conn, (1, 'gfx950'), owner='')
    tasks.complete(conn, (1, 'gfx950'), owner=None)
    tasks.complete(conn, (1, 'gfx950'), owner='h/reader-0')
    (none_sql, none_params), (own_sql, own_params) = conn.calls
    assert none_params['owner'] is None and own_params['owner'] == 'h/reader-0'
    assert "%(owner)s::text IS NULL OR t.logistics->'lease'->>'owner' = %(owner)s::text" in own_sql
