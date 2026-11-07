# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.perfmon.pq without a server: the SQL text and parameters of each
function (via the pgdef pgfake), and the schema file against the Python side.
test_perfmon_pq_pg.py (marker `pg`) runs them on a real PostgreSQL."""

import re
from pathlib import Path

import pytest

from aotriton.pgdef.pq import admin
from aotriton.perfmon import pq
from aotriton.perfmon.pq import measurements

from pgfake import FakeConn, check_statement

SCHEMA = Path(pq.__file__).with_name('schema.sql')


def _report(**patch):
    base = {'iface': 'op_attn_fwd', 'backend': 'triton', 'status': 'OK', 'gpu': 2,
            'payload': {'stats': {'median': 1.0}}, 'error': None}
    base.update(patch)
    return base


# -- save_measurement ---------------------------------------------------------------------

def test_save_measurement_writes_one_replacing_report_row():
    conn = FakeConn(rows=[[(1,)], [(9,)]])
    assert pq.save_measurement(conn, (4, 'gfx950'), _report(), attempt=2) == 9
    (lock, lock_params), (text, params) = conn.calls
    assert conn.transactions == 1
    assert 'FOR UPDATE' in lock and lock_params == (4, 'gfx950')
    check_statement(text, list(params))
    assert text.startswith('WITH replaced AS (DELETE FROM pgdef.report')
    # (task, pool, cat_level0, rkey) for the delete, then the row itself
    assert params[:4][2] == 'perfmon' and params[3].obj == {'iface': 'op_attn_fwd', 'backend': 'triton'}
    assert params[4:7] == (4, 'gfx950', 'perfmon')
    assert params[7] == 2 and params[9] == 'OK'
    assert params[-1] == 'gpu-2'


def test_the_unforced_operator_has_a_null_backend_in_its_rkey():
    conn = FakeConn(rows=[[(1,)], [(9,)]])
    pq.save_measurement(conn, (4, 'gfx950'), _report(backend=None), attempt=1)
    assert conn.calls[1][1][3].obj == {'iface': 'op_attn_fwd', 'backend': None}


def test_save_measurement_stores_no_payload_for_a_result_without_data():
    conn = FakeConn(rows=[[(1,)], [(9,)]])
    pq.save_measurement(conn, (4, 'gfx950'), _report(status='crash', payload=None, error={'errno': 5}), attempt=1)
    params = conn.calls[1][1]
    assert params[10] is None and params[11].obj == {'errno': 5}


@pytest.mark.parametrize('status', ['OK', 'NotOK', 'crash', 'excluded'])
def test_every_documented_status_is_accepted(status):
    conn = FakeConn(rows=[[(1,)], [(9,)]])
    pq.save_measurement(conn, (4, 'gfx950'), _report(status=status), attempt=1)
    assert pq.STATUSES == ('OK', 'NotOK', 'crash', 'excluded')


def test_an_unknown_status_is_refused_before_any_sql():
    conn = FakeConn()
    with pytest.raises(ValueError, match='status'):
        pq.save_measurement(conn, (4, 'gfx950'), _report(status='ERROR'), attempt=1)
    assert conn.calls == []


# -- reads -----------------------------------------------------------------------------------------

def test_iter_latest_filters_and_orders():
    conn = FakeConn(rows=[[{'id': 1}, {'id': 2}]])
    rows = list(pq.iter_latest(conn, arch='gfx950', variant='any', subjects=['s1', 's2'], module='flash'))
    assert [r['id'] for r in rows] == [1, 2]
    (text, params), = conn.calls
    check_statement(text, params)
    assert 'FROM perfmon.latest_measurement' in text
    assert text.count('%s') == 4 and params == ['flash', 'gfx950', 'any', ['s1', 's2']]
    assert 'subject = ANY(%s)' in text
    assert 'ORDER BY variant, subject, iface, backend NULLS FIRST' in text


def test_iter_latest_without_optional_filters():
    conn = FakeConn(rows=[[]])
    assert list(pq.iter_latest(conn, arch='gfx950')) == []
    text, params = conn.calls[0]
    assert params == ['gfx950'] and 'variant' not in text.split('WHERE')[1].split('ORDER')[0]


def test_count_superseded_compares_the_two_views_with_the_same_selection():
    conn = FakeConn(rows=[[{'superseded': 7}]])
    assert pq.count_superseded(conn, arch='gfx950', subjects=['s1']) == 7
    (text, params), = conn.calls
    check_statement(text, params)
    assert 'perfmon.measurement' in text and 'perfmon.latest_measurement' in text
    assert params == ['gfx950', ['s1'], 'gfx950', ['s1']]
    assert pq.count_superseded(FakeConn(rows=[[]]), arch='gfx950') == 0


def test_list_selections_groups_and_counts_by_status():
    conn = FakeConn(rows=[[{'subject': 's'}]])
    assert pq.list_selections(conn, arch='gfx950') == [{'subject': 's'}]
    (text, params), = conn.calls
    check_statement(text, params)
    assert params == ['gfx950']
    for status in ('OK', 'NotOK', 'crash'):
        assert f"status = '{status}'" in text
    assert 'GROUP BY module, arch, variant, subject' in text
    assert 'ORDER BY module, arch, variant, subject' in text
    # no filter at all lists everything
    conn = FakeConn(rows=[[]])
    pq.list_selections(conn)
    assert conn.calls[0][1] == [] and 'WHERE TRUE' in conn.calls[0][0]


def test_subject_progress_counts_the_tasks_of_the_application_only():
    conn = FakeConn(rows=[[]])
    pq.subject_progress(conn, arch='gfx950')
    (text, params), = conn.calls
    check_statement(text, params)
    assert "t.cat_level0 = 'perfmon'" in text and params == ['gfx950']
    assert "COALESCE(t.subpool, 'any')" in text


def test_sql_is_not_hidden_in_a_mix_of_styles():
    """Statements use %s with a list, never f-string values."""
    source = Path(measurements.__file__).read_text()
    assert 'params.append' in source
    assert not re.search(r"execute\(f['\"]", source)


# -- the schema file ----------------------------------------------------------------------------------

def test_the_schema_is_in_initdbs_list_after_the_engines():
    files = admin.all_schema_files()
    assert SCHEMA.resolve() in [p.resolve() for p in files]
    assert files[0].parent.parent.name == 'pgdef'
    assert 'perfmon' in admin.schema_names()


def test_the_views_expose_every_column_the_functions_select():
    text = SCHEMA.read_text()
    view = text[text.index('CREATE OR REPLACE VIEW perfmon.measurement'):
                text.index('CREATE OR REPLACE VIEW perfmon.latest_measurement')]
    aliases = set(re.findall(r'\bAS\s+(\w+)', view)) | {'id', 'task_id', 'status', 'error', 'worker',
                                                         'attempt', 'created_at'}
    wanted = {c.strip() for c in measurements.LATEST_COLUMNS.split(',')}
    assert wanted <= aliases, f'not in the view: {wanted - aliases}'
    for name in ('arch', 'variant', 'subject', 'module', 'iface', 'backend', 'entry'):
        assert name in aliases


def test_the_schema_is_idempotent_and_confined_to_its_own_schema():
    text = SCHEMA.read_text()
    code = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('--'))
    assert 'CREATE SCHEMA IF NOT EXISTS perfmon' in code
    assert 'CREATE TABLE' not in code                          # views only
    assert re.findall(r'CREATE (?:OR REPLACE )?VIEW (\S+)', code) == ['perfmon.measurement',
                                                                      'perfmon.latest_measurement']
    assert 'CREATE INDEX IF NOT EXISTS' in code
    assert 'public.' not in code
    assert "cat_level0 = 'perfmon'" in code
