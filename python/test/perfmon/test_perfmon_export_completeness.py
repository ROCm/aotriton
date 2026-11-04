# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export completeness: a task marked done may not have all its results yet.
pq.task_shortfalls compares the appcookie's done pairs with the report rows;
the export leaves incomplete tasks out unless told otherwise."""

from aotriton.perfmon import export, pq

from perfmon_fakes import FakeDesc, row
from pgfake import FakeConn, check_statement


def _task(task_id=1, status='done', missing=(), reported=2, **patch):
    out = {'task_id': task_id, 'arch': 'gfx950', 'variant': 'any', 'subject': 'aotriton-0.13b+rocm10.1.0',
           'module': 'fake', 'entry': {}, 'status': status, 'missing': [list(m) for m in missing],
           'reported': reported, 'reason': 'why'}
    out.update(patch)
    return out


def _db_task(task_id, status, done, reported, current=True):
    return {'task_id': task_id, 'arch': 'gfx950', 'variant': 'any', 'subject': 's', 'module': 'flash',
            'entry': {}, 'status': status, 'done': done, 'reported': reported, 'current': current}


def test_task_shortfalls_sql_selects_perfmon_tasks_only():
    conn = FakeConn(rows=[[]])
    assert pq.task_shortfalls(conn, module='flash', arch='gfx950', variant='any', subjects=['s']) == []
    (text, params), = conn.calls
    check_statement(text, params)
    assert "t.cat_level0 = 'perfmon'" in text and "r.cat_level0 = 'perfmon'" in text
    assert params == ['flash', 'gfx950', 'any', ['s']]


def test_a_done_task_with_all_its_reports_is_complete():
    pairs = [['op_a', None], ['op_a', 'triton']]
    conn = FakeConn(rows=[[_db_task(1, 'done', pairs, pairs)]])
    assert pq.task_shortfalls(conn, arch='gfx950') == []


def test_recorded_pairs_without_report_rows_are_the_shortfall():
    done = [['op_a', None], ['op_a', 'triton'], ['op_b', 'aiter']]
    conn = FakeConn(rows=[[_db_task(1, 'done', done, [['op_a', None]])]])
    (task,) = pq.task_shortfalls(conn, arch='gfx950')
    assert task['missing'] == [['op_a', 'triton'], ['op_b', 'aiter']]
    assert task['reported'] == 1 and '2 of 3' in task['reason']


def test_a_task_not_done_is_incomplete_even_with_every_pair_reported():
    pairs = [['op_a', None]]
    conn = FakeConn(rows=[[_db_task(1, 'running', pairs, pairs)]])
    (task,) = pq.task_shortfalls(conn, arch='gfx950')
    assert task['reason'] == 'task is running' and task['missing'] == []


def test_a_done_task_with_no_report_is_flagged():
    conn = FakeConn(rows=[[_db_task(1, 'done', [], [], current=False)]])
    (task,) = pq.task_shortfalls(conn, arch='gfx950')
    assert 'no report row' in task['reason']


def test_a_task_superseded_by_a_newer_dispatch_is_ignored():
    conn = FakeConn(rows=[[_db_task(1, 'failed', [], [['op_a', None]], current=False)]])
    assert pq.task_shortfalls(conn, arch='gfx950') == []


def test_split_incomplete_leaves_out_the_rows_of_short_tasks():
    desc = FakeDesc()
    entries = list(desc.entries(desc.entry_sets()['prime']))
    rows = [row(desc, entries[0], task_id=1), row(desc, entries[1], task_id=2),
            row(desc, entries[2], task_id=2)]
    kept, held = export.split_incomplete(rows, [_task(2, missing=[('op_a', 'triton')])])
    assert [r['task_id'] for r in kept] == [1]
    assert [t['task_id'] for t in held] == [2]
    kept, held = export.split_incomplete(rows, [])
    assert len(kept) == 3 and held == []


def test_describe_shortfall_names_the_missing_pairs():
    text = export.describe_shortfall(_task(7, missing=[('op_a', None), ('op_b', 'aiter')]))
    assert 'task 7' in text and 'op_a/unforced' in text and 'op_b/aiter' in text


def _selection(**patch):
    out = {'module': 'fake', 'arch': 'gfx950', 'variant': 'any', 'subject': 'aotriton-0.13b+rocm10.1.0',
           'measured': 4, 'ok': 4, 'notok': 0, 'crash': 0, 'newest': None}
    out.update(patch)
    return out


def test_listing_shows_the_shortfall_column_only_when_given():
    given = {'module': 'fake', 'arch': 'gfx950'}
    plain = export.format_listing([_selection()], given=given, exported=lambda s: False)
    assert 'incomplete' not in plain[0]
    short = [_task(1, missing=[('op_a', None), ('op_b', None)]), _task(2, status='running')]
    lines = export.format_listing([_selection(), _selection(variant='b')],
                                  given=given,
                                  exported=lambda s: False,
                                  shortfalls=short)
    assert lines[0].split()[-1] == 'incomplete'
    assert '2 tasks, 2 results missing' in lines[1]
    assert lines[2].rstrip().endswith('-')
