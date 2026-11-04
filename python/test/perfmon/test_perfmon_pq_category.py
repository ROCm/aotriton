# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The database is shared with the tuner (category ('tune', ...) tasks, a
`tune` schema): every perfmon query must be confined to perfmon's category.
A guard over all of perfmon.pq's SQL, without a server."""

import inspect
import re
from pathlib import Path

from aotriton.perfmon import pq
from aotriton.perfmon.pq import measurements

from pgfake import FakeConn

SCHEMA = Path(pq.__file__).with_name('schema.sql')


def _statements(call):
    conn = FakeConn(rows=[[]] * 4)
    try:
        call(conn)
    except Exception:       # a fake returning nothing may upset the reader, not the statement
        pass
    return [text for text, _ in conn.calls]


def _readers():
    return {
        'iter_latest': lambda c: list(pq.iter_latest(c, arch='gfx950')),
        'count_superseded': lambda c: pq.count_superseded(c, arch='gfx950'),
        'list_selections': lambda c: pq.list_selections(c),
        'subject_progress': lambda c: pq.subject_progress(c),
        'task_shortfalls': lambda c: pq.task_shortfalls(c),
    }


def test_every_reader_goes_through_a_perfmon_view_or_filters_the_category():
    for name, call in _readers().items():
        for text in _statements(call):
            via_view = 'perfmon.measurement' in text or 'perfmon.latest_measurement' in text
            filtered = "cat_level0 = 'perfmon'" in text
            assert via_view or filtered, f'{name} reads without confining itself to perfmon: {text}'
            if 'pgdef.task' in text:
                assert "t.cat_level0 = 'perfmon'" in text, name
            if 'pgdef.report' in text:
                assert "cat_level0 = 'perfmon'" in text, name


def test_every_reader_of_the_module_is_covered_by_this_file():
    public = {n for n, f in inspect.getmembers(measurements, inspect.isfunction)
              if f.__module__ == measurements.__name__ and not n.startswith('_') and n != 'save_measurement'}
    assert public == set(_readers()), 'a new perfmon query needs a category check here'


def test_the_views_filter_reports_and_tasks_to_perfmon():
    code = '\n'.join(line for line in SCHEMA.read_text().splitlines() if not line.lstrip().startswith('--'))
    view = code[code.index('CREATE OR REPLACE VIEW perfmon.measurement'):
                code.index('CREATE OR REPLACE VIEW perfmon.latest_measurement')]
    assert "WHERE r.cat_level0 = 'perfmon'" in view
    assert re.search(r"JOIN pgdef\.task t ON .*t\.cat_level0 = 'perfmon'", view)


def test_the_writer_stamps_the_perfmon_category():
    conn = FakeConn(rows=[[(1,)], [(9,)]])
    pq.save_measurement(conn,
                        (4, 'gfx950'),
                        {'iface': 'i', 'backend': None, 'status': 'OK', 'gpu': 0},
                        attempt=1)
    assert conn.calls[1][1][6] == 'perfmon'
