# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The `iterations` metadata of an exported perfdb file: the one count all its
lines used, else MIN-MAX. Old reports without the count are 100."""

import pytest

from aotriton.perfmon import export, perfdb
from aotriton.perfmon.perfdb import PerfdbError

from perfmon_fakes import FakeDesc, row

ARCH = 'gfx950'


@pytest.fixture(scope='module')
def desc():
    return FakeDesc()


def _rows(desc, counts, **patch):
    """One unforced op_a row per count, on the first entries of the family."""
    entries = list(desc.entries(desc.entry_sets()['prime']))
    return [row(desc, entries[i], iterations=n, **patch) for i, n in enumerate(counts)]


def _iterations(root, tag='0.13b'):
    path = root / tag / 'data' / ARCH / 'op_a.perfdb'
    return perfdb.read_perfdb(path).metadata.get('iterations')


def _export(root, desc, rows):
    return export.export_rows(root, desc, ARCH, rows, ())


def test_one_count_is_written_as_is(tmp_path, desc):
    _export(tmp_path, desc, _rows(desc, [100, 100, 100]))
    assert _iterations(tmp_path) == '100'
    _export(tmp_path, desc, _rows(desc, [20, 20]))                  # replaces two lines, keeps one at 100
    assert _iterations(tmp_path) == '20-100'


def test_different_counts_are_a_range(tmp_path, desc):
    _export(tmp_path, desc, _rows(desc, [100, 74, 27, 20]))
    assert _iterations(tmp_path) == '20-100'


def test_a_report_without_the_count_used_100(tmp_path, desc):
    _export(tmp_path, desc, _rows(desc, [None, 20]))
    assert _iterations(tmp_path) == '20-100'
    (tmp_path / '0.14.2b').mkdir()
    _export(tmp_path, desc, _rows(desc, [None, None], subject='aotriton-0.14.2b+rocm10.1.0'))
    assert _iterations(tmp_path, '0.14.2b') == '100'


def test_replacing_every_line_forgets_the_old_range(tmp_path, desc):
    _export(tmp_path, desc, _rows(desc, [100, 20]))
    assert _iterations(tmp_path) == '20-100'
    _export(tmp_path, desc, _rows(desc, [100, 100]))
    assert _iterations(tmp_path) == '100'


def test_lines_without_data_do_not_count(tmp_path, desc):
    rows = _rows(desc, [20, 20])
    rows[1] = dict(rows[1], status='crash', iterations=100, median_us=None)
    _export(tmp_path, desc, rows)
    assert _iterations(tmp_path) == '20'


def test_the_range_survives_an_unrelated_export(tmp_path, desc):
    _export(tmp_path, desc, _rows(desc, [100, 20]))
    again = _rows(desc, [100, 20])
    result = _export(tmp_path, desc, again)
    assert result.changed == 0 and _iterations(tmp_path) == '20-100'


def test_parse_iterations():
    assert export.parse_iterations('100', 'f') == (100, 100)
    assert export.parse_iterations('20-100', 'f') == (20, 100)
    for bad in ('', 'x', '0', '100-20', '-5', '1-2-3'):
        with pytest.raises(PerfdbError, match='iterations'):
            export.parse_iterations(bad, 'f')
    assert export.format_iterations(100, 100) == '100'
    assert export.format_iterations(20, 100) == '20-100'


def test_compare_does_not_refuse_files_that_differ_in_iterations(tmp_path, desc):
    _export(tmp_path, desc, _rows(desc, [100, 100]))
    _export(tmp_path, desc, _rows(desc, [100, 20], subject='aotriton-0.14.2b+rocm10.1.0'))
    old, new = (perfdb.load_arch(tmp_path, tag, ARCH) for tag in ('0.13b', '0.14.2b'))
    assert old['op_a.perfdb'].metadata['iterations'] == '100'
    assert new['op_a.perfdb'].metadata['iterations'] == '20-100'
    ratios, refused = perfdb.compare_files(old, new)
    assert refused == [] and ratios
