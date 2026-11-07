# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The live comparison of head against committed tags and other subjects."""

import argparse

import pytest

from aotriton.perfmon import export, live, perfdb
from aotriton.perfmon.perfdb import PerfdbError

from perfmon_fakes import FakeDesc, row

ARCH = 'gfx950'
HEAD = 'aotriton-head+rocm10.1.0'
TAG = '0.13b'


@pytest.fixture(scope='module')
def desc():
    return FakeDesc()


@pytest.fixture
def entries(desc):
    return list(desc.entries(desc.entry_sets()['prime']))


def _args(**patch):
    out = dict(arch=[ARCH], variant=None, subject=None, against=[], against_subject=[], noise=0.03,
               forced_check_baselines=False, check=[live.REGRESSION, live.FORCED_BEATS_UNFORCED])
    out.update(patch)
    return argparse.Namespace(**out)


def _run(desc, root, args, rows):
    def fetch(arch, subjects):
        return [r for r in rows if subjects is None or r['subject'] in subjects]

    def subjects(arch):
        return sorted({r['subject'] for r in rows})
    return live.run(desc, root, args, fetch, subjects)


def _release(tmp_path, desc, entries, medians, **patch):
    rows = [row(desc, e, median=m, **patch) for e, m in zip(entries, medians)]
    export.export_rows(tmp_path, desc, ARCH, rows, ())


def _head(desc, entries, medians, **patch):
    return [row(desc, e, median=m, subject=HEAD, **patch) for e, m in zip(entries, medians)]


def test_a_head_point_slower_than_the_release_beyond_noise_is_flagged(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:3], [100.0, 100.0, 100.0])
    rows = _head(desc, entries[:3], [100.0, 102.0, 120.0])      # same, within 3%, slower by 20%
    flags, warnings, counts = _run(desc, tmp_path, _args(against=[TAG]), rows)
    (flag,) = flags
    assert flag.check == live.REGRESSION and flag.reference == TAG
    assert flag.coordinates == live.entry_key(desc, entries[2])
    assert flag.ratio == pytest.approx(100 / 120)
    assert counts['compared'] == 3 and warnings == []


def test_faster_head_points_are_not_flagged(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:2], [100.0, 100.0])
    flags, _, _ = _run(desc, tmp_path, _args(against=[TAG]), _head(desc, entries[:2], [50.0, 90.0]))
    assert flags == []


def test_the_noise_option_is_the_threshold(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:1], [100.0])
    rows = _head(desc, entries[:1], [105.0])
    assert len(_run(desc, tmp_path, _args(against=[TAG]), rows)[0]) == 1
    assert _run(desc, tmp_path, _args(against=[TAG], noise=0.06), rows)[0] == []


def test_points_without_data_on_either_side_are_not_compared(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:2], [100.0, 100.0])
    rows = _head(desc, entries[:1], [300.0])
    rows.append(row(desc, entries[1], subject=HEAD, status='crash'))        # head crashed: not a flag
    rows.append(row(desc, entries[2], subject=HEAD, median=300.0))          # the release has no line
    flags, _, counts = _run(desc, tmp_path, _args(against=[TAG]), rows)
    assert len(flags) == 1 and flags[0].coordinates == live.entry_key(desc, entries[0])
    assert counts['compared'] == 1 and counts['no_reference'] == 1


def test_a_different_timing_method_is_refused(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:1], [100.0])
    rows = _head(desc, entries[:1], [300.0], timing_method='other')
    flags, _, counts = _run(desc, tmp_path, _args(against=[TAG]), rows)
    assert flags == [] and counts['refused'] == 1


def test_a_missing_tag_is_a_warning(tmp_path, desc, entries):
    flags, warnings, _ = _run(desc, tmp_path, _args(against=['0.9b']), _head(desc, entries[:1], [1.0]))
    assert flags == [] and 'no committed data' in warnings[0]


def test_another_gpu_gets_a_warning(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:1], [100.0], gpu_device='0x75a0')
    rows = _head(desc, entries[:1], [100.0], gpu_device='0x74a1')
    _, warnings, _ = _run(desc, tmp_path, _args(against=[TAG]), rows)
    assert any('0.06' in w for w in warnings)


def test_compare_against_another_subject_in_the_database(tmp_path, desc, entries):
    other = 'aotriton-0.14.2b+rocm10.1.0'
    rows = _head(desc, entries[:2], [100.0, 200.0]) + [row(desc, e, median=100.0, subject=other)
                                                       for e in entries[:2]]
    flags, _, _ = _run(desc, tmp_path, _args(against_subject=[other]), rows)
    (flag,) = flags
    assert flag.reference == other and flag.measured_us == 200.0


def test_forced_backend_beating_the_unforced_operator_is_flagged(tmp_path, desc, entries):
    e = entries[0]
    rows = [row(desc, e, median=100.0, subject=HEAD),
            row(desc, e, backend='triton', median=80.0, subject=HEAD),         # beats by 25%
            row(desc, e, backend='aiter', median=99.0, subject=HEAD),          # within noise
            row(desc, e, backend='other', median=50.0, subject=HEAD, status='NotOK')]
    flags, _, _ = _run(desc, tmp_path, _args(), rows)
    (flag,) = flags
    assert flag.check == live.FORCED_BEATS_UNFORCED and flag.backend == 'triton'
    assert flag.ratio == pytest.approx(1.25)


def test_forced_check_runs_on_baselines_only_when_asked(tmp_path, desc, entries):
    e = entries[0]
    other = 'aotriton-0.14.2b+rocm10.1.0'
    rows = _head(desc, [e], [100.0]) + [row(desc, e, median=100.0, subject=other),
                                        row(desc, e, backend='triton', median=50.0, subject=other)]
    assert _run(desc, tmp_path, _args(against_subject=[other]), rows)[0] == []
    flags, _, _ = _run(desc, tmp_path, _args(against_subject=[other], forced_check_baselines=True), rows)
    assert [f.reference for f in flags] == [f'unforced ({other})']


def test_a_single_head_subject_is_found_and_two_are_an_error(tmp_path, desc, entries):
    rows = _head(desc, entries[:1], [1.0])
    assert _run(desc, tmp_path, _args(), rows)[1] == []
    rows += _head(desc, entries[:1], [1.0]) and [row(desc, entries[0], subject='aotriton-head+rocm9.0.0')]
    with pytest.raises(PerfdbError, match='one head subject'):
        _run(desc, tmp_path, _args(), rows)


def test_csv_and_markdown_carry_the_full_coordinates(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:1], [100.0])
    flags, _, _ = _run(desc, tmp_path, _args(against=[TAG]), _head(desc, entries[:1], [200.0]))
    header = live.header_of(desc)
    assert header[:6] == list(live.FIXED_COLUMNS)
    assert header[6:6 + len(desc.PROBLEM_COLUMNS)] == list(desc.PROBLEM_COLUMNS)
    csv_lines = live.render_csv(desc, flags).splitlines()
    assert csv_lines[0] == ','.join(header)
    cells = csv_lines[1].split(',')
    assert cells[0] == 'regression' and cells[-3:] == ['100', '200', '0.5']
    assert tuple(cells[6:6 + len(live.entry_key(desc, entries[0]))]) == live.entry_key(desc, entries[0])
    md = live.render_markdown(desc, flags).splitlines()
    assert md[0].startswith('| check |') and len(md) == 3


def test_flags_are_ordered_worst_first(tmp_path, desc, entries):
    _release(tmp_path, desc, entries[:3], [100.0, 100.0, 100.0])
    flags, _, _ = _run(desc, tmp_path, _args(against=[TAG]), _head(desc, entries[:3], [130.0, 200.0, 150.0]))
    assert [f.measured_us for f in flags] == [200.0, 150.0, 130.0]


def test_points_from_perfdb_matches_points_from_rows(tmp_path, desc, entries):
    rows = [row(desc, e, median=10.0 + i) for i, e in enumerate(entries[:3])]
    export.export_rows(tmp_path, desc, ARCH, rows, ())
    assert live.points_from_perfdb(tmp_path, desc, TAG, ARCH) == live.points_from_rows(desc, rows)
    assert perfdb.NOISE_FRACTION == 0.03


def test_check_selects_which_checks_run(tmp_path, desc, entries):
    e = entries[0]
    _release(tmp_path, desc, [e], [100.0])
    rows = [row(desc, e, median=200.0, subject=HEAD),
            row(desc, e, backend='triton', median=80.0, subject=HEAD)]
    both, _, _ = _run(desc, tmp_path, _args(against=[TAG]), rows)
    assert {f.check for f in both} == {live.REGRESSION, live.FORCED_BEATS_UNFORCED}
    only, _, counts = _run(desc, tmp_path, _args(against=[TAG], check=[live.FORCED_BEATS_UNFORCED]), rows)
    assert {f.check for f in only} == {live.FORCED_BEATS_UNFORCED} and counts['compared'] == 0
    only, _, _ = _run(desc, tmp_path, _args(against=[TAG], check=[live.REGRESSION]), rows)
    assert {f.check for f in only} == {live.REGRESSION}


def test_the_summary_counts_per_backend_and_lists_the_top_with_coordinates(tmp_path, desc, entries):
    rows = []
    for i, e in enumerate(entries[:3]):
        rows += [row(desc, e, median=100.0, subject=HEAD),
                 row(desc, e, backend='triton', median=100.0 / (1.5 + i), subject=HEAD)]
    rows.append(row(desc, entries[0], backend='aiter', median=10.0, subject=HEAD))
    flags, _, _ = _run(desc, tmp_path, _args(), rows)
    text = '\n'.join(live.summary_lines(desc, flags, top=2))
    assert 'forced_beats_unforced: 4 flagged' in text
    assert 'top 2 by margin' in text and text.count(' vs unforced: ') == 2
    assert 'margin 10' in text and 'dtype=' in text         # the aiter point is the largest margin
    rows_text = [line.split() for line in text.splitlines() if line.startswith('  unforced')]
    assert rows_text[0][3:5] == ['aiter', '1'] and rows_text[1][3:5] == ['triton', '3']
    assert live.summary_lines(desc, [], top=5) == ['nothing flagged']


def test_the_parser_names_the_checks_and_uses_underscores():
    args = live.build_parser().parse_args(['--arch', ARCH, '--check', 'forced_beats_unforced', '--summary_only',
                                           '--top', '3'])
    assert args.check == ['forced_beats_unforced'] and args.summary_only and args.top == 3
    with pytest.raises(SystemExit):
        live.build_parser().parse_args(['--arch', ARCH, '--check', 'nonsense'])
