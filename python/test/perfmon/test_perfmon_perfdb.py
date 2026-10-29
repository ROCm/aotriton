# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""perfdb (the repository result format) and the export core, on a tiny fake
family. No GPU, database or network."""

import os
from pathlib import Path

import pytest

from aotriton.perfmon import export, perfdb
from aotriton.perfmon.perfdb import EXCLUDED, NO_DATA, PerfdbError

from perfmon_fakes import FakeDesc, FakeEntry, row

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'perfdb'
S13 = 'aotriton-0.13b+rocm10.1.0'
S14 = 'aotriton-0.14.2b+rocm10.1.0'
ARCH = 'gfx950'


@pytest.fixture(scope='module')
def desc():
    return FakeDesc()


def _prime(desc):
    return list(desc.entries(desc.entry_sets()['prime']))


def _llm(desc):
    return [c.entry for c in desc.llm_cases() if not c.pending]


def _campaign(desc, subject, *, scale=1.0, variant='any', day=1, backends=(None, 'fast')):
    """Rows for every prime entry on op_a and op_b, forced or not, plus the LLM cases."""
    rows = []
    for n, entry in enumerate(_prime(desc) + _llm(desc)):
        for iface in ('op_a', 'op_b'):
            if iface == 'op_b' and entry.seqlen_q == 100:       # model-y is forward only
                continue
            for b, backend in enumerate(backends):
                median = (100.0 + 10 * n + 3 * b) * scale * (2.5 if iface == 'op_b' else 1)
                rows.append(row(desc, entry, iface, backend, median, subject=subject, variant=variant, day=day))
    return rows


def _export(root, desc, rows, arch=ARCH, excluded=()):
    return export.export_rows(root, desc, arch, rows, excluded)


def _tree(root, desc):
    """Every file under root as {relative name: text}."""
    return {str(p.relative_to(root)): p.read_text(encoding='utf-8')
            for p in sorted(root.rglob('*')) if p.is_file()}


# -- numbers and names ---------------------------------------------------------------------

FMT4_CASES = [(12.3456, '12.35'), (123456, '123500'), (0.081, '0.081'), (12.0, '12'),
              (0.00012345, '0.0001234'), (1e-9, '0.000000001'), (99999.9, '100000'), (0, '0'),
              ('7.5', '7.5')]


@pytest.mark.parametrize('value,text', FMT4_CASES)
def test_fmt4_is_four_significant_digits_without_exponent(value, text):
    assert perfdb.fmt4(value) == text


def test_fmt4_refuses_non_finite():
    with pytest.raises(PerfdbError):
        perfdb.fmt4(float('nan'))


def test_subject_tag_and_release_order():
    assert perfdb.subject_tag('aotriton-0.13b+rocm10.1.0') == ('0.13b', '10.1.0')
    with pytest.raises(PerfdbError):
        perfdb.subject_tag('0.13b')
    tags = ['0.14.2b', '0.9.2b', '0.11.2b', '0.10b', '0.13b']
    assert sorted(tags, key=perfdb.tag_sort_key) == ['0.9.2b', '0.10b', '0.11.2b', '0.13b', '0.14.2b']


def test_file_names_split_iface_and_backend_at_the_first_dot():
    assert perfdb.file_name('op_attn_bwd', None) == 'op_attn_bwd.perfdb'
    assert perfdb.file_name('op_attn_bwd', 'triton_split') == 'op_attn_bwd.triton_split.perfdb'
    assert perfdb.parse_file_name('op_attn_bwd.triton_split.perfdb') == ('op_attn_bwd', 'triton_split')
    assert perfdb.parse_file_name('op_attn_bwd.perfdb') == ('op_attn_bwd', None)
    with pytest.raises(PerfdbError):
        perfdb.file_name('op_a', 'Bad-Name')


# -- one data file --------------------------------------------------------------------------------

def _file():
    f = perfdb.new_file('op_a', 'fast')
    f.set_measurement_keys({'timing_method': 'hipgraph_ev100', 'iterations': 100, 'l2_flush': 1, 'unit': 'us'})
    lines = {3: (25.61, 25.7, 0.12, 25.43, 25.48, 25.94),
             1: (12.34, 12.4, 0.081, 12.21, 12.25, 12.61),
             2: NO_DATA,
             7: EXCLUDED}
    f.merge('any', lines)
    return f


GOLDEN_FILE = '''#!/usr/bin/env -S python -m aotriton.perfmon.perfdb
#! metadata
iface,op_a
backend,fast
timing_method,hipgraph_ev100
iterations,100
l2_flush,1
unit,us
#! results any
entry,median,mean,stddev,min,p05,p95
1,12.34,12.4,0.081,12.21,12.25,12.61
2,x
3,25.61,25.7,0.12,25.43,25.48,25.94
7,-
'''


def test_the_writer_is_canonical_and_round_trips(tmp_path):
    f = _file()
    assert perfdb.perfdb_text(f) == GOLDEN_FILE
    path = tmp_path / f.name
    assert perfdb.write_perfdb(path, f) is True
    assert perfdb.write_perfdb(path, f) is False          # same bytes: nothing written
    back = perfdb.read_perfdb(path)
    assert back.blocks == f.blocks and back.metadata == f.metadata
    assert perfdb.perfdb_text(back) == GOLDEN_FILE
    # reading is tolerant of order and spelling, writing is not
    shuffled = ['3,25.610,25.7,0.12,25.43,25.48,25.94', '7,-', '2,x', '1,12.34,12.4,0.081,12.21,12.25,12.61']
    path.write_text(GOLDEN_FILE.split('entry,median')[0] + 'entry,median,mean,stddev,min,p05,p95\n'
                    + '\n'.join(shuffled) + '\n')
    assert perfdb.perfdb_text(perfdb.read_perfdb(path)) == GOLDEN_FILE


def test_the_unforced_file_says_backend_dash():
    f = perfdb.new_file('op_a', None)
    assert f.name == 'op_a.perfdb' and f.metadata['backend'] == '-'


def test_a_reader_checks_the_file_name_against_the_metadata(tmp_path):
    (tmp_path / 'op_a.other.perfdb').write_text(GOLDEN_FILE)
    with pytest.raises(PerfdbError, match='disagree'):
        perfdb.read_perfdb(tmp_path / 'op_a.other.perfdb')


BROKEN_FILES = [
    (GOLDEN_FILE.replace('#! results any', '#! results'), 'unknown section'),
    (GOLDEN_FILE.replace('#! results any', '#! blocks any'), 'unknown section'),
    (GOLDEN_FILE.replace('entry,median', 'entry,mean'), 'must start with'),
    (GOLDEN_FILE.replace('2,x', '1,x'), 'twice'),
    (GOLDEN_FILE.replace('12.34', 'abc'), 'bad line'),
    (GOLDEN_FILE.replace('2,x', '2,12.3'), 'bad line'),
    (GOLDEN_FILE.replace('iface,op_a\n', 'iface,op_a,extra\n'), 'key,value'),
    ('#!/usr/bin/env -S python -m aotriton.perfmon.perfdb\n1,2\n' + GOLDEN_FILE.split('\n', 1)[1],
     'before the first section'),
]


@pytest.mark.parametrize('broken,message', BROKEN_FILES)
def test_a_malformed_file_is_refused(tmp_path, broken, message):
    path = tmp_path / 'op_a.fast.perfdb'
    path.write_text(broken)
    with pytest.raises(PerfdbError, match=message):
        perfdb.read_perfdb(path)


def test_merge_replaces_given_lines_and_keeps_the_others():
    f = _file()
    assert f.merge('any', {1: (1.0, 1.0, 0.0, 1.0, 1.0, 1.0), 3: (25.61, 25.7, 0.12, 25.43, 25.48, 25.94)}) == 1
    assert f.blocks['any'][1][0] == '1'
    assert set(f.blocks['any']) == {1, 2, 3, 7}              # never erases
    assert f.merge('any', {2: EXCLUDED}) == 1                  # a marker can change to another
    assert f.merge('other', {5: NO_DATA}) == 1
    assert set(f.blocks) == {'any', 'other'}


def test_a_file_never_mixes_measurement_methods():
    f = _file()
    f.set_measurement_keys({'iterations': 100})
    with pytest.raises(PerfdbError, match='never mixes'):
        f.set_measurement_keys({'timing_method': 'stream_ev'})
    assert f.measurement_keys()['timing_method'] == 'hipgraph_ev100'


def test_variant_blocks_are_sorted_and_empty_blocks_vanish():
    f = _file()
    f.merge('pool_a', {4: (1, 1, 1, 1, 1, 1)})
    f.merge('zzz', {})
    text = perfdb.perfdb_text(f)
    assert text.index('#! results any') < text.index('#! results pool_a')
    assert '#! results zzz' not in text


# -- the entry list ----------------------------------------------------------------------------------

def test_ordinals_are_implicit_append_only_and_found_by_content(tmp_path, desc):
    entries = perfdb.EntryList.load(tmp_path, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    a, b = _prime(desc)[0], _prime(desc)[1]
    assert entries.ordinal_of(desc, a) is None
    assert entries.ordinal_of(desc, a, add=True) == 1
    assert entries.ordinal_of(desc, b, add=True) == 2
    assert entries.ordinal_of(desc, a, add=True) == 1
    assert entries.appended == {'problems': 1, 'entries': 2}        # a and b share a problem
    entries.save(tmp_path)
    assert (tmp_path / '.metadata' / 'problems.csv').read_text() == 'T_io,hdim,causal\nbf16,16,0\n'
    assert (tmp_path / '.metadata' / 'entries.csv').read_text() == (
        'problem,seqlen_q,seqlen_k,batch,n_heads_q\n1,16,16,2,8\n1,32,32,2,8\n')

    again = perfdb.EntryList.load(tmp_path, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    assert again.appended == {'problems': 0, 'entries': 0}
    new = FakeEntry('fp16', 16, 0, 16, 16, 2, 8)
    assert again.ordinal_of(desc, new, add=True) == 3               # appended after the old ones
    assert again.ordinal_of(desc, b) == 2 and again.ordinal_of(desc, a) == 1
    assert again.entry_of(desc, 3) == new
    again.save(tmp_path)
    assert again.problems[1] == ('fp16', '16', '0')


def test_a_grid_entry_and_an_llm_case_with_other_sizes_are_two_entries(tmp_path, desc):
    entries = perfdb.EntryList(desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    one = FakeEntry('bf16', 64, 0, 128, 128, 2, 8)
    two = FakeEntry('bf16', 64, 0, 128, 128, 4, 12)
    assert (entries.ordinal_of(desc, one, add=True), entries.ordinal_of(desc, two, add=True)) == (1, 2)
    assert len(entries.problems) == 1


def test_the_entry_list_refuses_a_changed_header(tmp_path, desc):
    entries = perfdb.EntryList(desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    entries.ordinal_of(desc, _prime(desc)[0], add=True)
    entries.save(tmp_path)
    with pytest.raises(PerfdbError, match='migration'):
        perfdb.EntryList.load(tmp_path, desc.PROBLEM_COLUMNS + ('extra',), desc.SIZE_COLUMNS)


def test_the_entry_list_refuses_a_dangling_problem(tmp_path, desc):
    meta = tmp_path / '.metadata'
    meta.mkdir()
    (meta / 'problems.csv').write_text('T_io,hdim,causal\nbf16,16,0\n')
    (meta / 'entries.csv').write_text('problem,seqlen_q,seqlen_k,batch,n_heads_q\n2,16,16,2,8\n')
    with pytest.raises(PerfdbError, match='out of range'):
        perfdb.EntryList.load(tmp_path, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)


# -- platform.csv -------------------------------------------------------------------------------------

def test_platform_rows_span_their_dates_and_conflicts_are_found():
    cfg = {'variant': 'any', 'gpu_device_id': '0x75a0', 'vram_gib': '288', 'power_cap_w': None,
           'power_cap_default_w': '1400', 'gpus_per_node': 8, 'thermal_gate': 'off', 'cpu': 'C', 'ram_speed': None,
           'os': 'Debian', 'kernel': '6.12', 'amdgpu': None, 'rocm': '10.1.0', 'harness': 'abc'}
    rows = perfdb.merge_platform([], [(cfg, '2026-11-03'), (cfg, '2026-11-01'), (cfg, '2026-11-02')])
    assert len(rows) == 1 and rows[0]['measured'] == '2026-11-01..2026-11-03'
    rows = perfdb.merge_platform(rows, [(cfg, '2026-11-09')])
    assert rows[0]['measured'] == '2026-11-01..2026-11-09' and rows[0]['power_cap_w'] == ''
    assert perfdb.platform_conflicts(rows) == {}
    rows = perfdb.merge_platform(rows, [({**cfg, 'gpu_device_id': '0x75a1'}, '2026-11-09')])
    assert len(rows) == 2 and perfdb.platform_conflicts(rows) == {'any': [('0x75a0', '1400'), ('0x75a1', '1400')]}
    text = perfdb.platform_text(rows)
    assert text.splitlines()[0] == ','.join(perfdb.PLATFORM_COLUMNS)


# -- export_rows ---------------------------------------------------------------------------------------

def test_export_writes_one_file_per_iface_and_backend_in_the_tags_directory(tmp_path, desc):
    result = _export(tmp_path, desc, _campaign(desc, S13))
    data = tmp_path / '0.13b' / 'data' / ARCH
    assert sorted(p.name for p in data.iterdir()) == ['op_a.fast.perfdb', 'op_a.perfdb', 'op_b.fast.perfdb',
                                                      'op_b.perfdb', 'platform.csv']
    assert result.tags == ['0.13b'] and len(result.files) == 4
    assert result.appended == {'problems': 6, 'entries': 10}     # 8 prime + 2 llm cases
    f = perfdb.read_perfdb(data / 'op_a.perfdb')
    assert f.metadata == {'iface': 'op_a', 'backend': '-', 'timing_method': 'hipgraph_ev100',
                          'iterations': '100', 'l2_flush': '1', 'unit': 'us'}
    assert set(f.blocks) == {'any'} and len(f.blocks['any']) == 10
    assert f.blocks['any'][1] == ('100', '100', '0.05', '99.8', '99.9', '100.3')
    assert len(perfdb.read_perfdb(data / 'op_b.perfdb').blocks['any']) == 9      # model-y has no backward
    assert not (tmp_path / 'head').exists()


def test_status_mapping(tmp_path, desc):
    entries = _prime(desc)[:4]
    rows = [row(desc, entries[0], status='OK'),
            row(desc, entries[1], status='NotOK'),
            row(desc, entries[2], status='crash'),
            row(desc, entries[3], status='excluded')]
    result = _export(tmp_path, desc, rows)
    block = perfdb.read_perfdb(tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb').blocks['any']
    assert block[1][0] == '100' and block[2] == NO_DATA and block[3] == NO_DATA and block[4] == EXCLUDED
    assert (result.no_data, result.excluded) == (2, 1)


def test_an_ok_row_without_statistics_is_no_data(tmp_path, desc):
    _export(tmp_path, desc, [row(desc, _prime(desc)[0], median_us=None)])
    block = perfdb.read_perfdb(tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb').blocks['any']
    assert block == {1: NO_DATA}


def test_head_is_never_exported(tmp_path, desc):
    with pytest.raises(PerfdbError, match='head'):
        _export(tmp_path, desc, [row(desc, _prime(desc)[0], subject='aotriton-head+rocm10.1.0')])
    assert not list(tmp_path.iterdir())


def test_two_rocm_versions_of_one_tag_are_an_error(tmp_path, desc):
    rows = [row(desc, _prime(desc)[0], subject=S13), row(desc, _prime(desc)[1], subject='aotriton-0.13b+rocm7.14')]
    with pytest.raises(PerfdbError, match='two ROCm'):
        _export(tmp_path, desc, rows)


def test_two_tags_go_to_two_directories(tmp_path, desc):
    _export(tmp_path, desc, _campaign(desc, S13)[:3] + _campaign(desc, S14)[:3])
    assert perfdb.list_tags(tmp_path) == ['0.13b', '0.14.2b']
    assert perfdb.list_arches(tmp_path, '0.14.2b') == [ARCH]


def test_exporting_again_is_a_no_op_and_ordinals_are_stable(tmp_path, desc):
    rows = _campaign(desc, S13)
    _export(tmp_path, desc, rows)
    before = _tree(tmp_path, desc)
    result = _export(tmp_path, desc, list(reversed(rows)))
    assert _tree(tmp_path, desc) == before and result.files == [] and result.changed == 0
    assert result.appended == {'problems': 0, 'entries': 0}


def test_a_partial_export_keeps_what_the_database_does_not_have(tmp_path, desc):
    rows = _campaign(desc, S13)
    _export(tmp_path, desc, rows)
    path = tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb'
    before = perfdb.read_perfdb(path).blocks['any']
    newer = [r for r in rows if r['iface'] == 'op_a' and r['backend'] is None][:2]
    for r in newer:
        r['median_us'] = 50.0
    result = _export(tmp_path, desc, newer)
    after = perfdb.read_perfdb(path).blocks['any']
    assert set(after) == set(before) and result.changed == 2
    assert after[1][0] == '50' and after[3] == before[3]


def test_new_entries_are_appended_after_the_old_ones(tmp_path, desc):
    _export(tmp_path, desc, [row(desc, _prime(desc)[2])])
    assert _export(tmp_path, desc, [row(desc, _prime(desc)[0])]).appended == {'problems': 1, 'entries': 1}
    entries = perfdb.EntryList.load(tmp_path, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    assert entries.ordinal_of(desc, _prime(desc)[2]) == 1 and entries.ordinal_of(desc, _prime(desc)[0]) == 2


def test_a_method_change_skips_the_row_and_warns_instead_of_aborting(tmp_path, desc):
    _export(tmp_path, desc, [row(desc, _prime(desc)[0])])
    path = tmp_path / '0.13b' / 'data' / ARCH / 'op_a.fast.perfdb'
    before = path.read_text() if path.exists() else None
    rows = [row(desc, _prime(desc)[1], timing_method='stream_ev'),
            row(desc, _prime(desc)[2])]
    result = _export(tmp_path, desc, rows)
    assert result.skipped == 1 and result.written == 1
    assert any('never mixes' in w for w in result.warnings)
    assert any('1 rows skipped in total' in w for w in result.warnings)
    entries = perfdb.EntryList.load(tmp_path, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    assert entries.ordinal_of(desc, _prime(desc)[1]) is None        # the skipped row added no entry
    assert entries.ordinal_of(desc, _prime(desc)[2]) is not None
    bad = export.verify(tmp_path, desc, ARCH, rows)
    assert bad == [], bad                                           # verify does not flag the skipped row


def test_excluded_entries_are_rederived_for_the_covered_files(tmp_path, desc):
    rejected = [FakeEntry('bf16', 256, 0, 16, 16, 2, 8), FakeEntry('bf16', 512, 0, 16, 16, 2, 8)]
    rows = [row(desc, _prime(desc)[0]), row(desc, _prime(desc)[0], iface='op_b', backend='fast')]
    excluded = [(rejected[0], None), (rejected[1], ('op_a',))]
    result = _export(tmp_path, desc, rows, arch='gfx1100', excluded=excluded)
    data = tmp_path / '0.13b' / 'data' / 'gfx1100'
    a, b = perfdb.read_perfdb(data / 'op_a.perfdb'), perfdb.read_perfdb(data / 'op_b.fast.perfdb')
    assert a.blocks['any'] == {1: a.blocks['any'][1], 2: EXCLUDED, 3: EXCLUDED}
    assert b.blocks['any'][2] == EXCLUDED and 3 not in b.blocks['any']      # op_a only
    assert result.excluded == 3


def test_excluded_lines_never_replace_measured_ones(tmp_path, desc):
    entry = _prime(desc)[0]
    _export(tmp_path, desc, [row(desc, entry)])
    _export(tmp_path, desc, [row(desc, _prime(desc)[1])], excluded=[(entry, None)])
    block = perfdb.read_perfdb(tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb').blocks['any']
    assert block[1][0] == '100'


def test_excluded_for_sets_follows_validate_hw_feature(desc):
    out = export.excluded_for_sets(desc, 'gfx1100', ['prime', 'llm'])
    assert len(out) == 4 and all(e.hdim == 256 and ifaces is None for e, ifaces in out)
    assert export.excluded_for_sets(desc, 'gfx950', ['prime', 'llm']) == []


def test_platform_csv_comes_from_the_rows(tmp_path, desc):
    rows = [row(desc, _prime(desc)[0], day=3), row(desc, _prime(desc)[1], day=1),
            row(desc, _prime(desc)[2], day=2, gpu_device='0x75a1')]
    result = _export(tmp_path, desc, rows)
    platform = perfdb.read_platform(tmp_path / '0.13b' / 'data' / ARCH / 'platform.csv')
    assert [(r['gpu_device_id'], r['measured']) for r in platform] == [('0x75a0', '2026-11-01..2026-11-03'),
                                                                        ('0x75a1', '2026-11-02')]
    assert platform[0]['rocm'] == '10.1.0' and platform[0]['harness'] == 'abc1234'
    assert platform[0]['vram_gib'] == '288' and platform[0]['thermal_gate'] == 'off'
    assert len(result.warnings) == 1 and "variant 'any'" in result.warnings[0]


def test_a_numeric_pci_device_id_is_spelled_in_hex(tmp_path, desc):
    r = row(desc, _prime(desc)[0])
    r['platform']['pci_device_id'] = 0x75a0
    r['platform']['thermal_gate'] = True
    _export(tmp_path, desc, [r])
    (platform,) = perfdb.read_platform(tmp_path / '0.13b' / 'data' / ARCH / 'platform.csv')
    assert (platform['gpu_device_id'], platform['thermal_gate']) == ('0x75a0', 'on')


def test_verify_reads_the_files_back(tmp_path, desc):
    rows = _campaign(desc, S13)
    _export(tmp_path, desc, rows)
    assert export.verify(tmp_path, desc, ARCH, rows) == []
    path = tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb'
    path.write_text(path.read_text().replace('\n1,100,', '\n1,101,'))
    bad = export.verify(tmp_path, desc, ARCH, rows)
    assert len(bad) == 1 and 'op_a.perfdb' in bad[0] and 'entry 1' in bad[0]
    assert export.verify(tmp_path / 'elsewhere', desc, ARCH, rows[:1])[0].endswith('missing')


# -- two variants ----------------------------------------------------------------------------------------

def test_two_variant_blocks_merge_independently(tmp_path, desc):
    _export(tmp_path, desc, _campaign(desc, S13, variant='any') + _campaign(desc, S13, variant='pool_a', scale=2.0))
    path = tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb'
    f = perfdb.read_perfdb(path)
    assert sorted(f.blocks) == ['any', 'pool_a']
    assert f.blocks['pool_a'][1][0] == '200' and f.blocks['any'][1][0] == '100'
    assert perfdb.variants_present(tmp_path, '0.13b', ARCH) == {'any', 'pool_a'}
    assert perfdb.variants_present(tmp_path, '0.13b', 'gfx000') == set()
    # re-exporting one variant leaves the other block as it was
    before = path.read_text()
    again = [r for r in _campaign(desc, S13, variant='pool_a', scale=3.0)
             if r['iface'] == 'op_a' and r['backend'] is None][:2]
    _export(tmp_path, desc, again)
    after = perfdb.read_perfdb(path)
    assert after.blocks['pool_a'][1][0] == '300' and after.blocks['any'] == f.blocks['any']
    assert after.blocks['pool_a'][4] == f.blocks['pool_a'][4]
    assert before != path.read_text()


# -- compare ---------------------------------------------------------------------------------------------------

def test_compare_joins_files_variants_and_entries(tmp_path, desc):
    _export(tmp_path, desc, _campaign(desc, S13) + _campaign(desc, S13, variant='pool_a'))
    _export(tmp_path, desc, _campaign(desc, S14, scale=0.5) + _campaign(desc, S14, variant='pool_a', scale=2.0))
    old, new = perfdb.load_arch(tmp_path, '0.13b', ARCH), perfdb.load_arch(tmp_path, '0.14.2b', ARCH)
    ratios, refused = perfdb.compare_files(old, new)
    assert refused == []
    assert len(ratios) == 2 * (10 + 9) * 2                       # variants x entries per iface x (op, forced)
    by_variant = {}
    for name, variant, entry, ma, mb, ratio in ratios:
        by_variant.setdefault(variant, set()).add(round(ratio, 2))
    assert by_variant == {'any': {2.0}, 'pool_a': {0.5}}
    only, _ = perfdb.compare_files(old, new, variant='any')
    assert {r[1] for r in only} == {'any'}
    unforced, _ = perfdb.compare_files(old, new, names={'op_a.perfdb'})
    assert {r[0] for r in unforced} == {'op_a.perfdb'}
    assert perfdb.geomean(r[5] for r in only) == pytest.approx(2.0, rel=1e-2)


def test_compare_refuses_files_whose_measurement_keys_differ(tmp_path, desc):
    _export(tmp_path, desc, _campaign(desc, S13)[:4])
    _export(tmp_path, desc, _campaign(desc, S14)[:4])
    path = tmp_path / '0.14.2b' / 'data' / ARCH / 'op_a.perfdb'
    path.write_text(path.read_text().replace('timing_method,hipgraph_ev100', 'timing_method,stream_ev'))
    ratios, refused = perfdb.compare_files(perfdb.load_arch(tmp_path, '0.13b', ARCH),
                                           perfdb.load_arch(tmp_path, '0.14.2b', ARCH))
    assert refused == ['op_a.perfdb']
    assert 'op_a.perfdb' not in {r[0] for r in ratios} and ratios


def test_the_compare_command_prints_ratios_and_fails_on_a_refusal(tmp_path, desc, capsys):
    _export(tmp_path, desc, _campaign(desc, S13)[:4])
    _export(tmp_path, desc, _campaign(desc, S14, scale=0.5)[:4])
    assert perfdb.main(['compare', '0.13b', '0.14.2b', '--perfdb', str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert 'arch,file,variant,entry,median_a,median_b,a/b' in out
    assert 'overall geometric mean' in out and 'a/b 2' in out
    path = tmp_path / '0.14.2b' / 'data' / ARCH / 'op_a.perfdb'
    path.write_text(path.read_text().replace('l2_flush,1', 'l2_flush,0'))
    assert perfdb.main(['compare', '0.13b', '0.14.2b', '--perfdb', str(tmp_path), '--summary_only']) == 1


# -- rendering -------------------------------------------------------------------------------------------------------

def _golden_tree(root, desc):
    _export(root, desc, _campaign(desc, S13) + _campaign(desc, S13, variant='pool_a', scale=2.0))
    campaign = _campaign(desc, S14, scale=0.5) + _campaign(desc, S14, variant='pool_a', scale=2.0, day=2)
    campaign[3]['status'], campaign[3]['median_us'] = 'NotOK', None
    _export(root, desc, campaign)
    # an arch that cannot run hdim 256: those cells are excluded, not blank
    small = [r for r in _campaign(desc, S13) if r['entry']['hdim'] == 16 and r['backend'] is None]
    _export(root, desc, small, arch='gfx1100', excluded=export.excluded_for_sets(desc, 'gfx1100', ['prime']))
    perfdb.render(root, desc)


def _read_golden(name):
    return (GOLDEN / name).read_text(encoding='utf-8')


def test_rendering_matches_the_golden_files(tmp_path, desc):
    _golden_tree(tmp_path, desc)
    tree = _tree(tmp_path, desc)
    if os.environ.get('AOTRITON_UPDATE_GOLDEN') == '1':
        for name, text in tree.items():
            (GOLDEN / name).parent.mkdir(parents=True, exist_ok=True)
            (GOLDEN / name).write_text(text, encoding='utf-8')
    golden = {str(p.relative_to(GOLDEN)) for p in GOLDEN.rglob('*') if p.is_file()}
    assert set(tree) == golden
    for name, text in tree.items():
        assert text == _read_golden(name), f'{name} differs from golden/perfdb/{name} (AOTRITON_UPDATE_GOLDEN=1 regenerates)'


def test_render_loads_each_tag_and_arch_once(tmp_path, desc, monkeypatch):
    _golden_tree(tmp_path, desc)
    calls = []
    real = perfdb.load_arch
    monkeypatch.setattr(perfdb, 'load_arch', lambda root, tag, arch: calls.append((tag, arch)) or real(root, tag, arch))
    perfdb.render_all(tmp_path, desc)
    assert calls and len(calls) == len(set(calls))


def test_verify_loads_each_tag_once(tmp_path, desc, monkeypatch):
    rows = _campaign(desc, S13)
    _export(tmp_path, desc, rows)
    calls = []
    real = perfdb.load_arch
    monkeypatch.setattr(perfdb, 'load_arch', lambda root, tag, arch: calls.append(tag) or real(root, tag, arch))
    assert export.verify(tmp_path, desc, ARCH, rows) == []
    assert calls == ['0.13b']


def test_render_is_deterministic_and_check_detects_drift(tmp_path, desc):
    _golden_tree(tmp_path, desc)
    first = _tree(tmp_path, desc)
    assert perfdb.render(tmp_path, desc) == []                       # nothing differs: nothing written
    assert perfdb.render(tmp_path, desc, check=True) == []
    readme = tmp_path / '0.13b' / ARCH
    page = tmp_path / '0.13b' / f'{ARCH}.md'
    page.write_text(page.read_text() + 'hand edit\n')
    assert perfdb.render(tmp_path, desc, check=True) == [page]
    assert page.read_text().endswith('hand edit\n')                  # --check writes nothing
    assert perfdb.render(tmp_path, desc) == [page]
    assert _tree(tmp_path, desc) == first


def test_the_render_check_command_exits_nonzero_when_out_of_date(tmp_path, desc, monkeypatch, capsys):
    _golden_tree(tmp_path, desc)
    monkeypatch.setattr(perfdb, '_desc_for', lambda root, module: desc)
    assert perfdb.main(['render', str(tmp_path), '--check']) == 0
    (tmp_path / README_NAME).write_text('stale\n')
    assert perfdb.main(['render', str(tmp_path), '--check']) == 1
    assert 'out of date' in capsys.readouterr().err
    assert perfdb.main(['render', str(tmp_path)]) == 0
    assert perfdb.main(['render', str(tmp_path), '--check']) == 0


README_NAME = 'README.md'


def test_the_page_has_one_section_per_variant_and_the_markers(tmp_path, desc):
    _golden_tree(tmp_path, desc)
    page = (tmp_path / '0.14.2b' / f'{ARCH}.md').read_text()
    assert page.index('## Variant `any`') < page.index('## Variant `pool_a`')
    assert '✗' in page                                                  # the NotOK line
    assert 'model-x' in page and 'model-y' in page and 'model-z' not in page   # pending is not listed
    index = (tmp_path / 'README.md').read_text()
    assert '[0.13b](0.13b/README.md)' in index and '[0.14.2b](0.14.2b/README.md)' in index


def test_show_prints_a_file_with_its_entries_spelled_out(tmp_path, desc, monkeypatch, capsys):
    _export(tmp_path, desc, _campaign(desc, S13)[:2])
    monkeypatch.setattr(perfdb, '_desc_for', lambda root, module: desc)
    path = tmp_path / '0.13b' / 'data' / ARCH / 'op_a.perfdb'
    assert perfdb.main([str(path)]) == 0                                # the shebang form
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'variant,entry,T_io,hdim,causal,seqlen_q,seqlen_k,batch,n_heads_q,median,mean,stddev,min,p05,p95'
    assert lines[1].startswith('any,1,bf16,16,0,16,16,2,8,100,')


# -- the export command ----------------------------------------------------------------------------------------------

def test_the_listing_prints_only_the_levels_that_were_not_given():
    selections = [
        {'module': 'flash', 'arch': ARCH, 'variant': 'any', 'subject': S13, 'measured': 2816, 'ok': 2790,
         'notok': 26, 'crash': 0, 'newest': __import__('datetime').datetime(2026, 11, 20, 3, 0)},
        {'module': 'flash', 'arch': ARCH, 'variant': 'any', 'subject': S14, 'measured': 2816, 'ok': 2801,
         'notok': 15, 'crash': 0, 'newest': __import__('datetime').datetime(2026, 11, 21, 3, 0)},
    ]
    given = {'module': 'flash', 'arch': ARCH}
    lines = export.format_listing(selections, given=given, exported=lambda s: s['subject'] == S13)
    assert lines == [
        'variant  subject                      measured    OK  NotOK  crash  newest      exported',
        'any      aotriton-0.13b+rocm10.1.0        2816  2790     26      0  2026-11-20  yes',
        'any      aotriton-0.14.2b+rocm10.1.0      2816  2801     15      0  2026-11-21  no',
    ]
    everything = export.format_listing(selections[:1], given={}, exported=lambda s: False)
    assert everything[0].split()[:4] == ['module', 'arch', 'variant', 'subject']


def test_is_exported_looks_for_the_variants_block_in_the_tags_directory(tmp_path, desc):
    _export(tmp_path, desc, _campaign(desc, S13)[:2])
    one = {'module': 'fake', 'arch': ARCH, 'variant': 'any', 'subject': S13}
    assert export._is_exported(one, tmp_path) is True
    assert export._is_exported({**one, 'variant': 'pool_a'}, tmp_path) is False
    assert export._is_exported({**one, 'subject': S14}, tmp_path) is False
    assert export._is_exported({**one, 'subject': 'garbage'}, tmp_path) is False


def test_export_arguments_use_underscores_and_defaults():
    parser = export.build_parser()
    args = parser.parse_args(['--module', 'flash', '--arch', ARCH])
    assert args.entry_set == ['prime', 'llm'] and args.list_only is False and args.verify is False
    assert parser.parse_args(['-l']).list_only is True
    dashed = [o for a in parser._actions for o in a.option_strings if o.startswith('--') and '-' in o[2:]]
    assert dashed == []


def test_export_requires_module_and_arch_unless_listing(monkeypatch):
    with pytest.raises(SystemExit, match='--module and --arch'):
        export.main([])
