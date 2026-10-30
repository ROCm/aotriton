# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""perfmon's TaskSource, driven through pgdef's own parser and planner. No
database: `plan()` is the engine's pure core."""

import dataclasses
import json

import pytest

from aotriton.pgdef.pq import dispatch as pgdispatch
from aotriton.perfmon import dispatch as pdispatch
from aotriton.perfmon.registry import make_perf_desc

S13 = 'aotriton-0.13b+rocm10.1.0'
S14 = 'aotriton-0.14.2b+rocm10.1.0'


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / 'config.rc').write_text('# no settings needed here\n')
    for subject in (S14, S13):
        (tmp_path / 'installed' / 'perfmon' / 'gfx950' / subject).mkdir(parents=True)
    (tmp_path / 'installed' / 'perfmon' / 'gfx950' / 'core').mkdir()       # not a subject
    return tmp_path


def _plan(workdir, *argv, pool='gfx950'):
    argv = ['--workdir', str(workdir), '--module', 'flash', '--pool', pool, *argv]
    source = pdispatch.PerfmonTaskSource(argv)
    args = pgdispatch.build_parser(source).parse_args(argv)
    source.configure(args)
    specs, _ = pgdispatch.plan(source, args, pool, None, source.skip_key)
    return source, args, specs


def _subjects(specs):
    return [s.payload['subject'] for s in specs]


SMALL = ['--hdim', '64', '--causal', '0', '--seqlen', '16', '32']


def test_installed_subjects_exclude_the_core(workdir):
    assert pdispatch.installed_subjects(workdir, 'gfx950') == sorted([S13, S14])
    assert pdispatch.installed_subjects(workdir, 'gfx000') == []


def test_default_category_is_perfmon_with_no_second_level(workdir):
    _, args, specs = _plan(workdir, *SMALL)
    assert args.category == ('perfmon', None)
    assert all(s.category == ('perfmon', None) for s in specs)


def test_the_payload_is_module_subject_entry_and_the_arch_is_the_pool(workdir):
    source, _, specs = _plan(workdir, *SMALL, '--subject', S13)
    payload = specs[0].payload
    assert set(payload) == {'module', 'subject', 'entry'}
    assert payload['module'] == 'flash' and payload['subject'] == S13
    assert 'arch' not in json.dumps(payload) and 'gfx950' not in json.dumps(payload)
    assert make_perf_desc('flash').entry_from_dict(payload['entry']).hdim_qk == 64
    assert specs[0].subpool is None


def test_tasks_are_subject_major(workdir):
    _, _, specs = _plan(workdir, *SMALL)
    # 2 dtypes x 2 dropout x 2 seqlens = 8 entries per subject
    assert len(specs) == 16
    assert _subjects(specs) == [S13] * 8 + [S14] * 8
    assert [s.payload['entry'] for s in specs[:8]] == [s.payload['entry'] for s in specs[8:]]


def test_subject_order_follows_the_command_line(workdir):
    _, _, specs = _plan(workdir, *SMALL, '--subject', S14, S13)
    assert _subjects(specs) == [S14] * 8 + [S13] * 8


def test_a_subject_that_is_not_installed_is_an_error_at_dispatch(workdir):
    with pytest.raises(SystemExit) as e:
        _plan(workdir, *SMALL, '--subject', 'aotriton-0.9.2b+rocm10.1.0')
    assert 'not installed' in str(e.value)


def test_no_installed_subject_is_an_error(workdir):
    with pytest.raises(SystemExit) as e:
        _plan(workdir, *SMALL, pool='gfx942')
    assert 'no subject installed' in str(e.value)


def test_axis_options_replace_the_set_defaults(workdir):
    options = ['--subject', S13, '--hdim', '64', '128', '--causal', '1',
               '--T_io', 'bf16', '--dropout_p', '0.1', '--seqlen', '16']
    _, _, specs = _plan(workdir, *options)
    entries = [s.payload['entry'] for s in specs]
    assert len(entries) == 2
    assert {e['hdim_qk'] for e in entries} == {64, 128}
    assert {(e['causal'], e['T_io'], e['dropout_p'], e['seqlen_q']) for e in entries} == {(1, 'bf16', 0.1, 16)}


def test_the_full_prime_set_is_the_default(workdir):
    _, _, specs = _plan(workdir, '--subject', S13)
    assert len(specs) == 704


def test_the_coverage_set_is_an_l_shape(workdir):
    options = ['--subject', S13, '--entry_set', 'coverage', '--hdim', '64',
               '--T_io', 'bf16', '--dropout_p', '0', '--causal', '0', '--bias_shape', '-']
    _, _, specs = _plan(workdir, *options)
    assert len(specs) == 3 * 11 - 2


def test_entries_the_arch_rejects_are_reported_and_not_dispatched(workdir, capsys):
    (workdir / 'installed' / 'perfmon' / 'gfx1100' / S13).mkdir(parents=True)
    options = ['--subject', S13, '--hdim', '64', '512', '--causal', '0', '--seqlen', '16']
    _, _, specs = _plan(workdir, *options, pool='gfx1100')
    assert {s.payload['entry']['hdim_qk'] for s in specs} == {64}
    assert len(specs) == 4
    assert 'gfx1100 does not support hdim > 256' in capsys.readouterr().out


def test_llm_cases_are_dispatched_with_their_own_sizes(workdir):
    _, _, specs = _plan(workdir, '--subject', S13, '--entry_set', 'llm')
    entries = [s.payload['entry'] for s in specs]
    assert len(entries) == 10
    assert (entries[0]['batch'], entries[0]['bias_shape']) == (512, 'b1qk')
    assert all('ifaces' not in s.payload for s in specs)          # fwd_bwd: every iface


def test_llm_phase_filter_and_forward_only_cases(workdir, monkeypatch):
    desc_class = type(make_perf_desc('flash'))
    original = desc_class.llm_cases
    fwd = dataclasses.replace(original(make_perf_desc('flash'))[0], phase='fwd', model='decode')
    pending = dataclasses.replace(fwd, pending=True, model='later')
    monkeypatch.setattr(desc_class, 'llm_cases', lambda self: original(self)[:1] + [fwd, pending])
    _, _, specs = _plan(workdir, '--subject', S13, '--entry_set', 'llm')
    assert len(specs) == 2                                         # pending is skipped
    assert 'ifaces' not in specs[0].payload and specs[1].payload['ifaces'] == ['op_attn_fwd']
    _, _, specs = _plan(workdir, '--subject', S13, '--entry_set', 'llm', '--phase', 'fwd')
    assert [s.payload.get('ifaces') for s in specs] == [['op_attn_fwd']]


def test_llm_takes_no_grid_options_and_grid_takes_no_phase(workdir):
    with pytest.raises(SystemExit):
        _plan(workdir, '--entry_set', 'llm', '--hdim', '64')
    with pytest.raises(SystemExit):
        _plan(workdir, '--phase', 'fwd')


def test_a_category_with_a_second_level_is_refused(workdir):
    with pytest.raises(SystemExit):
        _plan(workdir, '--category', 'perfmon,kernel')


def test_skip_key_identifies_a_task_by_module_subject_and_entry(workdir):
    source, _, specs = _plan(workdir, *SMALL)
    keys = [source.skip_key(s.payload) for s in specs]
    assert len(set(keys)) == len(keys) == 16
    again = json.loads(json.dumps(specs[0].payload))               # as read back from PostgreSQL
    assert source.skip_key(again) == keys[0]
    other = {**again, 'ifaces': ['op_attn_fwd']}
    assert source.skip_key(other) != keys[0]
    hash(keys[0])


def test_skip_completed_drops_the_done_tasks(workdir):
    source, args, specs = _plan(workdir, *SMALL)
    done = {source.skip_key(s.payload) for s in specs[:3]}
    kept, skipped = pgdispatch.plan(source, args, 'gfx950', done, source.skip_key)
    assert skipped == 3 and len(kept) == 13


def test_dry_run_through_the_driver_prints_and_inserts_nothing(workdir, capsys):
    argv = ['--workdir', str(workdir), '--module', 'flash', '--arch', 'gfx950', *SMALL, '--dry_run']
    assert pdispatch.main(argv) == 0
    assert 'Dry run' in capsys.readouterr().out
