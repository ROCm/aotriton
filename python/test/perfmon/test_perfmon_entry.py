# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The flash perfmon plugin's entry sets, size rule, byte model, entry PON
and workloads list. No GPU, torch, engine or database."""

import csv
import dataclasses
from pathlib import Path

import pytest

from aotriton.perfmon.registry import make_perf_desc
from aotriton.utils.pon import parse_pon

_FLASH_PERFMON = Path(__file__).resolve().parents[3] / 'modules' / 'flash' / 'perfmon'

#: The runner's entry PON, in order (the native codec reads these keys).
PON_KEYS = ('T_io', 'hdim_qk', 'hdim_v', 'causal', 'window_left', 'window_right', 'dropout_p',
            'bias_type', 'bias_shape', 'gqa_ratio', 'varlen', 'storage_flip', 'seqlen_q',
            'seqlen_k', 'batch', 'n_heads_q')


@pytest.fixture(scope='module')
def desc():
    return make_perf_desc('flash')


@pytest.fixture(scope='module')
def prime(desc):
    return list(desc.entries(desc.entry_sets()['prime']))


def _block(name):
    import importlib
    from aotriton.perfmon.registry import load_family_perfmon
    mod = load_family_perfmon('flash')
    return importlib.import_module(f'.{name}', package=mod.__name__)


# -- entry sets ---------------------------------------------------------------------

def test_prime_set_has_704_entries(prime):
    # 8 hdims x 2 causal x 2 dtypes x 2 dropout x 11 seqlens
    assert len(prime) == 8 * 2 * 2 * 2 * 11 == 704
    assert len(set(prime)) == 704


def test_prime_set_is_the_plain_diagonal(prime):
    assert all(e.seqlen_q == e.seqlen_k for e in prime)
    assert {e.dropout_p for e in prime} == {0.0, 0.1}
    assert {e.T_io for e in prime} == {'bf16', 'fp16'}
    assert {e.hdim_qk for e in prime} == {16, 32, 64, 128, 192, 256, 384, 512}
    assert all((e.bias_type, e.gqa_ratio, e.varlen, e.storage_flip) == (0, 1, 0, 0) for e in prime)
    assert all((e.window_left, e.window_right) == (-1, -1) for e in prime)
    assert all(e.hdim_qk == e.hdim_v for e in prime)


def test_prime_dropout_twins_share_their_sizes(prime):
    by_key = {}
    for e in prime:
        key = (e.T_io, e.hdim_qk, e.causal, e.seqlen_q)
        by_key.setdefault(key, set()).add((e.batch, e.n_heads_q))
    assert all(len(sizes) == 1 for sizes in by_key.values())


@pytest.mark.parametrize('n', [1, 2, 4, 11])
def test_l_shape_has_3n_minus_2_pairs(n):
    entry = _block('entry')
    ladder = [16 * 2 ** i for i in range(n)]
    pairs = entry.seqlen_pairs(ladder, 'l_shape')
    assert len(pairs) == 3 * n - 2
    top = ladder[-1]
    assert all(q == k or q == top or k == top for q, k in pairs)


def test_diagonal_pairs():
    entry = _block('entry')
    assert entry.seqlen_pairs([64, 16, 32], 'diagonal') == [(16, 16), (32, 32), (64, 64)]
    with pytest.raises(ValueError):
        entry.seqlen_pairs([16], 'zigzag')


def test_coverage_is_an_l_shape_over_the_supported_problems(desc):
    axes = dict(desc.entry_sets()['coverage'])
    axes.update({'T_io': ['bf16'], 'hdim': [64], 'dropout_p': [0.0], 'bias_shape': ['-', 'bhqk'],
                 'causal': [0, 1]})
    entries = list(desc.entries(axes))
    # causal with a bias is not offered: 3 supported problems
    assert len(entries) == 3 * (3 * 11 - 2)
    assert not [e for e in entries if e.causal and e.bias_type]


def test_axes_filter_dispatch_style(desc):
    axes = dict(desc.entry_sets()['prime'])
    axes.update({'hdim': [64], 'causal': [0], 'seqlen': [16, 32]})
    assert len(list(desc.entries(axes))) == 2 * 2 * 2     # dtypes x dropout x seqlens


def test_entry_validation():
    entry = _block('entry')
    ok = {'T_io': 'bf16', 'hdim_qk': 64, 'hdim_v': 64, 'causal': 0, 'window_left': -1, 'window_right': -1,
          'dropout_p': 0.0, 'bias_type': 0, 'bias_shape': '-', 'gqa_ratio': 1, 'varlen': 0,
          'storage_flip': 0, 'seqlen_q': 16, 'seqlen_k': 16, 'batch': 1, 'n_heads_q': 4}
    entry.FlashPerfEntry(**ok)
    for patch in ({'T_io': 'int8'}, {'bias_shape': 'zz'}, {'bias_type': 1}, {'n_heads_q': 3, 'gqa_ratio': 2}):
        with pytest.raises(ValueError):
            entry.FlashPerfEntry(**{**ok, **patch})


# -- the size rule -----------------------------------------------------------------------

def test_size_rule_short_seqlens_are_capped_by_the_batch_cap(desc, prime):
    size = _block('size')
    short = [e for e in prime if e.seqlen_q == 16]
    assert all(e.batch == size.BATCH_CAP for e in short)
    # constant hidden size: heads = 2048 / hdim
    assert all(e.n_heads_q == 2048 // e.hdim_qk for e in short)


def test_size_rule_batch_is_a_power_of_two_within_the_budget(prime):
    size = _block('size')
    for e in prime:
        assert e.batch & (e.batch - 1) == 0
        assert max(size.fwd_bytes(e), size.bwd_bytes(e)) <= size.BUDGET_BYTES
        if e.batch < size.BATCH_CAP:
            bigger = dataclasses.replace(e, batch=e.batch * 2)
            assert max(size.fwd_bytes(bigger), size.bwd_bytes(bigger)) > size.BUDGET_BYTES


def test_size_rule_is_independent_of_the_gpu_and_deterministic(desc, prime):
    again = list(desc.entries(desc.entry_sets()['prime'], arch='gfx1100'))
    assert again == prime


def test_size_rule_gqa_keeps_heads_a_multiple_of_the_ratio(desc):
    problem = {'T_io': 'bf16', 'hdim_qk': 128, 'hdim_v': 128, 'causal': 0, 'window_left': -1,
               'window_right': -1, 'dropout_p': 0.0, 'bias_type': 0, 'bias_shape': '-', 'gqa_ratio': 4,
               'varlen': 0, 'storage_flip': 0}
    batch, heads = desc.size(problem, 1024, 1024)
    assert heads % 4 == 0 and heads == 16 and batch >= 1


def test_size_rule_halves_heads_then_gives_up():
    size = _block('size')
    calls = []

    def make(batch, heads):
        calls.append((batch, heads))
        return _Fake(batch, heads, fits=heads <= 8)

    class _Fake:
        def __init__(self, batch, heads, fits):
            self.batch, self.heads, self.fits = batch, heads, fits

    # fwd/bwd bytes of a stand-in entry: huge unless heads <= 8 and batch == 1
    real = (size.fwd_bytes, size.bwd_bytes)
    size.fwd_bytes = lambda e: 0 if e.fits and e.batch == 1 else size.BUDGET_BYTES + 1
    size.bwd_bytes = size.fwd_bytes
    try:
        assert size.size_rule(make, 64, 1) == (1, 8)
        assert size.size_rule(lambda b, h: _Fake(b, h, False), 64, 1) is None
    finally:
        size.fwd_bytes, size.bwd_bytes = real


# -- the byte model ---------------------------------------------------------------------------

def _entry(**kw):
    base = {'T_io': 'fp16', 'hdim_qk': 64, 'hdim_v': 64, 'causal': 0, 'window_left': -1, 'window_right': -1,
            'dropout_p': 0.0, 'bias_type': 0, 'bias_shape': '-', 'gqa_ratio': 1, 'varlen': 0,
            'storage_flip': 0, 'seqlen_q': 128, 'seqlen_k': 128, 'batch': 2, 'n_heads_q': 4}
    base.update(kw)
    return _block('entry').FlashPerfEntry(**base)


def test_byte_model_hand_computed():
    size = _block('size')
    e = _entry()
    q = 2 * 4 * 128 * 64          # elements of Q (= K = V = O here)
    lse = 2 * 4 * 128
    assert size.fwd_bytes(e) == 2 * 4 * q + 4 * lse + 4
    assert size.bwd_bytes(e) == 2 * 2 * 4 * q + 8 * lse + 4 * q + 4
    assert size.input_bytes(e, 'op_attn_fwd') == size.fwd_bytes(e)
    assert size.input_bytes(e, 'op_attn_bwd') == size.bwd_bytes(e)
    with pytest.raises(ValueError):
        size.input_bytes(e, 'op_attn_zz')


def test_byte_model_counts_the_real_bias_dimensions_padded_to_8():
    size = _block('size')
    plain = size.fwd_bytes(_entry(seqlen_k=100))
    full = size.fwd_bytes(_entry(seqlen_k=100, bias_type=1, bias_shape='bhqk')) - plain
    row = size.fwd_bytes(_entry(seqlen_k=100, bias_type=1, bias_shape='b1qk')) - plain
    flat = size.fwd_bytes(_entry(seqlen_k=100, bias_type=1, bias_shape='11qk')) - plain
    assert full == 2 * (2 * 4 * 128 * 104)        # fp16 x B x H x Sq x round8(Sk)
    assert row == 2 * (2 * 1 * 128 * 104)
    assert flat == 2 * (1 * 1 * 128 * 104)


def test_byte_model_gqa_shrinks_k_and_v_only():
    size = _block('size')
    assert size.fwd_bytes(_entry(gqa_ratio=4)) < size.fwd_bytes(_entry())


def test_byte_model_fp32_doubles_the_element_size():
    size = _block('size')
    e16, e32 = _entry(), _entry(T_io='fp32')
    lse = 4 * (2 * 4 * 128) + 4
    assert size.fwd_bytes(e32) - lse == 2 * (size.fwd_bytes(e16) - lse)


def test_fit_check_excludes_instead_of_shrinking(desc):
    size = _block('size')
    e = _entry(batch=4096, seqlen_q=4096, seqlen_k=4096)
    ok, reason = desc.fits_vram(e, 'op_attn_fwd', 16 * size.GiB)
    assert not ok and 'GiB' in reason
    assert desc.fits_vram(_entry(), 'op_attn_bwd', 16 * size.GiB) == (True, '')


# -- hardware limits ------------------------------------------------------------------------------

def test_validate_hw_feature(desc, prime):
    big = [e for e in prime if e.hdim_qk == 512][0]
    small = [e for e in prime if e.hdim_qk == 64][0]
    ok, reason = desc.validate_hw_feature('gfx1100', big)
    assert not ok and 'gfx1100' in reason
    assert desc.validate_hw_feature('gfx1100', small) == (True, '')
    assert desc.validate_hw_feature('gfx942', big) == (True, '')


# -- the wire form ---------------------------------------------------------------------------------

def test_entry_pon_has_exactly_the_runner_keys_in_order(desc, prime):
    for e in (prime[0], prime[-1]):
        wire = desc.entry_pon(e)
        assert ' ' not in wire
        assert tuple(parse_pon(wire)) == PON_KEYS
        assert parse_pon(wire)['seqlen_q'] == e.seqlen_q


def test_entry_dataclass_fields_are_the_pon_keys(desc):
    assert desc.entry_fields() == PON_KEYS
    assert desc.PROBLEM_COLUMNS + desc.SIZE_COLUMNS == PON_KEYS


def test_no_two_prime_entries_share_a_pon(desc, prime):
    assert len({desc.entry_pon(e) for e in prime}) == len(prime)


def test_entry_payload_roundtrip(desc, prime):
    for e in prime[::50]:
        assert desc.entry_from_dict(desc.entry_to_dict(e)) == e
    with pytest.raises(ValueError):
        desc.entry_from_dict({**desc.entry_to_dict(prime[0]), 'bogus': 1})


def test_problem_and_size_rows_roundtrip(desc, prime):
    e = prime[-1]
    assert desc.entry_from_rows(desc.problem_row(e), desc.size_row(e)) == e
    assert desc.problem_row(e)[6] == '0.1'
    assert desc.problem_row(prime[0])[6] == '0'      # canonical: 0.0 and 0 are one cell


# -- workloads.csv -------------------------------------------------------------------------------------

def test_workloads_are_the_first_list(desc):
    cases = desc.llm_cases()
    assert len(cases) == 10
    assert [c.model for c in cases[:4]] == ['roberta-large', 'bert-large-uncased',
                                            'distilbert-base-uncased', 'gpt2']
    assert all(c.phase == 'fwd_bwd' and not c.pending for c in cases)
    sweep = [c.entry for c in cases if c.model == 'roberta-large']
    assert len(sweep) == 7
    # the sweep keeps batch x seqlen
    assert {e.batch * e.seqlen_q for e in sweep} == {512 * 384}
    assert all(e.bias_shape == 'b1qk' and e.dropout_p == 0.1 and e.T_io == 'fp16' for e in sweep)
    gpt2 = cases[3].entry
    assert (gpt2.batch, gpt2.n_heads_q, gpt2.seqlen_q, gpt2.causal) == (72, 12, 1024, 1)


def test_workloads_columns_and_entries_validate():
    workloads = _block('workloads')
    with open(_FLASH_PERFMON / 'workloads.csv', newline='') as f:
        header = next(csv.reader(f))
    assert set(header) == set(workloads.COLUMNS) and len(header) == len(workloads.COLUMNS)
    rows = workloads.read_workloads()
    for row in rows:
        entry = workloads.row_entry(row)
        assert entry.n_heads_q == row['n_heads_q']
        assert entry.n_heads_kv == row['n_heads_kv']


def test_workloads_reject_a_bad_file(tmp_path):
    workloads = _block('workloads')
    path = tmp_path / 'w.csv'
    path.write_text('model,phase\nx,fwd\n')
    with pytest.raises(ValueError):
        workloads.read_workloads(path)
    text = (_FLASH_PERFMON / 'workloads.csv').read_text().splitlines()
    path.write_text('\n'.join([text[0], text[1].replace('fwd_bwd', 'bwd')]) + '\n')
    with pytest.raises(ValueError, match='phase'):
        workloads.read_workloads(path)


def test_workloads_errors_name_the_real_line_and_validate_head_counts(tmp_path):
    workloads = _block('workloads')
    text = (_FLASH_PERFMON / 'workloads.csv').read_text().splitlines()
    header = text[0].split(',')
    good = text[1].split(',')
    i_kv = header.index('n_heads_kv')
    path = tmp_path / 'w.csv'

    def write(kv_heads):
        bad = list(good)
        bad[i_kv] = kv_heads
        # a comment and a blank line push the bad row to line 6
        path.write_text('\n'.join(['# c', text[0], '', '# c2', ','.join(good), ','.join(bad)]) + '\n')

    write('0')
    with pytest.raises(ValueError, match=r'w\.csv, line 6: n_heads_kv must be positive'):
        workloads.read_workloads(path)
    write('5')
    with pytest.raises(ValueError, match=r'line 6: n_heads_q must be a multiple'):
        workloads.read_workloads(path)


def test_phase_ifaces(desc):
    assert desc.phase_ifaces('fwd') == ('op_attn_fwd',)
    assert desc.phase_ifaces('fwd_bwd') == ('op_attn_fwd', 'op_attn_bwd')
    with pytest.raises(ValueError):
        desc.phase_ifaces('bwd')
