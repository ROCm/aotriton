# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests of the tuning tensor file interpreter. Only numpy is needed, except for
the cross-check against tools/database_to_csv.py, which runs on the real
database and is skipped when the tool or the database is absent."""

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aotriton.database import interpreter as it

SHEBANG = '#!/usr/bin/env -S python -m aotriton.database.interpreter\n'
KEYS2 = '#! tuning keys\nq,16,32\nk,16,32,64\n'
KEYS3 = '#! tuning keys\nq,16,32\nk,16,32,64\nh,64,128,256\n'
PREC = '#! precedence\nbits_per_slot,4\nchars_per_slot,1\n'


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def dense2(header='hsaco_options'):
    return (KEYS2 + f'#! tuning tensor\nfunctional,tuning keys->{header}\n'
            '1,\n1,2,3\n4,5,6\n2,\n7[3]\n[2],9\n')


def test_dense_rank2_shebang_rle(tmp_path):
    p = write(tmp_path, 'attn_fwd.csv', SHEBANG + '# comment\n\n' + dense2())
    t = it.load(p)
    assert t.tuning_keys == ['q', 'k']
    assert t.ladders == [(16, 32), (16, 32, 64)]
    assert t.functional.tolist() == [1, 2]
    assert t.tensor.shape == (2, 2, 3)
    assert t.tensor.tolist() == [[[1, 2, 3], [4, 5, 6]], [[7, 7, 7], [0, 0, 9]]]
    assert t.tensor.dtype == np.uint8
    assert t.target == 'hsaco_options'
    assert t.bits_per_slot is None and t.chars_per_slot is None


def test_no_shebang_is_fine(tmp_path):
    assert it.load(write(tmp_path, 'attn_fwd.csv', dense2())).tensor.shape == (2, 2, 3)


def test_dense_rank3_blank_lines(tmp_path):
    text = (KEYS3 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n'
            '1,\n10,10,11\n10,10,11\n10,10,11\n\n11,11,12\n11,11,12\n11,11,12\n')
    t = it.load(write(tmp_path, 'k.csv', text))
    assert t.tensor.shape == (1, 2, 3, 3)
    assert t.tensor[0, 0, 0].tolist() == [10, 10, 11]
    assert t.tensor[0, 1, 2].tolist() == [11, 11, 12]


def test_dtype_selection(tmp_path):
    for value, dtype in ((255, np.uint8), (256, np.uint16), (65536, np.uint32)):
        text = KEYS2 + f'#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n{value}[3]\n1[3]\n'
        assert it.load(write(tmp_path, 'k.csv', text)).tensor.dtype == dtype


def test_precedence_packing(tmp_path):
    text = (KEYS2 + PREC + '#! tuning tensor\nfunctional,tuning keys->precedence\n'
            '1,\n1AF,12,2\n,,F\n')
    t = it.load(write(tmp_path, 'op.csv', text))
    # bords (15, 10, 1), most preferred first
    assert t.tensor[0, 0, 0] == 0x1AF
    assert t.tensor[0, 0, 1] == 0x12      # bords (2, 1)
    assert t.tensor[0, 0, 2] == 2
    assert t.tensor[0, 1, 2] == 15
    assert t.tensor[0, 1, 0] == 0
    assert (t.bits_per_slot, t.chars_per_slot) == (4, 1)
    assert t.tensor.dtype == np.uint16


def test_precedence_chars_per_slot_2_and_roundtrip():
    assert it.encode_precedence([1, 5], 2) == '0501'
    assert it.decode_precedence('0501', 2) == [1, 5]
    assert it.pack_precedence([1, 5], 6) == 1 | (5 << 6)
    for bords in ([15, 10, 1], [61], [3, 2, 1]):
        assert it.decode_precedence(it.encode_precedence(bords)) == bords


def test_precedence_chars2_file(tmp_path):
    text = (KEYS2.replace('k,16,32,64', 'k,16') + 'x'[:0] + '#! precedence\nbits_per_slot,6\n'
            'chars_per_slot,2\n#! tuning tensor\nfunctional,tuning keys->precedence\n1,\n0501\n01\n')
    t = it.load(write(tmp_path, 'op.csv', text))
    assert t.tensor[0, :, 0].tolist() == [1 | (5 << 6), 1]


def patch_text(body, keys=KEYS2, target='hsaco_options'):
    return keys + f'#! tuning tensor\nfunctional,tuning keys->{target}\n' + body


def test_discovery_order_and_dense_patches(tmp_path):
    base = write(tmp_path, 'attn_fwd.csv', dense2())
    write(tmp_path, 'attn_fwd.0002-b.dbpatch', patch_text('1,\n[3]\n,,50\n'))
    write(tmp_path, 'attn_fwd.0001-a.dbpatch', patch_text('1,\n,20,\n,,40\n3,\n8,8,8\n,,\n'))
    write(tmp_path, 'other.0001-a.dbpatch', 'garbage')
    assert [p.name for p in it.patch_files(base)] == ['attn_fwd.0001-a.dbpatch', 'attn_fwd.0002-b.dbpatch']
    t = it.load(base)
    assert t.functional.tolist() == [1, 2, 3]
    # 0001 sets (0,0,1)=20 and (0,1,2)=40; 0002 overrides (0,1,2)=50
    assert t.tensor[0].tolist() == [[1, 20, 3], [4, 5, 50]]
    assert t.tensor[1].tolist() == [[7, 7, 7], [0, 0, 9]]
    assert t.tensor[2].tolist() == [[8, 8, 8], [0, 0, 0]]


def test_patch_alone_and_into_base(tmp_path):
    base = it.load(write(tmp_path, 'attn_fwd.csv', dense2()))
    p = write(tmp_path, 'attn_fwd.0001-a.dbpatch', patch_text('2,\n,,\n300,,\n'))
    s = it.load(p)
    assert isinstance(s, it.SparsePatch)
    assert s.sparse.tolist() == [[2, 1, 0, 300]]
    out = it.load(p, base=base)
    assert out is base
    assert base.tensor.dtype == np.uint16
    assert base.tensor[1, 1, 0] == 300 and base.tensor[1, 1, 2] == 9


def test_sparse_patch_partial_and_new_functional(tmp_path):
    write(tmp_path, 'attn_fwd.csv', dense2())
    text = (KEYS2 + '#! sparse tuning tensor\nfunctional,hsaco_options<-coordinates\n'
            '1,\n95<-0{0}1{1,2}\n33<-0{2}\n9,\n5<-1{0-1}\n')
    write(tmp_path, 'attn_fwd.0001-s.dbpatch', text)
    t = it.load(tmp_path / 'attn_fwd.csv')
    assert t.functional.tolist() == [1, 2, 9]
    assert t.tensor[0].tolist() == [[95, 2, 33], [4, 95, 95]]
    assert t.tensor[2].tolist() == [[0, 0, 0], [5, 5, 0]]
    assert t.tensor[1].tolist() == [[7, 7, 7], [0, 0, 9]]


def test_sparse_rank3_and_whitespace(tmp_path):
    base = write(tmp_path, 'k.csv', KEYS3 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n'
                 + '1[3]\n' * 6)
    p = write(tmp_path, 'k.0001-s.dbpatch', KEYS3 + '#! sparse tuning tensor\n'
              'functional,hsaco_options<-coordinates\n1,\n7 <- 0,2{0-1, 2}1,0{1}\n')
    t = it.load(base)
    it.load(p, base=t)
    assert t.tensor[0, 0, 2].tolist() == [7, 7, 7]
    assert t.tensor[0, 1, 0].tolist() == [1, 7, 1]
    assert (t.tensor == 7).sum() == 4


def test_sparse_matches_dense(tmp_path):
    dense = write(tmp_path, 'a.0001-d.dbpatch', patch_text('1,\n,5,\n6,,\n'))
    sparse = write(tmp_path, 'b.0001-s.dbpatch', KEYS2 + '#! sparse tuning tensor\n'
                   'functional,hsaco_options<-coordinates\n1,\n5<-0{1}\n6<-1{0}\n')
    assert it.load(dense).sparse.tolist() == it.load(sparse).sparse.tolist()


def test_apply_api(tmp_path):
    t = it.load(write(tmp_path, 'k.csv', dense2()))
    other = it.load(write(tmp_path, 'k.0001-a.dbpatch', patch_text('2,\n,,\n,,200\n')))
    assert it.apply(t, other) is t
    assert t.tensor[1, 1, 2] == 200 and t.tensor.dtype == np.uint8


def test_apply_checks(tmp_path):
    t = it.load(write(tmp_path, 'k.csv', dense2()))
    fields = dict(tuning_keys=t.tuning_keys, ladders=t.ladders, target=t.target,
                  bits_per_slot=None, chars_per_slot=None, metadata=t.metadata)
    for sparse in ([[1, 2, 0, 5]], [[1, -1, 0, 5]], [[1, 0, 0, 0]], [[1, 0, 5]]):
        with pytest.raises(it.TensorFileError):
            it.apply(t, it.SparsePatch(**fields, sparse=np.array(sparse)))
    other = it.SparsePatch(**{**fields, 'ladders': [(16, 32), (16, 32, 65)]},
                           sparse=np.zeros((0, 4), dtype=np.int64))
    with pytest.raises(it.TensorFileError):
        it.apply(t, other)


REJECTED = {
    'no keys': '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n1\n',
    'no tensor': KEYS2,
    'both': dense2() + '#! sparse tuning tensor\nfunctional,hsaco_options<-coordinates\n',
    'bad header': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->foo\n1,\n',
    'sparse header on dense': KEYS2 + '#! tuning tensor\nfunctional,hsaco_options<-coordinates\n',
    'truncated': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n1,2,3\n',
    'short row': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n1,2\n1,2,3\n',
    'long row': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n1,2,3,4\n1,2,3\n',
    'no functional line': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,2,3\n1,2,3\n1,2,3\n',
    'duplicate functional': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n'
                            '1,\n1[3]\n1[3]\n1,\n1[3]\n1[3]\n',
    'non-ordinal cell': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\nx[3]\n1[3]\n',
    'zero cell': KEYS2 + '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n1,\n0[3]\n1[3]\n',
    'bad ladder': '#! tuning keys\nq,16,x\n#! tuning tensor\nfunctional,tuning keys->hsaco_options\n',
    'descending ladder': '#! tuning keys\nq,32,16\n#! tuning tensor\nfunctional,tuning keys->hsaco_options\n',
    'duplicate key': '#! tuning keys\nq,1\nq,2\n#! tuning tensor\nfunctional,tuning keys->hsaco_options\n',
    'precedence section on kernel': PREC + dense2(),
    'precedence without section': dense2('precedence'),
    'precedence missing chars': KEYS2 + '#! precedence\nbits_per_slot,4\n#! tuning tensor\n'
                                'functional,tuning keys->precedence\n',
    'precedence bord too wide for bits': KEYS2 + PREC + '#! tuning tensor\nfunctional,tuning keys->precedence\n'
                                         '1,\nG[3]\n1[3]\n',
    'precedence not base62': KEYS2 + PREC + '#! tuning tensor\nfunctional,tuning keys->precedence\n1,\n-[3]\n1[3]\n',
    'precedence with 0 bord': KEYS2 + PREC + '#! tuning tensor\nfunctional,tuning keys->precedence\n1,\n10[3]\n1[3]\n',
    'precedence not multiple of chars': KEYS2 + '#! precedence\nbits_per_slot,6\nchars_per_slot,2\n'
                                        '#! tuning tensor\nfunctional,tuning keys->precedence\n1,\n123[3]\n1[3]\n',
    'sparse csv': KEYS2 + '#! sparse tuning tensor\nfunctional,hsaco_options<-coordinates\n1,\n5<-0{0}\n',
}


@pytest.mark.parametrize('name', sorted(REJECTED))
def test_rejected_csv(tmp_path, name):
    with pytest.raises(it.TensorFileError):
        it.load(write(tmp_path, 'k.csv', REJECTED[name]))


def sparse_text(lines):
    return KEYS2 + '#! sparse tuning tensor\nfunctional,hsaco_options<-coordinates\n' + lines


SPARSE_REJECTED = {
    'value before functional': '5<-0{0}\n',
    'cell listed twice': '1,\n5<-0{0}\n6<-0{0}\n',
    'cell listed twice in a group': '1,\n5<-0{0-1,1}\n',
    'row out of bounds': '1,\n5<-2{0}\n',
    'column out of bounds': '1,\n5<-0{3}\n',
    'too few indices': '1,\n5<-{0}\n',
    'too many indices': '1,\n5<-0,0{0}\n',
    'malformed cells': '1,\n5<-0(0)\n',
    'empty range': '1,\n5<-0{2-1}\n',
    'duplicate functional': '1,\n5<-0{0}\n1,\n5<-0{1}\n',
    'bad value': '1,\nx<-0{0}\n',
    'zero value': '1,\n0<-0{0}\n',
    'garbage line': '1,\nfoo\n',
}


@pytest.mark.parametrize('name', sorted(SPARSE_REJECTED))
def test_rejected_sparse_patch(tmp_path, name):
    with pytest.raises(it.TensorFileError):
        it.load(write(tmp_path, 'k.0001-s.dbpatch', sparse_text(SPARSE_REJECTED[name])))


def test_patch_mismatches(tmp_path):
    base = write(tmp_path, 'k.csv', dense2())
    other_keys = KEYS2.replace('64', '65')
    cases = {
        'ladder': patch_text('1,\n1[3]\n1[3]\n', keys=other_keys),
        'key name': patch_text('1,\n1[3]\n1[3]\n', keys=KEYS2.replace('q,', 'z,')),
        'target': patch_text('1,\n1[3]\n1[3]\n', target='precedence'),
    }
    for name, text in cases.items():
        patch = write(tmp_path, f'k.0001-{name.replace(" ", "")}.dbpatch', text)
        # a precedence patch lacks the precedence section, so it fails to parse
        # or to match; either way it is rejected
        with pytest.raises(it.TensorFileError):
            it.load(base)
        patch.unlink()
    write(tmp_path, 'k.0001-meta.dbpatch', KEYS2 + '#! extra\na,1\n#! tuning tensor\n'
          'functional,tuning keys->hsaco_options\n1,\n1[3]\n1[3]\n')
    with pytest.raises(it.TensorFileError):
        it.load(base)


def test_patch_precedence_bits_mismatch(tmp_path):
    body = '#! tuning tensor\nfunctional,tuning keys->precedence\n1,\n1[3]\n1[3]\n'
    write(tmp_path, 'o.csv', KEYS2 + PREC + body)
    write(tmp_path, 'o.0001-p.dbpatch', KEYS2 + PREC.replace(',4', ',5') + body)
    with pytest.raises(it.TensorFileError):
        it.load(tmp_path / 'o.csv')


def test_bad_patch_names(tmp_path):
    base = write(tmp_path, 'k.csv', dense2())
    write(tmp_path, 'k.1-a.dbpatch', patch_text('1,\n1[3]\n1[3]\n'))
    with pytest.raises(it.TensorFileError):
        it.load(base)
    (tmp_path / 'k.1-a.dbpatch').unlink()
    for name in ('k.0001-a.dbpatch', 'k.00001-b.dbpatch'):   # the same number
        write(tmp_path, name, patch_text('1,\n1[3]\n1[3]\n'))
    with pytest.raises(it.TensorFileError):
        it.load(base)


def test_base_rejected_for_csv(tmp_path):
    t = it.load(write(tmp_path, 'k.csv', dense2()))
    with pytest.raises(ValueError):
        it.load(tmp_path / 'k.csv', base=t)


def base_ref(rel):
    return SHEBANG + f'#! base\n{rel}\n'


def test_base_reference(tmp_path):
    # b references a; a's own patch is inherited, b's patch applies on top
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    write(tmp_path / 'a', 'k.csv', dense2())
    write(tmp_path / 'a', 'k.0001-a.dbpatch', patch_text('1,\n,7,\n,,\n'))
    write(tmp_path / 'b', 'k.csv', base_ref('../a/k.csv'))
    write(tmp_path / 'b', 'k.0001-b.dbpatch', patch_text('1,\n99,,\n,,\n'))
    a = it.load(tmp_path / 'a' / 'k.csv').tensor
    b = it.load(tmp_path / 'b' / 'k.csv').tensor
    assert a[0, 0, 0] == 1 and a[0, 0, 1] == 7
    assert b[0, 0, 0] == 99 and b[0, 0, 1] == 7
    assert [p.name for p in it.applied_patches(tmp_path / 'b' / 'k.csv')] == ['k.0001-a.dbpatch', 'k.0001-b.dbpatch']
    assert [p.relative_to(tmp_path).as_posix() for p in it.source_files(tmp_path / 'b' / 'k.csv')] == \
        ['b/k.csv', 'a/k.csv', 'a/k.0001-a.dbpatch', 'b/k.0001-b.dbpatch']


def test_base_reference_chain_and_mod(tmp_path):
    (tmp_path / 'a' / 'mod1').mkdir(parents=True)
    (tmp_path / 'c').mkdir()
    write(tmp_path / 'a', 'k.csv', dense2())
    write(tmp_path / 'a' / 'mod1', 'k.csv', base_ref('../k.csv'))
    write(tmp_path / 'c', 'k.csv', base_ref('../a/mod1/k.csv'))
    assert it.load(tmp_path / 'c' / 'k.csv').tensor.tolist() == it.load(tmp_path / 'a' / 'k.csv').tensor.tolist()


@pytest.mark.parametrize('name,text,message', [
    ('cycle', None, 'cycle'),
    ('missing', base_ref('../x/k.csv'), 'does not exist'),
    ('other iface', base_ref('../a/other.csv'), 'same interface'),
    ('two paths', SHEBANG + '#! base\n../a/k.csv\n../a/k.csv\n', 'exactly one path'),
    ('with a tensor', SHEBANG + '#! base\n../a/k.csv\n' + dense2(), 'nothing but'),
])
def test_base_reference_rejected(tmp_path, name, text, message):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    write(tmp_path / 'a', 'k.csv', dense2())
    write(tmp_path / 'a', 'other.csv', dense2())
    if text is None:    # a -> b -> a
        write(tmp_path / 'a', 'k.csv', base_ref('../b/k.csv'))
        text = base_ref('../a/k.csv')
    write(tmp_path / 'b', 'k.csv', text)
    with pytest.raises(it.TensorFileError, match=message):
        it.load(tmp_path / 'b' / 'k.csv')


def test_patch_cannot_be_base_reference(tmp_path):
    write(tmp_path, 'k.csv', dense2())
    write(tmp_path, 'k.0001-x.dbpatch', base_ref('k.csv'))
    with pytest.raises(it.TensorFileError):
        it.load(tmp_path / 'k.csv')


def test_npz_and_cli(tmp_path, monkeypatch, capsys):
    base = write(tmp_path, 'attn_fwd.csv', SHEBANG + dense2())
    patch = write(tmp_path, 'attn_fwd.0001-a.dbpatch', SHEBANG + patch_text('2,\n,,\n,,200\n'))
    out = tmp_path / 'x.npz'
    it.main([str(base), '--output', str(out)])
    with np.load(out) as z:
        assert sorted(z.files) == ['functional', 'k', 'q', 'target', 'tensor', 'tuning_keys']
        assert z['tensor'].shape == (2, 2, 3)
        assert z['functional'].tolist() == [1, 2]
        assert z['tuning_keys'].tolist() == ['q', 'k']
        assert z['q'].tolist() == [16, 32] and z['k'].tolist() == [16, 32, 64]
        assert z['target'] == 'hsaco_options'
    it.main([str(patch), '--output', str(out)])
    with np.load(out) as z:
        assert 'tensor' not in z.files
        assert z['sparse'].tolist() == [[2, 1, 2, 200]]
        assert z['q'].tolist() == [16, 32]
    # without --output, prompt
    monkeypatch.setattr('builtins.input', lambda prompt='': str(tmp_path / 'prompted.npz'))
    it.main([str(base)])
    assert (tmp_path / 'prompted.npz').exists()
    # precedence scalars
    op = write(tmp_path, 'op.csv', KEYS2 + PREC + '#! tuning tensor\nfunctional,tuning keys->precedence\n'
               '1,\n12[3]\n21[3]\n')
    it.main([str(op), '--output', str(out)])
    with np.load(out) as z:
        assert int(z['bits_per_slot']) == 4 and int(z['chars_per_slot']) == 1
        assert z['target'] == 'precedence'
        assert z['tensor'][0, 0, 0] == 0x12


def test_python_dash_m_and_lazy_import(tmp_path):
    p = write(tmp_path, 'attn_fwd.csv', SHEBANG + dense2())
    out = tmp_path / 'x.npz'
    code = ('import sys, runpy; sys.argv = ["x", %r, "--output", %r]; '
            'runpy.run_module("aotriton.database.interpreter", run_name="__main__"); '
            'assert "pandas" not in sys.modules and "aotriton.database.factories" not in sys.modules'
            % (str(p), str(out)))
    subprocess.run([sys.executable, '-c', code], check=True, cwd=tmp_path)
    assert out.exists()


# Cross-check against the reference reader on the real database
ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / 'tools' / 'database_to_csv.py'
DATABASE = ROOT / 'modules' / 'flash' / 'database' / 'amd'


def real_files():
    return sorted([*DATABASE.glob('gfx*/*.csv'), *DATABASE.glob('gfx*/mod*/*.csv')])


@pytest.mark.skipif(not TOOL.exists() or not real_files(), reason='needs the tool and the real database')
@pytest.mark.parametrize('fn', real_files(), ids=lambda p: str(p.relative_to(DATABASE)))
def test_real_database_matches_reference(fn, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.parent))
    import database_to_csv as ref
    r = ref.load_tensor(fn)
    t = it.load(fn)
    assert [k for k, _ in r.keys] == t.tuning_keys
    assert [tuple(int(v) for v in ladder) for _, ladder in r.keys] == t.ladders
    assert set(r.cells) == set(t.functional.tolist())
    bits = chars = None
    if r.target == 'precedence':
        section = dict(r.metadata['precedence'])
        bits, chars = int(section['bits_per_slot']), int(section['chars_per_slot'])
    assert (t.bits_per_slot, t.chars_per_slot) == (bits, chars)
    assert t.tensor.dtype == it.smallest_uint(int(t.tensor.max()))
    for i, fid in enumerate(t.functional.tolist()):
        prefixes = itertools.product(*[range(len(ladder)) for ladder in t.ladders[:-1]])
        for prefix, row in zip(prefixes, r.cells[fid], strict=True):
            for j, c in enumerate(row):
                if c == '':
                    expect = 0
                elif r.target == 'hsaco_options':
                    expect = int(c)
                else:
                    expect = sum(b << (bits * n) for n, b in enumerate(ref.decode_precedence(c, chars)))
                assert int(t.tensor[(i, *prefix, j)]) == expect, (fid, prefix, j)
