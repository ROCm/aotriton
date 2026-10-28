# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for python/database/csvdb.py, the code generator's reader of the
CSV tuning database, on a small synthetic database. The real database is
covered end to end by comparing generated code with a reference build."""

from pathlib import Path
from types import SimpleNamespace
import pytest

from aotriton.database.csvdb import Database, tensor_path, tuning_source

SHEBANG = '#!/usr/bin/env -S python -m aotriton.database.interpreter\n'
KEYS = '#! tuning keys\nq,16,32\nk,16,32\n'


class Choice:
    """A stand-in TypedChoice: a tensor dtype or a constexpr value"""
    def __init__(self, value, is_tensor=False):
        self.triton_compile_signature = value
        self.is_tensor = is_tensor


def make_db(root):
    db = root / 'fam' / 'database'
    meta = db / '.metadata'
    (meta / 'k').mkdir(parents=True)
    (meta / 'op').mkdir(parents=True)
    (meta / 'copts.csv').write_text('waves_per_eu,num_warps,num_stages\n2,4,1\n1,4,1\n')
    (meta / 'k' / 'functional.csv').write_text('T_io,HDIM,PAD,EXTRA\nfp16,64,0,0\nbf16,64,0,0\nfp16,64,1,0\n')
    (meta / 'k' / 'psels.csv').write_text('BLOCK_M\n16\n32\n')
    (meta / 'k' / 'hsaco_options.csv').write_text('psels,copts\n1,1\n2,2\n')
    (meta / 'op' / 'functional.csv').write_text('T_io,HDIM,PAD\nfp16,64,0\n')
    (meta / 'op' / 'op_backend.csv').write_text('name,bord\ntriton,1\naiter,2\nflyc,3\n')
    gfx = db / 'amd' / 'gfx942'
    (gfx / 'mod1').mkdir(parents=True)
    dense = '#! tuning tensor\nfunctional,tuning keys->hsaco_options\n'
    (gfx / 'k.txtdb').write_text(SHEBANG + KEYS + dense + '1,\n1,2\n2,\n2,\n1[2]\n,1\n')
    (gfx / 'mod1' / 'k.txtdb').write_text(SHEBANG + KEYS + dense + '1,\n2,2\n2,2\n')
    (gfx / 'k.0001-p.dbpatch').write_text(SHEBANG + KEYS + '#! sparse tuning tensor\n'
                                          'functional,hsaco_options<-coordinates\n1,\n2<-1{1}\n')
    (gfx / 'op.txtdb').write_text(SHEBANG + KEYS + '#! precedence\nbits_per_slot,4\nchars_per_slot,1\n'
                                '#! tuning tensor\nfunctional,tuning keys->precedence\n1,\n12,1\n3,31\n')
    return root


def make_functional(name, choices, fallback=None, gpus=('gfx942_mod0',)):
    axes = [SimpleNamespace(signature_name='Q', var_name='T_io', kind='tensor'),
            SimpleNamespace(signature_name='HDIM', var_name='HDIM', kind='scalar'),
            SimpleNamespace(signature_name='PAD', var_name='PAD', kind='scalar')]
    meta = SimpleNamespace(NAME=name, axes_multi=axes)
    compact = {'Q': Choice(f'*{choices[0]}:16', is_tensor=True), 'HDIM': Choice(choices[1]),
               'PAD': Choice(choices[2])}
    fb = dict(compact)
    fb.update(fallback or {})
    return SimpleNamespace(meta_object=meta, compact_choices=compact, fallback_choices=fb,
                           optimized_for=list(gpus), unified_signature=f'sig-{name}-{choices}')


@pytest.fixture
def db(tmp_path):
    Database._cache.clear()
    return Database.for_module(make_db(tmp_path) / 'fam')


def test_gpu2archmod():
    from aotriton.gpu_targets import gpu2archmod
    assert gpu2archmod('gfx942_mod0') == ('gfx942', 0)
    assert gpu2archmod('gfx942_mod12') == ('gfx942', 12)
    for bad in ('gfx942', 'gfx942_modx', 'gfx942_1'):
        with pytest.raises(ValueError):
            gpu2archmod(bad)


def test_tensor_path():
    assert tensor_path('gfx942_mod0', 'k') == Path('amd/gfx942/k.txtdb')
    assert tensor_path('gfx942_mod1', 'k') == Path('amd/gfx942/mod1/k.txtdb')


def test_match_on_axis_columns_only(db):
    # EXTRA is not an axis: it is not compared
    view = db.lookup(make_functional('k', ('fp16', 64, False)))
    assert view.ordinal == 1 and not view.fallback
    assert view.slabs[0].cells.tolist() == [[1, 2], [2, 2]]       # 1{1} patched to 2
    assert view.hsaco_option(2) == ({'BLOCK_M': 32}, {'waves_per_eu': 1, 'num_warps': 4, 'num_stages': 1})


def test_patches_listed_only_with_a_block(db):
    v1 = db.lookup(make_functional('k', ('fp16', 64, False)))
    v2 = db.lookup(make_functional('k', ('bf16', 64, False)))
    assert v1.slabs[0].patches == [Path('amd/gfx942/k.0001-p.dbpatch')]
    assert v2.slabs[0].patches == []


def test_fallback(db):
    # PAD=1 has a row (ordinal 3) but no tuned cells -> fall back to PAD=0
    view = db.lookup(make_functional('k', ('fp16', 64, True), fallback={'PAD': False}))
    assert view.ordinal == 1 and view.fallback and not view.empty


def test_no_row_and_no_fallback_is_empty(db):
    view = db.lookup(make_functional('k', ('fp32', 64, False)))
    assert view.ordinal is None and view.empty


def test_no_tensor_file_is_none(db):
    f = make_functional('absent', ('fp16', 64, False))
    assert db.lookup(f) is None
    assert tuning_source(None, f) == f'tuning: no database for absent ({f.unified_signature})'


def test_mod_gpus(db):
    f = make_functional('k', ('fp16', 64, False), gpus=('gfx942_mod0', 'gfx942_mod1', 'gfx942_mod2'))
    view = db.lookup(f)
    assert view.slabs[1].path == Path('amd/gfx942/mod1/k.txtdb')
    assert view.slabs[1].cells.tolist() == [[2, 2], [2, 2]]
    assert view.slabs[2] is None


def test_precedence(db):
    view = db.lookup(make_functional('op', ('fp16', 64, False)))
    cells = view.slabs[0].cells
    # "12" -> bords (2, 1), "31" -> bords (1, 3); most preferred first
    assert view.backends(cells[0, 0]) == ['aiter', 'triton']
    assert view.backends(cells[0, 1]) == ['triton']
    assert view.backends(cells[1, 1]) == ['triton', 'flyc']


def test_source(db):
    f = make_functional('k', ('fp16', 64, True), fallback={'PAD': False})
    lines = tuning_source(db.lookup(f), f).split('\n// ')
    assert lines == [f'tuning: amd/gfx942/k.txtdb functional 1, fallback ({f.unified_signature})',
                     'tuning: possible patch amd/gfx942/k.0001-p.dbpatch']
