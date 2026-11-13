# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A flyc kernel enumerates its OWN BLOCK_DMODEL axis (its marker's `options=`,
narrowed per arch by its disable predicate), not the cited operator's.

The fake `flyc_attn_fwd` (fakefamily/flash/aot/flyc_attn_fwd.py) compiles 384 on
gfx942, a rung the operator axis does not have. The operator must keep binning
to its own axis; the flyc godel number, compiled-rung table and functionals
must agree on 384."""

import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from aotriton.codegen.flyc import FlycShimGenerator
from aotriton.codegen.linker import Linker
from aotriton.gpu_targets import cluster_gpus
from aotriton.utils import RegistryRepository

FAKE_MODULES = Path(__file__).resolve().parent / 'fakefamily'
TARGET_GPUS = ['gfx942_mod0', 'gfx950_mod0', 'gfx1201_mod0']
OP_HDIMS = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]
FLYC_HDIMS = [64, 128, 384, 512]


@lru_cache(maxsize=1)
def _linked():
    _k, ops, _a, flycs = Linker(FAKE_MODULES).link_all_families()
    op = next(o for o in ops if o.NAME == 'op_attn_fwd')
    flyc = next(k for k in flycs if k.NAME == 'flyc_attn_fwd')
    return op, flyc


def _tp(iface, name):
    return next(tp for tp in iface.list_functional_params() if tp.repr_name == name)


def _values(tp):
    return [c.triton_compile_signature for c in tp.choices]


def _enabled(flyc):
    return [f for f in flyc.gen_functionals(cluster_gpus(TARGET_GPUS))
            if not flyc.is_functional_disabled(f)]


def test_operator_axis_is_unchanged():
    op, flyc = _linked()
    _enabled(flyc)          # enumerating the flyc space must not touch the operator's
    assert _values(_tp(op, 'BLOCK_DMODEL')) == OP_HDIMS
    assert op.godel_number == 576
    assert _tp(op, 'BLOCK_DMODEL').godel_stride == 16


def test_flyc_axis_comes_from_the_marker():
    op, flyc = _linked()
    tp = _tp(flyc, 'BLOCK_DMODEL')
    assert _values(tp) == FLYC_HDIMS
    assert tp.repr_typed_choice.itype == _tp(op, 'BLOCK_DMODEL').repr_typed_choice.itype
    assert flyc.godel_number == 576 // len(OP_HDIMS) * len(FLYC_HDIMS)
    assert flyc.axis_by_var('BLOCK_DMODEL') is tp.axis
    assert flyc.axis_of_arg('BLOCK_DMODEL') is tp.axis
    # Signatures and entry names read axes_multi; it must be the flyc copy.
    assert tp.axis in flyc.axes_multi


def test_disable_narrows_per_arch():
    _op, flyc = _linked()
    by_arch = {}
    for f in _enabled(flyc):
        by_arch.setdefault(f.arch, set()).add(f.choices.BLOCK_DMODEL)
    assert by_arch == {'gfx942': {64, 128, 384, 512}, 'gfx950': {64, 128, 512}}


def test_godel_numbers_follow_the_flyc_strides():
    _op, flyc = _linked()
    tps = list(flyc.list_functional_params())
    seen = set()
    for f in _enabled(flyc):
        expected = sum(tp.axis.choices.index(f.choice[tp.axis.var_name]) * tp.godel_stride
                       for tp in tps if tp.radix > 1)
        assert f.godel_number == expected
        assert 0 <= f.godel_number < flyc.godel_number
        seen.add((f.arch, f.godel_number))
    assert len(seen) == len(_enabled(flyc))


def _generator(flyc):
    return FlycShimGenerator(SimpleNamespace(target_gpus=TARGET_GPUS),
                             flyc,
                             RegistryRepository())


def test_compiled_rung_table_has_384():
    _op, flyc = _linked()
    defs = _generator(flyc).codegen_compiled_rung_table_defs(_enabled(flyc))
    assert 'FlycAttnFwdContext_compiled_block_dmodel_0[] = { 64, 128, 384, 512 };' in defs
    assert 'FlycAttnFwdContext_compiled_block_dmodel_1[] = { 64, 128, 512 };' in defs
    assert 'FlycAttnFwdContext_compiled_block_dmodel_2[] = { -1 };' in defs
    assert 'compiled_block_dmodel_count[] = { 4, 3, 0 };' in defs


def test_godel_body_reads_the_flyc_ladder():
    _op, flyc = _linked()
    body = _generator(flyc).codegen_godel_number_body()
    assert 'if (scratch_params.flyc_block_dmodel == 384) number = 2 ;' in body
    assert 'flyc_block_dmodel == 16)' not in body


def _round_up(hdim, rungs):
    """flyc_round_up_rung (modules/flash/csrc/flyc_common.h)."""
    return next((r for r in rungs if r >= hdim), -1)


def test_selection_for_off_operator_head_dims():
    """Mirror lookup_optimal: helper rounds the true head dim on the arch's
    compiled row, godel_number() reads the result, autotune_table indexes it."""
    _op, flyc = _linked()
    enabled = {(f.arch, f.godel_number): f for f in _enabled(flyc)}
    tps = [tp for tp in flyc.list_functional_params() if tp.radix > 1]
    defs = _generator(flyc).codegen_compiled_rung_table_defs(_enabled(flyc))
    row = re.search(r'compiled_block_dmodel_0\[\] = \{ ([^}]*) \};', defs).group(1)
    gfx942_row = [int(v) for v in row.split(',')]
    cases = [(300, 384, True),
             (384, 384, False),
             (100, 128, True),
             (128, 128, False)]
    for hdim, rung, padded in cases:
        block = _round_up(hdim, gfx942_row)
        assert block == rung
        picked = {'BLOCK_DMODEL': block,
                  'PADDED_HEAD': block != hdim,
                  'T_io': '*fp16:16',
                  'ENABLE_DROPOUT': False,
                  'CAUSAL_TYPE': 0,
                  'BIAS_TYPE': 0}
        godel = 0
        for tp in tps:
            values = [c.triton_compile_signature for c in tp.choices]
            godel += values.index(picked[tp.axis.var_name]) * tp.godel_stride
        f = enabled[('gfx942', godel)]
        assert f.choices.BLOCK_DMODEL == rung
        assert f.choices.PADDED_HEAD is padded
