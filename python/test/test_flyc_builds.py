# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A flyc description's builds: naming one by its build knobs, finding it
again, and the selector the code generator emits over their boxes."""

import pytest

from aotriton.codegen.flytune import FlycTuneCodeGenerator
from aotriton.template_instantiation.ir.flyc.builds import (
    build_knobs, flyc_build_for, flyc_builds, render_build_knobs,
)

INT32_MIN, INT32_MAX = -2**31, 2**31 - 1
COMMON = {'flyc_source': 'gfx950/fwd.py', 'block_m': 128}


def _result():
    return iter([
        ([{'hdim_qk': (INT32_MIN, 96), 'max_seqlen_k': (INT32_MIN, 128)}], 'b0',
         dict(COMMON, hdim_qk_floor=0, exact_kv_tiles=True)),
        ([{'hdim_qk': (INT32_MIN, 96), 'max_seqlen_k': (129, INT32_MAX)}], 'b1',
         dict(COMMON, hdim_qk_floor=0, exact_kv_tiles=False)),
        ([{'hdim_qk': (97, INT32_MAX)}], 'b2',
         dict(COMMON, hdim_qk_floor=96, exact_kv_tiles=True)),
    ])


def test_a_single_build_serves_every_call():
    assert flyc_builds(('b', {'k': 1})) == [([{}], 'b', {'k': 1})]
    assert build_knobs([{'k': 1}]) == [{}]
    assert flyc_build_for(('b', {'k': 1}), '') == ('b', {'k': 1})


def test_build_knobs_are_the_knobs_that_differ():
    named = build_knobs([k for _, _, k in _result()])
    assert named == [{'exact_kv_tiles': True, 'hdim_qk_floor': 0},
                     {'exact_kv_tiles': False, 'hdim_qk_floor': 0},
                     {'exact_kv_tiles': True, 'hdim_qk_floor': 96}]
    assert render_build_knobs(named[1]) == 'exact_kv_tiles=False hdim_qk_floor=0'


def test_a_build_is_found_by_its_build_knobs():
    build, knobs = flyc_build_for(_result(), 'exact_kv_tiles=True hdim_qk_floor=96')
    assert build == 'b2' and knobs['hdim_qk_floor'] == 96


def test_build_knobs_must_name_exactly_one_build():
    with pytest.raises(ValueError, match='names 0'):
        flyc_build_for(_result(), '')
    with pytest.raises(ValueError, match='names 0'):
        flyc_build_for(_result(), 'exact_kv_tiles=False hdim_qk_floor=96')


def test_selector_conditions_omit_the_domain_ends():
    cxx = FlycTuneCodeGenerator._cxx_boxes
    assert cxx([{'max_seqlen_k': (INT32_MIN, 128)}]) == 'max_seqlen_k <= 128'
    assert cxx([{'max_seqlen_k': (129, INT32_MAX)}]) == 'max_seqlen_k > 128'
    assert cxx([{'hdim_qk': (97, INT32_MAX), 'max_seqlen_k': (65, 128)}]) == \
        'hdim_qk > 96 && max_seqlen_k > 64 && max_seqlen_k <= 128'
    assert cxx([{'max_seqlen_k': (INT32_MIN, 64)}, {'max_seqlen_k': (129, INT32_MAX)}]) == \
        '(max_seqlen_k <= 64) || (max_seqlen_k > 128)'
    # The rounder constant: inclusive at a power of two, strict next to one.
    assert cxx([{'max_seqlen_k': (1024, INT32_MAX)}]) == 'max_seqlen_k >= 1024'
    assert cxx([{'max_seqlen_k': (INT32_MIN, 1023)}]) == 'max_seqlen_k < 1024'
    assert cxx([{'max_seqlen_q': (INT32_MIN, 511)}]) == 'max_seqlen_q < 512'
    assert cxx([{}]) == 'true'
