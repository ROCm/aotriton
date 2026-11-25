# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-call rules: `ati.derives` yielded by a tuning module, read per call
(`ir/runtime_rules.py`)."""

import pytest

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.ir.runtime_rules import box, builds, check, overrides_at, verify

INT32 = (-2**31, 2**31 - 1)
DOMAINS = {'seqlen_q': INT32, 'seqlen_k': INT32}
LO, HI = INT32


def _box(when):
    return box(ati.derives('x', to=1, when=when), DOMAINS)


def test_predicates_are_inclusive_intervals():
    assert _box(ati.le('seqlen_k', 128)) == {'seqlen_k': (LO, 128)}
    assert _box(ati.lt('seqlen_k', 128)) == {'seqlen_k': (LO, 127)}
    assert _box(ati.gt('seqlen_k', 128)) == {'seqlen_k': (129, HI)}
    assert _box(ati.ge('seqlen_k', 128)) == {'seqlen_k': (128, HI)}
    assert _box(ati.eq('seqlen_k', 128)) == {'seqlen_k': (128, 128)}
    assert _box(None) == {}


def test_a_tuple_of_predicates_is_their_conjunction():
    assert _box((ati.gt('seqlen_k', 64), ati.le('seqlen_k', 256), ati.ge('seqlen_q', 8))) == {
        'seqlen_k': (65, 256), 'seqlen_q': (8, HI)}
    with pytest.raises(ValueError, match='holds for no call'):
        _box((ati.gt('seqlen_k', 64), ati.le('seqlen_k', 64)))


def test_what_a_call_cannot_evaluate_is_rejected():
    with pytest.raises(ValueError, match='not a runtime value'):
        _box(ati.le('seqlen', 3))
    with pytest.raises(ValueError, match='==, <, <=, > or >='):
        _box(ati.ne('seqlen_k', 3))
    with pytest.raises(ValueError, match='not a callable'):
        _box(lambda f: True)
    with pytest.raises(ValueError, match='literal'):
        box(ati.derives('x', to='BLOCK_DMODEL', when=None), DOMAINS)


def test_overlapping_rules_on_one_target_are_rejected():
    with pytest.raises(ValueError, match='overlapping'):
        check([ati.derives('x', to=1, when=ati.gt('seqlen_k', 128)),
               ati.derives('x', to=2, when=ati.gt('seqlen_k', 192))], DOMAINS)
    # Different targets may overlap; touching boxes are disjoint.
    check([ati.derives('x', to=1, when=ati.gt('seqlen_k', 128)),
           ati.derives('y', to=2, when=ati.gt('seqlen_k', 192))], DOMAINS)
    check([ati.derives('x', to=1, when=ati.le('seqlen_k', 128)),
           ati.derives('x', to=2, when=ati.gt('seqlen_k', 128))], DOMAINS)


def test_a_target_no_rule_holds_for_keeps_its_default():
    rules = [ati.derives('x', to=1, when=ati.gt('seqlen_k', 128))]
    assert overrides_at(rules, {'seqlen_k': 128}, DOMAINS) == {}
    assert overrides_at(rules, {'seqlen_k': 129}, DOMAINS) == {'x': 1}
    assert builds(rules, DOMAINS) == [([{'seqlen_k': (LO, 128)}], {}),
                                      ([{'seqlen_k': (129, HI)}], {'x': 1})]


def test_a_missing_runtime_value_a_rule_reads_raises():
    with pytest.raises(ValueError, match="'seqlen_k'"):
        overrides_at([ati.derives('x', to=1, when=ati.gt('seqlen_k', 128))], {'seqlen_q': 1}, DOMAINS)


def test_tuple_targets_pin_one_value():
    rules = [ati.derives(('x', 'y'), to=8, when=None)]
    assert overrides_at(rules, {}, DOMAINS) == {'x': 8, 'y': 8}


def test_builds_refine_rules_on_different_targets():
    rules = [ati.derives('x', to=1, when=ati.gt('seqlen_k', 128)),
             ati.derives('y', to=1, when=ati.gt('seqlen_k', 192)),
             ati.derives('z', to=1, when=ati.le('seqlen_q', 16))]
    assert [over for _, over in builds(rules, DOMAINS)] == [
        {'z': 1}, {}, {'x': 1, 'z': 1}, {'x': 1}, {'x': 1, 'y': 1, 'z': 1}, {'x': 1, 'y': 1}]
    verify(rules, DOMAINS)


def test_equal_overrides_merge_into_one_build():
    rules = [ati.derives('x', to=1, when=ati.le('seqlen_q', 16)),
             ati.derives('x', to=1, when=ati.gt('seqlen_q', 16))]
    assert builds(rules, DOMAINS) == [([{}], {'x': 1})]

