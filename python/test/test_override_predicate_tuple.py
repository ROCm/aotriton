# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`ati.derives` with a tuple of predicates: it fires when every one holds."""

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.ir import Override


class _Choice:
    def __init__(self, value):
        self.triton_compile_signature = value


class _Functional:
    choice = {'A': _Choice(1), 'B': _Choice(5)}


def test_a_predicate_tuple_fires_when_all_hold():
    assert Override('t', (ati.eq('A', 1), ati.gt('B', 4)), 0).fires(_Functional)
    assert not Override('t', (ati.eq('A', 1), ati.gt('B', 5)), 0).fires(_Functional)
    assert ati.derives('t', to=0, when=(ati.eq('A', 1), ati.ge('B', 5))).fires(_Functional)
    assert not ati.derives('t', to=0, when=(ati.le('A', 0),)).fires(_Functional)


def test_derives_without_when_always_fires():
    assert ati.derives('t', to=0).fires(_Functional)


def test_a_predicate_tuple_validates_every_predicate():
    Override('t', (ati.eq('A', 1), ati.gt('B', 4)), 0).validate({'A', 'B'})
    try:
        Override('t', (ati.eq('A', 1), ati.gt('C', 4)), 0).validate({'A', 'B'})
    except AssertionError as e:
        assert "'C'" in str(e)
    else:
        raise AssertionError('a predicate on a non-axis passed validation')
