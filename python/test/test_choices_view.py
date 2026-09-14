# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ChoiceView interface and its two backings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.ir import (
    TypedChoice, Axis, Override, eq, Interface,
    ChoiceView, ChoiceVarAbsent, FunctionalChoiceView,
)
# MappingChoiceView lives with its only user, the build-time driver.
from aotriton.flyc_compile import MappingChoiceView


class _IRStub(Interface):
    FAMILY = 'test'
    NAME = 'stub'
    def __init__(self, axes, overrides):
        self._axes = axes
        self._overrides = overrides
    def _axes_overrides(self):
        return self._axes, self._overrides
    # Interface abstract contract (no functional struct in this bare IR stub).
    @property
    def func_cfields(self):
        return []
    def list_functional_params(self):
        return []


def enumerate_functionals(axes, overrides, target_arch):
    return _IRStub(axes, overrides).gen_functionals(target_arch)


def _axis(var_name, arg_names, raw_choices, anchor, ranks=None):
    return Axis(var_name, arg_names,
                [TypedChoice.parse(c) for c in raw_choices], anchor, ranks=ranks)


def _functional(bias_type):
    axes = [
        _axis('T_io', ('Q', 'K', 'V', 'B', 'Out'),
              ['*fp16:16', '*bf16:16', '*fp32:16'], anchor=0,
              ranks={'Q': 4, 'K': 4, 'V': 4, 'B': 4, 'Out': 4}),
        _axis('CAUSAL_TYPE', ('CAUSAL_TYPE',), [0, 3], anchor=40),
        _axis('BIAS_TYPE', ('BIAS_TYPE',), [0, 1], anchor=50),
        _axis('Sm_scale', ('Sm_scale',), ['fp32'], anchor=5),
    ]
    overrides = [Override('B', eq('BIAS_TYPE', 0), value=0)]
    arches = {'gfx942': ['g0']}
    for f in enumerate_functionals(axes, overrides, arches):
        if f.choice['BIAS_TYPE'].triton_compile_signature == bias_type:
            return f
    raise AssertionError('no matching functional')


def test_attr_access_by_var_name():
    f = _functional(bias_type=1)
    assert f.choices.T_io == '*fp16:16'
    assert f.choices.CAUSAL_TYPE == 0
    assert f.choices.BIAS_TYPE == 1
    assert f.choices.Sm_scale == 'fp32'


def test_unknown_var_raises_listing():
    f = _functional(bias_type=1)
    try:
        _ = f.choices.NoSuchVar
    except AttributeError as e:
        assert 'NoSuchVar' in str(e)
        assert 'T_io' in str(e)        # lists valid vars
        return
    raise AssertionError('expected AttributeError')


def test_tc_returns_raw_choice():
    f = _functional(bias_type=1)
    tc = f.choices.tc('T_io')
    assert isinstance(tc, TypedChoice)
    assert tc.triton_compile_signature == '*fp16:16'


def test_arg_reads_resolved():
    # B keeps the dtype when bias on; becomes constexpr 0 when off
    on = _functional(bias_type=1)
    assert on.choices.arg('B') == '*fp16:16'
    off = _functional(bias_type=0)
    assert off.choices.arg('B') == 0
    assert off.choices.arg_tc('B').is_constexpr


def test_view_is_cached():
    f = _functional(bias_type=1)
    assert f.choices is f.choices


def test_bare_choiceview_uninstantiable():
    # ChoiceView is an ABC (ir/choices.py): it declares the interface but has
    # no backing of its own, so instantiating it directly must fail, naming
    # every unimplemented abstract method.
    try:
        ChoiceView()
    except TypeError as e:
        msg = str(e)
        for name in ('arg', '__getattr__'):
            assert name in msg, f'{name!r} missing from TypeError message: {msg!r}'
        return
    raise AssertionError('expected TypeError instantiating ChoiceView')


def test_tc_and_arg_tc_are_not_on_the_interface():
    # tc/arg_tc hand back a raw TypedChoice, which only a Functional has. They
    # are deliberately NOT part of the ABC: requiring them would force the
    # mapping-backed view to declare two methods whose only possible body is a
    # raise -- an interface that advertises an operation and then denies it. A
    # caller needing a TypedChoice must hold a FunctionalChoiceView, and finds
    # that out from the type rather than at the call.
    assert ChoiceView.__abstractmethods__ == frozenset({'arg', '__getattr__'})
    assert not hasattr(ChoiceView, 'tc')
    assert not hasattr(ChoiceView, 'arg_tc')
    assert callable(FunctionalChoiceView.tc)
    assert callable(FunctionalChoiceView.arg_tc)


def test_both_backings_implement_the_interface():
    # The claim this file's docstring makes -- "one interface, TWO backings" --
    # asserted rather than assumed, for each backing. `ChoiceView` is an ABC
    # with two abstract methods, so this also pins that neither backing has
    # stopped implementing one of them: an incomplete subclass is not
    # instantiable at all, and a class that quietly stopped inheriting from
    # ChoiceView would still work everywhere it is used today (both call sites
    # are duck-typed) while silently ending the shared contract.
    f = _functional(bias_type=1)
    assert isinstance(f.choices, ChoiceView)
    assert isinstance(MappingChoiceView({'BLOCK_DMODEL': 16}), ChoiceView)


def test_mapping_getattr_and_arg_read_the_same_dict():
    view = MappingChoiceView({'BLOCK_DMODEL': 16, 'Q': '*fp16:16'})
    assert view.BLOCK_DMODEL == 16
    assert view.arg('BLOCK_DMODEL') == 16
    assert view.arg('Q') == '*fp16:16'


def test_mapping_unknown_key_raises():
    view = MappingChoiceView({'BLOCK_DMODEL': 16})
    try:
        _ = view.NoSuchVar
    except ChoiceVarAbsent as e:
        assert isinstance(e, AttributeError)   # duck-typing contract (choices.py)
        assert 'NoSuchVar' in str(e)
        assert 'BLOCK_DMODEL' in str(e)
    else:
        raise AssertionError('expected ChoiceVarAbsent')
    try:
        view.arg('NoSuchVar')
    except KeyError as e:
        assert 'NoSuchVar' in str(e)
        return
    raise AssertionError('expected KeyError from MappingChoiceView.arg')


def test_both_backings_agree_on_shared_keys():
    # FunctionalChoiceView (real Functional) and MappingChoiceView (parsed
    # dict) must answer identically for a key/var both can honestly hold --
    # the whole point of ChoiceView being one declared interface with two
    # backings (ir/choices.py) rather than a dict on one side and an object
    # on the other.
    f = _functional(bias_type=1)
    mapping = MappingChoiceView({
        'T_io': f.choices.T_io,
        'CAUSAL_TYPE': f.choices.CAUSAL_TYPE,
        'BIAS_TYPE': f.choices.BIAS_TYPE,
    })
    # Attribute access is keyed by var_name on both backings.
    for var in ('T_io', 'CAUSAL_TYPE', 'BIAS_TYPE'):
        assert getattr(f.choices, var) == getattr(mapping, var)
    # .arg(aname) is keyed by real argument name; T_io's var_name is not one
    # of its own argument names (its axis spans Q/K/V/B/Out), but a
    # single-argument axis like CAUSAL_TYPE/BIAS_TYPE has var_name == its
    # only argument name, so both backings must agree there too.
    for var in ('CAUSAL_TYPE', 'BIAS_TYPE'):
        assert f.choices.arg(var) == mapping.arg(var)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} ChoiceView tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
