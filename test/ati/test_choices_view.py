# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for Functional.choices accessor view (executive plan Step 1.5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.ir import (
    Choice, Axis, Override, eq, Interface,
)


class _IRStub(Interface):
    FAMILY = 'test'
    NAME = 'stub'
    def __init__(self, axes, overrides):
        self._axes = axes
        self._overrides = overrides
    def _axes_overrides(self):
        return self._axes, self._overrides


def enumerate_functionals(axes, overrides, target_arch):
    return _IRStub(axes, overrides).gen_functionals(target_arch)


def _axis(var_name, arg_names, raw_choices, anchor, ranks=None):
    return Axis(var_name, arg_names,
                [Choice.parse(c) for c in raw_choices], anchor, ranks=ranks)


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
    assert isinstance(tc, Choice)
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


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} ChoiceView tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
