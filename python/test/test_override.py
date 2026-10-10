# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for Predicate / VarRef / Override (executive plan Step 1.3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.ir import (
    TypedChoice, Predicate, VarRef, Override, eq, ne, gt,
)


def _picked(**kw):
    return {k: TypedChoice.parse(v) for k, v in kw.items()}


def test_predicate_eq_int():
    p = eq('BIAS_TYPE', 0)
    assert p.holds(_picked(BIAS_TYPE=0))
    assert not p.holds(_picked(BIAS_TYPE=1))


def test_predicate_eq_bool():
    p = eq('ENABLE_DROPOUT', False)
    assert p.holds(_picked(ENABLE_DROPOUT=False))
    assert not p.holds(_picked(ENABLE_DROPOUT=True))


def test_predicate_ne_and_gt():
    assert ne('CAUSAL_TYPE', 3).holds(_picked(CAUSAL_TYPE=0))
    assert not ne('CAUSAL_TYPE', 3).holds(_picked(CAUSAL_TYPE=3))
    assert gt('num_seqlens', 0).holds(_picked(num_seqlens=1))
    assert not gt('num_seqlens', 0).holds(_picked(num_seqlens=0))


def test_override_literal_materialize():
    ov = Override('B', eq('BIAS_TYPE', 0), value=0)
    assert ov.targets == ('B',)
    c = ov.materialize(_picked(BIAS_TYPE=0))
    assert c.is_constexpr and c.triton_compile_signature == 0


def test_override_varref_copies_choice():
    # Hdim_qk <- BLOCK_DMODEL when PADDED_HEAD is False
    ov = Override('Hdim_qk', eq('PADDED_HEAD', False), value=VarRef('BLOCK_DMODEL'))
    picked = _picked(PADDED_HEAD=False, BLOCK_DMODEL=64)
    assert ov.predicate.holds(picked)
    got = ov.materialize(picked)
    assert got == picked['BLOCK_DMODEL']


def test_override_multi_target():
    ov = Override(['dropout_p', 'philox_seed_ptr'], eq('ENABLE_DROPOUT', False),
                  value=0)
    assert ov.targets == ('dropout_p', 'philox_seed_ptr')


def test_validate_free_axis_only():
    free = {'BIAS_TYPE', 'BLOCK_DMODEL', 'PADDED_HEAD'}
    Override('B', eq('BIAS_TYPE', 0), value=0).validate(free)          # ok
    Override('Hdim_qk', eq('PADDED_HEAD', False),
             value=VarRef('BLOCK_DMODEL')).validate(free)              # ok
    # predicate over a non-free variable -> reject
    try:
        Override('x', eq('Hdim_qk', 0), value=0).validate(free)
    except AssertionError:
        pass
    else:
        raise AssertionError('expected predicate-scope assertion')
    # VarRef to a non-free variable -> reject
    try:
        Override('x', eq('BIAS_TYPE', 0), value=VarRef('Hdim_qk')).validate(free)
    except AssertionError:
        pass
    else:
        raise AssertionError('expected VarRef-scope assertion')


def test_bad_op_rejected():
    try:
        Predicate('x', '~=', 0)
    except AssertionError:
        return
    raise AssertionError('expected bad-op assertion')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} Override tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
