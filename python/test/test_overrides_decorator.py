# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the overrides decorator + predicate builders (Step 2.2).

Checks that the agent-plans/ati_rev1.md §3.3 surface lowers to the right Override
objects from the IR layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.ir import Override, Predicate, VarRef, TypedChoice


def test_predicate_builders_return_predicates():
    for build, op in [(ati.eq, '=='), (ati.ne, '!='), (ati.lt, '<'), (ati.gt, '>')]:
        p = build('BIAS_TYPE', 0)
        assert isinstance(p, Predicate)
        assert p.var_name == 'BIAS_TYPE' and p.op == op and p.operand == 0


def test_literal_zero_override():
    ov = ati.overrides('encoded_softmax', to=0,
                       when=ati.eq('RETURN_ENCODED_SOFTMAX', False))
    assert isinstance(ov, Override)
    assert ov.targets == ('encoded_softmax',)
    assert ov.value == 0                      # literal, not VarRef
    assert ov.predicate.var_name == 'RETURN_ENCODED_SOFTMAX'
    # materializes to constexpr 0 when it fires
    c = ov.materialize({'RETURN_ENCODED_SOFTMAX': TypedChoice.parse(False)})
    assert c.is_constexpr and c.triton_compile_signature == 0


def test_string_to_is_varref():
    ov = ati.overrides('Hdim_qk', to='BLOCK_DMODEL',
                       when=ati.eq('PADDED_HEAD', False))
    assert isinstance(ov.value, VarRef)
    assert ov.value.var_name == 'BLOCK_DMODEL'
    picked = {'PADDED_HEAD': TypedChoice.parse(False), 'BLOCK_DMODEL': TypedChoice.parse(64)}
    assert ov.materialize(picked) == picked['BLOCK_DMODEL']


def test_multi_target_list():
    ov = ati.overrides(['dropout_p', 'philox_seed_ptr', 'philox_offset1'],
                       to=0, when=ati.eq('ENABLE_DROPOUT', False))
    assert ov.targets == ('dropout_p', 'philox_seed_ptr', 'philox_offset1')


def test_ne_predicate_sliding_window():
    ov = ati.overrides(['Window_left', 'Window_right'], to=0,
                       when=ati.ne('CAUSAL_TYPE', 3))
    assert ov.predicate.op == '!='
    assert ov.predicate.holds({'CAUSAL_TYPE': TypedChoice.parse(0)})
    assert not ov.predicate.holds({'CAUSAL_TYPE': TypedChoice.parse(3)})


def test_tensor_zero_is_literal_not_varref():
    # `to=0` on a tensor target (the former CDETensor case) is a literal zero,
    # distinguished from `to='SomeVar'` purely by type.
    ov = ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0))
    assert ov.value == 0 and not isinstance(ov.value, VarRef)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} overrides-decorator tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
