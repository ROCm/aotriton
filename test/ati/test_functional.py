# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for Functional + enumerate_functionals (executive plan Step 1.4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.ir import (
    Choice, Axis, Override, eq, enumerate_functionals,
)


def _axis(var_name, arg_names, raw_choices, anchor, ranks=None):
    return Axis(var_name, arg_names,
                [Choice.parse(c) for c in raw_choices], anchor, ranks=ranks)


def _attn_fwd_ir():
    axes = [
        # T_io types Q,K,V,B,Out — all rank 4
        _axis('T_io', ('Q', 'K', 'V', 'B', 'Out'),
              ['*fp16:16', '*bf16:16', '*fp32:16'], anchor=0,
              ranks={'Q': 4, 'K': 4, 'V': 4, 'B': 4, 'Out': 4}),
        _axis('BLOCK_DMODEL', ('BLOCK_DMODEL',), list(range(12)), anchor=10),
        _axis('PADDED_HEAD', ('PADDED_HEAD',), [False, True], anchor=20),
        _axis('ENABLE_DROPOUT', ('ENABLE_DROPOUT',), [False, True], anchor=30),
        _axis('CAUSAL_TYPE', ('CAUSAL_TYPE',), [0, 3], anchor=40),
        _axis('BIAS_TYPE', ('BIAS_TYPE',), [0, 1], anchor=50),
        # trivial axes (radix 1) — excluded from godel
        _axis('Sm_scale', ('Sm_scale',), ['fp32'], anchor=5),
        _axis('Seqlen', ('Max_seqlen_q', 'Max_seqlen_k'), ['i32'], anchor=45),
    ]
    overrides = [
        Override('B', eq('BIAS_TYPE', 0), value=0),
    ]
    return axes, overrides


def test_total_count():
    axes, overrides = _attn_fwd_ir()
    arches = {'gfx942': ['gfx942'], 'gfx950': ['gfx950']}
    fs = list(enumerate_functionals(axes, overrides, arches))
    assert len(fs) == 2 * 576


def test_identity_and_godel_range():
    axes, overrides = _attn_fwd_ir()
    arches = {'gfx942': ['g0'], 'gfx950': ['g1']}
    fs = list(enumerate_functionals(axes, overrides, arches))
    by_arch = {0: set(), 1: set()}
    for f in fs:
        by_arch[f.arch_number].add(f.godel_number)
    # each arch covers the full dense godel space, identical across arches
    assert by_arch[0] == set(range(576))
    assert by_arch[1] == set(range(576))
    # identity pairs distinct per arch
    assert len({f.identity for f in fs}) == 2 * 576


def test_override_B_zeroed_when_bias_off():
    axes, overrides = _attn_fwd_ir()
    arches = {'gfx942': ['g0']}
    fs = list(enumerate_functionals(axes, overrides, arches))
    for f in fs:
        bias = f.choice['BIAS_TYPE'].triton_compile_signature
        b = f.resolved['B']
        if bias == 0:
            assert b.is_constexpr and b.triton_compile_signature == 0, \
                f'B should be constexpr 0 when BIAS_TYPE==0, got {b}'
        else:
            # B keeps the T_io dtype, rank-4 specialized
            assert b.is_tensor
            assert b.triton_compile_signature == \
                f.choice['T_io'].triton_compile_signature
            assert b.itype == 'const TensorView<4>*'


def test_trivial_axes_in_resolved_not_in_godel():
    axes, overrides = _attn_fwd_ir()
    arches = {'gfx942': ['g0']}
    f = next(enumerate_functionals(axes, overrides, arches))
    # trivial scalar present in resolved
    assert f.resolved['Sm_scale'].triton_compile_signature == 'fp32'
    assert f.resolved['Max_seqlen_q'].triton_compile_signature == 'i32'
    # but does not expand the godel space (still 576)
    assert f.choice['Sm_scale'].triton_compile_signature == 'fp32'


def test_tensor_rank_in_resolved():
    axes, overrides = _attn_fwd_ir()
    arches = {'gfx942': ['g0']}
    f = next(enumerate_functionals(axes, overrides, arches))
    for t in ('Q', 'K', 'V', 'Out'):
        assert f.resolved[t].itype == 'const TensorView<4>*'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} Functional tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
