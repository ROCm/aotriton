# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for Axis + mixed-radix godel numbering (executive plan Step 1.2).

Reproduces the attn_fwd worked example in agent-plans/ati+newbinds_rev1.md §7."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3python.template_instantiation.ir import Choice, Axis, assign_godel, godel_of


def _axis(var_name, raw_choices, anchor):
    return Axis(var_name,
                arg_names=(var_name,),
                choices=[Choice.parse(c) for c in raw_choices],
                anchor=anchor)


def _attn_fwd_axes():
    # anchor = signature position; ascending anchor is canonical order.
    return [
        _axis('T_io', ['*fp16:16', '*bf16:16', '*fp32:16'], anchor=0),
        _axis('BLOCK_DMODEL', list(range(12)), anchor=10),   # 12 distinct values
        _axis('PADDED_HEAD', [False, True], anchor=20),
        _axis('ENABLE_DROPOUT', [False, True], anchor=30),
        _axis('CAUSAL_TYPE', [0, 3], anchor=40),
        _axis('BIAS_TYPE', [0, 1], anchor=50),
    ]


def test_radix_and_triviality():
    axes = _attn_fwd_axes()
    assert [a.radix for a in axes] == [3, 12, 2, 2, 2, 2]
    assert not any(a.is_trivial for a in axes)
    triv = _axis('Sm_scale', ['fp32'], anchor=5)
    assert triv.is_trivial


def test_strides_and_total():
    axes = _attn_fwd_axes()
    total = assign_godel(axes)
    assert total == 576                       # 3*12*2*2*2*2
    assert [a.godel_stride for a in axes] == [192, 16, 8, 4, 2, 1]


def test_sample_godel():
    axes = _attn_fwd_axes()
    assign_godel(axes)
    # T_io=idx1, BLOCK_DMODEL=idx3, PADDED_HEAD=0, ENABLE_DROPOUT=1,
    # CAUSAL_TYPE=1, BIAS_TYPE=0
    selection = [1, 3, 0, 1, 1, 0]
    expected = 1 * 192 + 3 * 16 + 0 * 8 + 1 * 4 + 1 * 2 + 0 * 1
    assert expected == 246
    assert godel_of(axes, selection) == 246


def test_godel_is_bijection():
    import itertools
    axes = _attn_fwd_axes()
    total = assign_godel(axes)
    seen = set()
    for sel in itertools.product(*[range(a.radix) for a in axes]):
        seen.add(godel_of(axes, sel))
    assert seen == set(range(total))          # dense, no collisions


def test_choice_for_arg_rank_specialization():
    axis = Axis('T_io', arg_names=('Q', 'L'),
                choices=[Choice.parse('*fp16:16')],
                anchor=0, ranks={'Q': 4, 'L': 2})
    assert axis.choice_for_arg(0, 'Q').itype == 'const TensorView<4>*'
    assert axis.choice_for_arg(0, 'L').itype == 'const TensorView<2>*'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} Axis tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
