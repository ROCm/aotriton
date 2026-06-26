# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the choice-declaring decorators (executive plan Step 2.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.decorators import (
    ChoiceVar, TensorSpec, ScalarSpec,
)

# A stand-in for the introspected @triton.jit parameter list.
PARAMS = [
    'Q', 'K', 'V', 'B', 'Sm_scale', 'L', 'Out',
    'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
    'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
    'stride_bz', 'stride_bh', 'stride_bm', 'stride_bn',
    'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
    'Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k',
    'CAUSAL_TYPE', 'BIAS_TYPE', 'NUM_XCDS',
]


def test_type_var_and_binding_share_var():
    T = ati.type_var('T_io', dtype=['*fp16:16', '*bf16:16', '*fp32:16'])
    assert isinstance(T, ChoiceVar) and T.kind == 'type'
    q = ati.tensor('Q', T, strides='stride_q?', contiguous=-1)
    k = ati.tensor('K', T, strides='stride_k?', contiguous=-1)
    assert isinstance(q, TensorSpec)
    assert q.var_name == 'T_io' == k.var_name      # shared dimension
    assert not q.is_literal_dtype


def test_literal_dtype_is_anonymous_var():
    spec = ati.tensor('L', '*fp32:16', rank=2)
    assert spec.is_literal_dtype
    assert spec.var_name == 'L'                     # named after the argument
    assert spec.resolve_rank(PARAMS) == 2


def test_stride_glob_match_in_signature_order():
    spec = ati.tensor('Q', ati.type_var('T', dtype=['*fp16:16']),
                      strides='stride_q?')
    assert spec.match_strides(PARAMS) == \
        ['stride_qz', 'stride_qh', 'stride_qm', 'stride_qk']


def test_rank_inferred_from_stride_count():
    T = ati.type_var('T', dtype=['*fp16:16'])
    assert ati.tensor('Q', T, strides='stride_q?').resolve_rank(PARAMS) == 4
    assert ati.tensor('B', T, strides='stride_b?').resolve_rank(PARAMS) == 4


def test_explicit_rank_overrides_inference():
    T = ati.type_var('T', dtype=['*fp16:16'])
    spec = ati.tensor('Q', T, strides='stride_q?', rank=3)
    assert spec.resolve_rank(PARAMS) == 3


def test_contiguous_index_resolves_to_stride_name():
    T = ati.type_var('T', dtype=['*fp16:16'])
    spec = ati.tensor('Q', T, strides='stride_q?', contiguous=-1)
    assert spec.resolve_contiguous(PARAMS) == 'stride_qk'    # last matched stride
    spec0 = ati.tensor('Q', T, strides='stride_q?', contiguous=0)
    assert spec0.resolve_contiguous(PARAMS) == 'stride_qz'


def test_contiguous_explicit_name():
    T = ati.type_var('T', dtype=['*fp16:16'])
    spec = ati.tensor('Q', T, strides='stride_q?', contiguous='stride_qk')
    assert spec.resolve_contiguous(PARAMS) == 'stride_qk'


def test_contiguous_none():
    T = ati.type_var('T', dtype=['*fp16:16'])
    spec = ati.tensor('B', T, strides='stride_b?')
    assert spec.resolve_contiguous(PARAMS) is None


def test_scalar_plain_runtime():
    spec = ati.scalar('Sm_scale', 'fp32')
    assert isinstance(spec, ScalarSpec)
    assert spec.type_ == 'fp32'
    assert spec.options is None
    assert spec.var_name == 'Sm_scale'
    assert spec.has_explicit_type


def test_scalar_options_enumerated():
    spec = ati.scalar('CAUSAL_TYPE', options=[0, 3])
    assert spec.options == [0, 3]
    assert spec.type_ is None
    assert not spec.has_explicit_type        # type inferred from values/annotation


def test_scalar_options_numpy_array_preserved():
    spec = ati.scalar('NUM_XCDS', options=np.array([8], np.int8))
    assert isinstance(spec.options, np.ndarray)
    assert spec.options.dtype == np.int8


def test_scalar_shared_choice_var():
    S = ati.scalar_var('Seqlen', options=['i32'])
    assert isinstance(S, ChoiceVar) and S.kind == 'scalar'
    a = ati.scalar('Max_seqlen_q', S)
    b = ati.scalar('Max_seqlen_k', S)
    assert a.var_name == 'Seqlen' == b.var_name


def test_scalar_no_type_defers_to_annotation():
    spec = ati.scalar('Num_head_q')          # type read from annotation in Step 2.3
    assert not spec.has_explicit_type
    assert spec.var_name == 'Num_head_q'


def test_scalar_name_list_shares_one_axis():
    spec = ati.scalar(['Q_descale', 'K_descale', 'P_scale'], options=[0])
    assert spec.arg_names == ('Q_descale', 'K_descale', 'P_scale')
    assert spec.arg_name == 'Q_descale'              # representative
    assert spec.var_name == 'Q_descale'
    assert spec.options == [0]


def test_tensor_name_list_strideless():
    T = ati.type_var('Tp', dtype=['*u64'])
    spec = ati.tensor(['philox_seed_ptr', 'philox_offset1'], T, rank=0)
    assert spec.arg_names == ('philox_seed_ptr', 'philox_offset1')
    assert spec.var_name == 'Tp'
    assert spec.resolve_rank([]) == 0


def test_tensor_name_list_forbids_strides():
    T = ati.type_var('Tp', dtype=['*u64'])
    try:
        ati.tensor(['a', 'b'], T, strides='stride_a?')
    except AssertionError as e:
        assert 'name list' in str(e)
        return
    raise AssertionError('expected strides-forbidden assertion')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} decorator tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
