# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for introspection + describe() + stacked-@ sugar (Step 2.3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.introspect import kernel_params, kernel_param_names
from aotriton.template_instantiation.describe import get_kernel_spec


# A plain function stands in for a @triton.jit kernel; introspection falls back to
# inspect.signature when triton is absent, reading names + constexpr-from-annotation.
def _fake_kernel(Q, K, Sm_scale, stride_qz, stride_qh, stride_qm, stride_qk,
                 stride_kz, stride_kh, stride_kn, stride_kk,
                 CAUSAL_TYPE: 'constexpr', BIAS_TYPE: 'constexpr'):
    pass


def _specs():
    T = ati.tensor_dtype('T_io', dtype=['*fp16:16', '*bf16:16'])
    return [
        ati.tensor('Q', T, strides='stride_q?', contiguous=-1),
        ati.tensor('K', T, strides='stride_k?', contiguous=-1),
        ati.scalar('Sm_scale', 'fp32'),
        ati.scalar('CAUSAL_TYPE', options=[0, 3]),
        ati.scalar('BIAS_TYPE', options=[0, 1]),
    ]


def test_introspection_names_and_constexpr():
    params = kernel_params(_fake_kernel)
    assert kernel_param_names(_fake_kernel)[:3] == ['Q', 'K', 'Sm_scale']
    by_name = {p.name: p for p in params}
    assert by_name['CAUSAL_TYPE'].is_constexpr
    assert not by_name['Sm_scale'].is_constexpr


def test_describe_mode_b():
    def k(Q, K, Sm_scale, stride_qz, stride_qh, stride_qm, stride_qk,
          stride_kz, stride_kh, stride_kn, stride_kk,
          CAUSAL_TYPE: 'constexpr', BIAS_TYPE: 'constexpr'):
        pass
    ati.describe(k, *_specs())
    spec = get_kernel_spec(k)
    assert spec is not None
    assert len(spec.tensors) == 2
    assert len(spec.scalars) == 3
    assert spec.param_names[0] == 'Q'


def test_stacked_at_sugar_matches_mode_b():
    # Mode A: specs as decorators, terminated by @ati.kernel (top of stack).
    T = ati.tensor_dtype('T_io', dtype=['*fp16:16', '*bf16:16'])

    @ati.kernel
    @ati.tensor('Q', T, strides='stride_q?', contiguous=-1)
    @ati.tensor('K', T, strides='stride_k?', contiguous=-1)
    @ati.scalar('Sm_scale', 'fp32')
    @ati.scalar('CAUSAL_TYPE', options=[0, 3])
    @ati.scalar('BIAS_TYPE', options=[0, 1])
    def k(Q, K, Sm_scale, stride_qz, stride_qh, stride_qm, stride_qk,
          stride_kz, stride_kh, stride_kn, stride_kk,
          CAUSAL_TYPE: 'constexpr', BIAS_TYPE: 'constexpr'):
        pass

    spec = get_kernel_spec(k)
    assert spec is not None
    # Same tensors/scalars as Mode B, in source order.
    assert [t.arg_name for t in spec.tensors] == ['Q', 'K']
    assert [s.arg_name for s in spec.scalars] == \
        ['Sm_scale', 'CAUSAL_TYPE', 'BIAS_TYPE']


def test_overrides_in_stack():
    T = ati.tensor_dtype('T_io', dtype=['*fp16:16'])

    @ati.kernel
    @ati.tensor('Q', T, strides='stride_q?')
    @ati.tensor('B', T, strides='stride_b?')
    @ati.scalar('BIAS_TYPE', options=[0, 1])
    @ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0))
    def k(Q, B, stride_qz, stride_qh, stride_qm, stride_qk,
          stride_bz, stride_bh, stride_bm, stride_bn,
          BIAS_TYPE: 'constexpr'):
        pass

    spec = get_kernel_spec(k)
    assert len(spec.overrides) == 1
    assert spec.overrides[0].targets == ('B',)


def test_completeness_flags_orphan():
    def k(Q, K, stride_qz, stride_qh, stride_qm, stride_qk):
        pass
    T = ati.tensor_dtype('T', dtype=['*fp16:16'])
    try:
        ati.describe(k, ati.tensor('Q', T, strides='stride_q?'))   # K unclaimed
    except AssertionError as e:
        assert "'K'" in str(e) and 'not claimed' in str(e)
        return
    raise AssertionError('expected orphan completeness error')


def test_completeness_flags_double_claim():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk):
        pass
    T = ati.tensor_dtype('T', dtype=['*fp16:16'])
    try:
        ati.describe(k,
                     ati.tensor('Q', T, strides='stride_q?'),
                     ati.scalar('Q', 'i32'))      # Q claimed twice
    except AssertionError as e:
        assert "'Q'" in str(e) and 'multiple' in str(e)
        return
    raise AssertionError('expected double-claim completeness error')


def test_get_kernel_spec_rejects_unterminated_stack():
    T = ati.tensor_dtype('T', dtype=['*fp16:16'])

    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk):
        pass
    # accumulate without @ati.kernel terminator
    ati.tensor('Q', T, strides='stride_q?')(k)
    try:
        get_kernel_spec(k)
    except AssertionError as e:
        assert 'un-finalized' in str(e)
        return
    raise AssertionError('expected un-finalized-stack error')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} describe/introspect tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
