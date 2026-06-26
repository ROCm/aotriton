# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Triton-free tests for type resolution from placeholder-def annotations.

String annotations on the @ati.source placeholder def are dispatched by type:
  - pointer types ('*...' / 'LazyTensor:...') → rank-0 TensorSpec (no strides)
  - all other types ('fp32', 'i32', ...) → ScalarSpec
The Triton source's own annotations are never read.
Also checks the kernel+parameter-named diagnostics the front-end emits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.decorators import KernelStub
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.builder import build_kernel, DescriptionError


def _stub(name, params, annotations=None):
    """A KernelStub like @ati.source produces: parameter names + the placeholder def's
    string annotations, with no triton import."""
    return KernelStub(name, params, source_path=None, annotations=annotations)


def _build(kernel, *specs):
    describe(kernel, *specs, _validate=False)
    return build_kernel(get_kernel_spec(kernel))


def test_explicit_type_wins():
    k = _stub('k', ['Sm_scale'])
    bk = _build(k, ati.scalar('Sm_scale', 'fp32'))
    ax = next(a for a in bk.axes if a.var_name == 'Sm_scale')
    assert ax.choices[0].triton_compile_signature == 'fp32'


def test_placeholder_annotation_scalar():
    # Non-pointer annotations -> ScalarSpec.
    k = _stub('k', ['Num_head_q', 'philox'],
              annotations={'Num_head_q': 'i32', 'philox': 'u64'})
    bk = _build(k)
    by = {a.var_name: a for a in bk.axes}
    assert by['Num_head_q'].kind == 'scalar'
    assert by['Num_head_q'].choices[0].triton_compile_signature == 'i32'
    assert by['philox'].kind == 'scalar'
    assert by['philox'].choices[0].triton_compile_signature == 'u64'


def test_placeholder_annotation_tensor_pointer():
    # Pointer type annotations ('*...') -> rank-0 TensorSpec (strideless pointer).
    k = _stub('k', ['seed_ptr', 'lse'],
              annotations={'seed_ptr': '*u64', 'lse': '*fp32:16'})
    bk = _build(k)
    by = {a.var_name: a for a in bk.axes}
    assert by['seed_ptr'].kind == 'tensor'
    assert by['seed_ptr'].choices[0].triton_compile_signature == '*u64'
    assert by['lse'].kind == 'tensor'
    assert by['lse'].choices[0].triton_compile_signature == '*fp32:16'


def test_placeholder_annotation_tensor_with_rank_suffix():
    # '*fp32:16[2]' syntax: tensor pointer with explicit rank from the type string.
    k = _stub('k', ['lse', 'ptr0'],
              annotations={'lse': '*fp32:16[2]', 'ptr0': '*u64[0]'})
    bk = _build(k)
    by = {a.var_name: a for a in bk.axes}
    # lse: rank-2 tensor; triton_compile_signature strips the [N] suffix
    assert by['lse'].kind == 'tensor'
    assert by['lse'].choices[0].triton_compile_signature == '*fp32:16'
    assert by['lse'].choices[0].itype == 'const TensorView<2>*'
    # ptr0: explicit rank 0 (same as default, but stated explicitly)
    assert by['ptr0'].kind == 'tensor'
    assert by['ptr0'].choices[0].itype == 'const TensorView<0>*'


def test_annotation_conflict_with_explicit_spec():
    # A parameter annotated on the def AND claimed by an explicit @ati.scalar is an
    # error: declare it only once.
    k = _stub('k', ['Sm_scale'], annotations={'Sm_scale': 'fp32'})
    try:
        _build(k, ati.scalar('Sm_scale', 'fp32'))
    except DescriptionError as e:
        msg = str(e)
        assert "'k'" in msg
        assert "'Sm_scale'" in msg
        assert 'once' in msg
        return
    raise AssertionError('expected DescriptionError for annotation/spec conflict')


def test_unrecognized_annotation_names_kernel_and_param():
    k = _stub('attn_fwd', ['weird'], annotations={'weird': 'not_a_type'})
    try:
        _build(k)
    except DescriptionError as e:
        msg = str(e)
        assert "'attn_fwd'" in msg          # names the kernel
        assert "'weird'" in msg             # names the parameter
        assert 'not_a_type' in msg          # shows the bad type string
        return
    raise AssertionError('expected DescriptionError')


def test_missing_type_names_kernel_and_param():
    # ati.scalar with no type and no annotation -> error.
    k = _stub('attn_fwd', ['Num_head_q'])
    try:
        _build(k, ati.scalar('Num_head_q'))
    except DescriptionError as e:
        msg = str(e)
        assert "'attn_fwd'" in msg
        assert "'Num_head_q'" in msg
        assert 'no type' in msg
        return
    raise AssertionError('expected DescriptionError')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} scalar-type tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
