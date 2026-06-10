# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Triton-free tests for scalar-type resolution in the builder (Step 2.4).

Covers the resolution chain that needs no triton type object: explicit type, and
string-annotation passthrough (validated against the ATI type vocabulary). The
triton type-OBJECT identity path (tl.float32 -> 'fp32' etc.) is covered against
the real kernel in test_builder.py. Also checks the kernel+parameter-named
diagnostics the front-end emits, in the spirit of the Triton compiler frontend it
partially reimplements."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v3python.template_instantiation as ati
from v3python.template_instantiation.describe import describe, get_kernel_spec
from v3python.template_instantiation.builder import build_kernel, AtiDescriptionError


def _build(kernel, *specs):
    describe(kernel, *specs, _validate=False)
    return build_kernel(get_kernel_spec(kernel))


def test_explicit_type_wins():
    def k(Sm_scale): pass
    bk = _build(k, ati.scalar('Sm_scale', 'fp32'))
    ax = next(a for a in bk.axes if a.var_name == 'Sm_scale')
    assert ax.choices[0].triton_compile_signature == 'fp32'


def test_string_annotation_passthrough():
    # A string annotation is validated then handed to Choice.parse, which parses
    # the ATI type vocabulary. This is the form real kernels use for pointer
    # scalars, e.g. philox: '*u64'.
    def k(Num_head_q: 'i32', philox: 'u64', seed_ptr: '*u64'): pass
    bk = _build(k, ati.scalar('Num_head_q'), ati.scalar('philox'),
                ati.scalar('seed_ptr'))
    by = {a.var_name: a.choices[0].triton_compile_signature for a in bk.axes}
    assert by['Num_head_q'] == 'i32'
    assert by['philox'] == 'u64'
    assert by['seed_ptr'] == '*u64'


def test_unrecognized_string_annotation_names_kernel_and_param():
    def attn_fwd(weird: 'not_a_type'): pass
    try:
        _build(attn_fwd, ati.scalar('weird'))
    except AtiDescriptionError as e:
        msg = str(e)
        assert "'attn_fwd'" in msg          # names the kernel
        assert "'weird'" in msg             # names the parameter
        assert 'not_a_type' in msg          # shows the bad annotation
        assert 'ati.scalar' in msg          # tells how to fix
        return
    raise AssertionError('expected AtiDescriptionError')


def test_missing_type_and_annotation_names_kernel_and_param():
    def attn_fwd(Num_head_q): pass           # no annotation at all
    try:
        _build(attn_fwd, ati.scalar('Num_head_q'))
    except AtiDescriptionError as e:
        msg = str(e)
        assert "'attn_fwd'" in msg
        assert "'Num_head_q'" in msg
        assert 'no type' in msg or 'no annotation' in msg
        return
    raise AssertionError('expected AtiDescriptionError')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} scalar-type tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
