# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 1: @ati.tensor_dtype as a named decorator + string dtype
refs on @ati.tensor / @ati.scalar (agent-plans/ati_aux-kernel-xref_rev0.md §4.2).

A dtype variable declared by name must build an identical BuiltKernel to the
object-threaded form, and an unknown string dtype (neither a same-kernel dtype-var
nor a literal ATI type) must raise AtiDescriptionError."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

import v3python.template_instantiation as ati
from v3python.template_instantiation.describe import describe, get_kernel_spec
from v3python.template_instantiation.builder import build_kernel, AtiDescriptionError

from fwd_kernel import attn_fwd

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']
BLOCK_DMODEL_VALUES = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]


def _common_specs():
    return [
        ati.scalar('BLOCK_DMODEL', options=BLOCK_DMODEL_VALUES),
        ati.scalar('PADDED_HEAD', options=[False, True]),
        ati.scalar('ENABLE_DROPOUT', options=[False, True]),
        ati.scalar('CAUSAL_TYPE', options=[0, 3]),
        ati.scalar('BIAS_TYPE', options=[0, 1]),
        ati.scalar('Max_seqlen_q', ati.choice_set('Seqlen', options=['i32'])),
        ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0)),
    ]


def _describe_object_form():
    # OLD form: dtype variable threaded by object reference.
    T_io = ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q')
    specs = [
        ati.tensor('Q', T_io, strides='stride_q?', contiguous=-1),
        ati.tensor('K', T_io, strides='stride_k?', contiguous=-1),
        ati.tensor('V', T_io, strides='stride_v?', contiguous=-1),
        ati.tensor('Out', T_io, strides='stride_o?', contiguous=-1),
        ati.tensor('B', T_io, strides='stride_b?'),
    ] + _common_specs()
    describe(attn_fwd, *specs, _validate=False)
    return build_kernel(get_kernel_spec(attn_fwd))


def _describe_named_form():
    # NEW form: dtype variable declared as a standalone record; tensors name it by
    # string. (Mode-B describe() with the tensor_dtype record in the spec list.)
    specs = [
        ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q'),
        ati.tensor('Q', 'T_io', strides='stride_q?', contiguous=-1),
        ati.tensor('K', 'T_io', strides='stride_k?', contiguous=-1),
        ati.tensor('V', 'T_io', strides='stride_v?', contiguous=-1),
        ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1),
        ati.tensor('B', 'T_io', strides='stride_b?'),
    ] + _common_specs()
    describe(attn_fwd, *specs, _validate=False)
    return build_kernel(get_kernel_spec(attn_fwd))


def _axis_fingerprint(bk):
    # Compare the salient axis structure: var, args, radix, anchor, kind, sig name.
    out = []
    for a in sorted(bk.axes, key=lambda a: a.anchor):
        out.append((a.var_name, tuple(a.arg_names), a.radix, a.anchor,
                    a.kind, a.signature_name))
    return out


def test_named_form_matches_object_form():
    obj = _describe_object_form()
    named = _describe_named_form()
    assert _axis_fingerprint(obj) == _axis_fingerprint(named)
    assert obj.arguments == named.arguments


def test_named_tio_groups_five_tensors():
    bk = _describe_named_form()
    tio = next(a for a in bk.axes if a.var_name == 'T_io')
    assert set(tio.arg_names) == {'Q', 'K', 'V', 'Out', 'B'}
    assert tio.signature_name == 'Q'
    assert tio.radix == 3


def test_literal_string_dtype_still_works():
    # A literal ATI type string in the dtype slot is NOT a dtype-var name and must
    # remain a single-choice anonymous axis.
    specs = [
        ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Q'),
        ati.tensor('Q', 'T_io', strides='stride_q?', contiguous=-1),
        ati.tensor('K', 'T_io', strides='stride_k?', contiguous=-1),
        ati.tensor('V', 'T_io', strides='stride_v?', contiguous=-1),
        ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1),
        ati.tensor('B', 'T_io', strides='stride_b?'),
        ati.tensor('L', '*fp32:16', rank=2),     # literal type string
    ] + _common_specs()
    describe(attn_fwd, *specs, _validate=False)
    bk = build_kernel(get_kernel_spec(attn_fwd))
    L = next(a for a in bk.axes if a.var_name == 'L')
    assert L.is_trivial                          # single literal choice
    assert L.choices[0].is_tensor


def test_unknown_string_dtype_raises():
    specs = [
        ati.tensor('Q', 'T_nope', strides='stride_q?', contiguous=-1),
    ]
    describe(attn_fwd, *specs, _validate=False)
    try:
        build_kernel(get_kernel_spec(attn_fwd))
    except AtiDescriptionError as e:
        assert 'T_nope' in str(e)
        return
    raise AssertionError('expected AtiDescriptionError for unknown dtype var name')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} named-dtype tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
