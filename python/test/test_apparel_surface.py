# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 3: apparel_of() is applied on the OUTWARD codegen surface
(struct fields, launch vector, feature-table getter, persisted signature) while the
IR stays keyed on REAL argument names (agent-plans/ati_aux-kernel-xref_rev0.md §4.3).

Uses a debug-like kernel with R wired to encoded_softmax."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import InterfaceRegistry, _testonly_build_kernel_description

from dropout_rng import debug_simulate_encoded_softmax

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


def _describe_debug(registry):
    specs = [
        ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='R'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar('dropout_p', 'fp32'),
        ati.scalar(['Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k'], 'i32'),
        ati.tensor(['philox_seed_ptr', 'philox_offset1'], '*u64', rank=0),
        ati.scalar('philox_offset2', 'u64'),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),   # placeholder constexpr
    ]
    describe(debug_simulate_encoded_softmax, *specs, _validate=False)
    return _testonly_build_kernel_description(debug_simulate_encoded_softmax, family='flash',
                                    source_path='tritonsrc/flash.py',
                                    registry=registry)


def test_struct_field_is_apparel():
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    names = [cf.aname for cf in kdesc.func_cfields]
    assert 'encoded_softmax' in names      # apparel surfaced
    assert 'R' not in names                # real name not in the struct


def test_launch_vector_uses_apparel():
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    largs = list(kdesc.iter_launch_arguments())
    by = {la.aname: la for la in largs}
    # The tensor pointer surfaces as apparel; expr addresses the apparel field.
    assert 'encoded_softmax' in by
    assert by['encoded_softmax'].expr == 'params.encoded_softmax->kparam_data_ptr()'
    assert 'R' not in by
    # A stride keeps its real name but references the apparel tensor in its expr.
    assert by['stride_rz'].expr == 'params.encoded_softmax->kparam_stride(0)'


def test_feature_getter_name_is_apparel():
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    reprs = [tp.repr_name for tp in kdesc.list_functional_params()]
    assert 'encoded_softmax' in reprs      # get_encoded_softmax_choices
    assert 'R' not in reprs


def test_persisted_signature_uses_signature_name_label():
    # The persisted key (compact_choices / unified_signature) is the axis's
    # signature_name LABEL, independent of wiring/apparel. Here T_io's
    # signature_name is 'R', so the persisted key is 'R' (not the apparel
    # 'encoded_softmax', and not derived from the real repr_arg either — it is the
    # free label). The VALUE is looked up by the real repr_arg.
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    f = next(kdesc.gen_functionals({'gfx942': ['gfx942']}))
    assert 'R' in f.compact_choices
    assert 'encoded_softmax' not in f.compact_choices
    assert f.unified_signature.startswith('R=')


def test_ir_stays_keyed_on_real_names():
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    f = next(kdesc.gen_functionals({'gfx942': ['gfx942']}))
    # The resolved IR table is keyed on the REAL argument name.
    assert 'R' in f.resolved
    assert 'encoded_softmax' not in f.resolved
    # pp_arg_doc accepts the apparel name (as emitted by iter_launch_arguments) and
    # maps it back to the real key internally.
    assert kdesc.real_of('encoded_softmax') == 'R'
    is_constexpr, _ = f.pp_arg_doc('encoded_softmax')
    assert is_constexpr is False            # R is a runtime tensor pointer


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} apparel-surface tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
