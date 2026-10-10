# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 2: wires_to= records the real->apparel argument wiring on
the kdesc (agent-plans/ati_aux-kernel-xref_rev0.md §4.3).

This step only RECORDS the wiring (apparel_of); applying it to the generated C++
is Step 3. So here we assert the kdesc reports the mapping for a wired argument and
identity for everything else. Uses a fake citing kernel (fakekernels.debug_stub),
independent of the real flash sources."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe
from registry import InterfaceRegistry, _testonly_build_kernel_description
from fakekernels import debug_stub

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


def _describe_debug(registry):
    # Describe a fake debug kernel: R is wired to the operator operand
    # encoded_softmax. (Completeness off — we only need the wiring recorded.)
    debug = debug_stub()
    specs = [
        ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='R'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar('dropout_p', 'fp32'),
        ati.scalar(['Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k'], 'i32'),
        ati.tensor(['philox_seed_ptr', 'philox_offset1'], '*u64', rank=0),
        ati.scalar('philox_offset2', 'u64'),
    ]
    describe(debug, *specs, _validate=False)
    return _testonly_build_kernel_description(debug,
                                    family='flash',
                                    triton_kernel_name='debug_simulate_encoded_softmax',
                                    registry=registry)


def test_apparel_of_wired_arg():
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    assert kdesc.apparel_of('R') == 'encoded_softmax'


def test_apparel_of_unwired_is_identity():
    reg = InterfaceRegistry()
    kdesc = _describe_debug(reg)
    assert kdesc.apparel_of('stride_rz') == 'stride_rz'
    assert kdesc.apparel_of('dropout_p') == 'dropout_p'
    assert kdesc.apparel_of('philox_offset2') == 'philox_offset2'


def test_unwired_kernel_has_empty_wiring():
    # A kernel with no wires_to= has identity apparel everywhere.
    reg = InterfaceRegistry()
    debug = debug_stub()
    specs = [
        ati.tensor('R', '*fp16:16', strides='stride_r?', contiguous=-1),
        ati.scalar('dropout_p', 'fp32'),
        ati.scalar(['Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k'], 'i32'),
        ati.tensor(['philox_seed_ptr', 'philox_offset1'], '*u64', rank=0),
        ati.scalar('philox_offset2', 'u64'),
    ]
    describe(debug, *specs, _validate=False)
    kdesc = _testonly_build_kernel_description(debug, family='flash',
                                    registry=reg)
    assert kdesc.apparel_of('R') == 'R'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} wires_to tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
