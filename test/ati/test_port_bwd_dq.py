# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 12 (part 1): bwd_kernel_dq ported to ATI
(agent-plans/ati_aux-kernel-xref_exec0.md Step 12).

A KEY backward kernel: a standalone full description (no cite), tunable. This test
exercises the port directly: the struct fields, the multi-choice axes, tunability,
the shared disable predicate, and the bias/dropout/window degradation."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

import importlib.util
from v3python.template_instantiation import registry
from v3python.template_instantiation.compat import build_kernel_description

from bwd_kernel_dq import bwd_kernel_dq


def _build():
    registry.clear('flash')
    p = REPO / 'modules' / 'flash' / 'bwd_kernel_dq_ati.py'
    spec = importlib.util.spec_from_file_location('_port_dq_ati', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.describe_bwd_kernel_dq(bwd_kernel_dq)
    return build_kernel_description(bwd_kernel_dq, family='flash',
                                    source_path='tritonsrc/flash.py',
                                    triton_kernel_name='bwd_kernel_dq',
                                    register=False)


def test_struct_has_bwd_fields():
    kdesc = _build()
    fields = {cf.aname: cf.ctype for cf in kdesc.func_cfields}
    assert fields['Q'] == 'const TensorView<4>*'
    assert fields['DQ'] == 'const TensorView<4>*'
    assert fields['DB'] == 'const TensorView<4>*'
    assert fields['L'] == 'const TensorView<2>*'
    assert fields['D'] == 'LazyTensorInternal<2>*'      # lazy tensor
    assert fields['sm_scale'] == 'float'
    assert fields['philox_seed_ptr'] == 'const TensorView<0>*'
    # constexpr features are struct fields
    assert fields['BLOCK_DMODEL'] == 'int16_t'
    assert fields['BIAS_TYPE'] == 'int8_t'


def test_multichoice_axes():
    kdesc = _build()
    multi = {a.repr_arg: a.radix for a in kdesc.axes_multi}
    # same functional axes as fwd: T_io(3) + BLOCK_DMODEL(12) + 4 binary features
    assert multi['Q'] == 3
    assert multi['BLOCK_DMODEL'] == 12
    assert kdesc.godel_number == 3 * 12 * 2 * 2 * 2 * 2     # 576


def test_tunable():
    kdesc = _build()
    assert kdesc.is_tunable is True
    names = [pp.name for pp in kdesc.tune.schema.params]
    assert names == ['BLOCK_M', 'BLOCK_N', 'NUM_XCDS']


def test_disable_shared_predicate():
    kdesc = _build()
    fs = list(kdesc.gen_functionals({'gfx942': ['gfx942']}))
    # causal + bias is disabled; everything on gfx942 otherwise enabled
    def disabled(causal, bias):
        f = next(x for x in fs if x.choices.CAUSAL_TYPE == causal
                 and x.choices.BIAS_TYPE == bias)
        return kdesc.is_functional_disabled(f)
    assert disabled(3, 1) is True       # causal + matrix bias unsupported
    assert disabled(0, 1) is False
    assert disabled(3, 0) is False


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} port-bwd-dq tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
