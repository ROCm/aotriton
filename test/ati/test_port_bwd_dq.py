# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 12 (part 1): bwd_kernel_dq ported to ATI
(agent-plans/ati_aux-kernel-xref_exec0.md Step 12).

A KEY backward kernel, tunable. Since the ATI-linker acceptance demo
(ati_linker_exec0 Step 7), bwd_kernel_dq is CITE-based: it declares only its dQ/dB
outputs + perf and @ati.cites the whole metro (op_attn_bwd.triton_split) for the
shared inputs. So it is built through the two-pass LINKER (which resolves the
whole-metro cite), not a standalone build_kernel_description. This test exercises the
linked result: struct fields (incl. the inherited Q/L/D + local DQ/DB), the
multi-choice axes, tunability, and the inherited shared disable predicate."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aotriton.codegen.linker import link_all_families


def _build():
    kernels, _ops, _aff = link_all_families()
    return next(k for k in kernels if k.NAME == 'bwd_kernel_dq')


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
    # same functional axes as fwd: T_io(3) + BLOCK_DMODEL(12) + 4 binary features.
    # The T_io tensor axis groups Q/K/V/B/DO (inherited) + DQ/DB (local); its repr is
    # the representative member, so assert on the grouped T_io axis carrying Q.
    tio = next(a for a in kdesc.axes_multi if 'Q' in a.arg_names)
    assert tio.radix == 3
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
