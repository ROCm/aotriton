# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 3.3: @ati.disable functional-disable decorator. The flash predicates
reproduce the exact legacy attn_fwd disabled set across all functionals."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))
sys.path.insert(0, str(REPO / 'modules'))

import v3python.template_instantiation as ati
from v3python.template_instantiation.describe import describe, get_kernel_spec
from v3python.template_instantiation.builder import build_kernel
from v3python.template_instantiation.compat import build_kernel_description
from v3python.gpu_targets import cluster_gpus
from fwd_kernel import attn_fwd
from flash.attn_fwd_ati import describe_attn_fwd
import v3python.rules.flash as F


def test_disable_spec_partitioned_and_built():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk, CAUSAL_TYPE: 'constexpr'):
        pass
    T = ati.tensor_dtype('T', dtype=['*fp16:16'])
    describe(k,
             ati.tensor('Q', T, strides='stride_q?'),
             ati.scalar('CAUSAL_TYPE', options=[0, 3]),
             ati.disable(when=lambda f: f.choices.CAUSAL_TYPE != 0),
             _validate=False)
    bk = build_kernel(get_kernel_spec(k))
    assert len(bk.disables) == 1


def test_disable_requires_callable():
    try:
        ati.disable(when='not callable')
    except AssertionError:
        return
    raise AssertionError('expected callable assertion')


def test_flash_disabled_set_matches_legacy():
    # The @ati.disable predicates must reproduce the legacy is_functional_disabled
    # set for attn_fwd, functional-for-functional, on a representative arch.
    describe_attn_fwd(attn_fwd)
    ak = build_kernel_description(attn_fwd, family='flash')
    leg = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    ta = cluster_gpus(['gfx942_mod0'])
    lf = {f.godel_number: f for f in leg.gen_functionals(ta)}
    mf = {f.godel_number: f for f in ak.gen_functionals(ta)}
    mism = [g for g in lf
            if leg.is_functional_disabled(lf[g]) != ak.is_functional_disabled(mf[g])]
    assert not mism, f'{len(mism)} disable mismatches, e.g. {mism[:5]}'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} disable tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
