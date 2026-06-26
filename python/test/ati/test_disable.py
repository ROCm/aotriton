# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 3.3: @ati.disable functional-disable decorator. The flash predicates
reproduce the exact legacy attn_fwd disabled set across all functionals."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

import pytest
import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.builder import build_kernel
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import InterfaceRegistry, _testonly_build_kernel_description
from aotriton.gpu_targets import cluster_gpus
from aot.attn_fwd import attn_fwd
try:
    import v3python.rules.flash as F   # legacy reference for parity comparison
except ModuleNotFoundError:
    F = None   # legacy reference unavailable (v3python removed) -> parity tests skip


def test_disable_spec_partitioned_and_built():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk, CAUSAL_TYPE: 'constexpr'):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
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
    if F is None:
        pytest.skip('v3python legacy reference unavailable')
    # The @ati.disable predicates must reproduce the legacy is_functional_disabled
    # set for attn_fwd, functional-for-functional, on a representative arch.
    reg = InterfaceRegistry()
    ak = _testonly_build_kernel_description(attn_fwd, family='flash',
                                    registry=reg)
    leg = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    ta = cluster_gpus(['gfx942_mod0'])
    lf = {f.godel_number: f for f in leg.gen_functionals(ta)}
    mf = {f.godel_number: f for f in ak.gen_functionals(ta)}
    mism = [g for g in lf
            if leg.is_functional_disabled(lf[g]) != ak.is_functional_disabled(mf[g])]
    assert not mism, f'{len(mism)} disable mismatches, e.g. {mism[:5]}'


def main():
    if F is None:
        print('SKIP: v3python legacy reference unavailable; parity test skipped.')
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} disable tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
