# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5.3: --sancheck description validation. Clean on a good description;
reports ALL errors at once on a seeded-bad one (not first-failure)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.tools import sancheck_kernel_spec


def _good_kernel():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk, CAUSAL_TYPE: 'constexpr'):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
    describe(k,
             ati.tensor('Q', T, strides='stride_q?', contiguous=-1),
             ati.scalar('CAUSAL_TYPE', options=[0, 3]),
             ati.derives('Q', to=0, when=ati.eq('CAUSAL_TYPE', 0)),
             _validate=False)
    return get_kernel_spec(k)


def test_clean_description_no_errors():
    assert sancheck_kernel_spec(_good_kernel()) == []


def test_orphan_and_bad_target_reported_together():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk, Z, CAUSAL_TYPE: 'constexpr'):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
    describe(k,
             ati.tensor('Q', T, strides='stride_q?'),
             ati.scalar('CAUSAL_TYPE', options=[0, 3]),
             ati.derives('NoSuchArg', to=0, when=ati.eq('CAUSAL_TYPE', 0)),
             _validate=False)
    errs = sancheck_kernel_spec(get_kernel_spec(k))
    # both problems surface in one pass
    assert any("'Z'" in e and 'not claimed' in e for e in errs)
    assert any("'NoSuchArg'" in e for e in errs)
    assert len(errs) >= 2


def test_double_claim_reported():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
    describe(k,
             ati.tensor('Q', T, strides='stride_q?'),
             ati.scalar('Q', 'i32'),             # Q claimed twice
             _validate=False)
    errs = sancheck_kernel_spec(get_kernel_spec(k))
    assert any("'Q'" in e and 'multiple' in e for e in errs)


def test_predicate_scope_reported():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk, CAUSAL_TYPE: 'constexpr'):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
    describe(k,
             ati.tensor('Q', T, strides='stride_q?'),
             ati.scalar('CAUSAL_TYPE', options=[0, 3]),
             # predicate over a non-free, non-arch variable -> scope error
             ati.derives('Q', to=0, when=ati.eq('NotAnAxis', 0)),
             _validate=False)
    errs = sancheck_kernel_spec(get_kernel_spec(k))
    assert any('NotAnAxis' in e for e in errs)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} sancheck tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
