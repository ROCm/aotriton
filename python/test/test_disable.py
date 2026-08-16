# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 3.3: @ati.disable functional-disable decorator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.builder import build_kernel


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


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} disable tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
