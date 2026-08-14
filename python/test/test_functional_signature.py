# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.2.3: AtiFunctional signature/packing surface -- the multi-choice
signature_name requirement is enforced.

(Legacy parity tests against v3python.rules.flash's attn_fwd functionals were
removed when v3python/ was deleted; see agent-plans/modular-tune.md Phase 1
step 5 / F-list §2g.)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.builder import build_kernel, DescriptionError


def test_multichoice_shared_var_requires_signature_name():
    # A multi-choice variable spanning >1 arg without signature_name -> error.
    def k(a, b):
        pass
    T = ati.type_var('T', dtype=['*fp16:16', '*bf16:16'])   # 2 choices, no sig name
    describe(k, ati.tensor('a', T, rank=2), ati.tensor('b', T, rank=2),
             _validate=False)
    try:
        build_kernel(get_kernel_spec(k))
    except DescriptionError as e:
        assert 'signature_name' in str(e)
        return
    raise AssertionError('expected signature_name requirement error')


def test_single_choice_shared_var_exempt():
    # A single-choice variable is trivial -> no signature_name needed.
    def k(a, b):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])               # 1 choice
    describe(k, ati.tensor('a', T, rank=2), ati.tensor('b', T, rank=2),
             _validate=False)
    bk = build_kernel(get_kernel_spec(k))           # must not raise
    assert bk is not None


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} functional-signature tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
