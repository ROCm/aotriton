# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Modular migration Step 4: @ati.source imports a Triton source by path and
returns the kernel matching the placeholder def, so the stacked @ati.* decorators
attach to the real kernel without copying it (agent-plans/ati_modular_rev0.md §5a)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash' / 'kernel'))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.decorators import AtiSourceError
from aotriton.template_instantiation.describe import get_kernel_spec

KSRC = '../../modules/flash/kernel/fwd_kernel.py'   # relative to THIS test file's dir (test/ati/)


def test_source_returns_real_kernel():
    @ati.source(KSRC)
    def attn_fwd():
        pass
    # The decorator replaced the placeholder with the real @triton.jit kernel.
    from fwd_kernel import attn_fwd as real
    assert attn_fwd is real


def test_source_name_override():
    @ati.source(KSRC, name='attn_fwd')
    def some_other_local_name():
        pass
    from fwd_kernel import attn_fwd as real
    assert some_other_local_name is real


def test_stacked_specs_attach_to_real_kernel():
    # Stack a spec above @ati.source (NOT @ati.kernel — that would run the full
    # completeness validation, out of scope here). The spec must accumulate onto the
    # REAL kernel returned by source(), proving the stack targets it.
    @ati.scalar('Sm_scale', 'fp32')
    @ati.source(KSRC)
    def attn_fwd():
        pass
    from fwd_kernel import attn_fwd as real
    assert attn_fwd is real
    pending = getattr(real, '__ati_pending__', None)
    assert pending is not None and any('Sm_scale' in s.arg_names for s in pending)
    delattr(real, '__ati_pending__')      # cleanup (avoid leaking onto the kernel)


def test_missing_symbol_errors():
    try:
        @ati.source(KSRC, name='no_such_kernel')
        def whatever():
            pass
    except AtiSourceError as e:
        assert 'no_such_kernel' in str(e)
        return
    raise AssertionError('expected AtiSourceError for missing symbol')


def test_missing_file_errors():
    try:
        @ati.source('../kernel/does_not_exist.py')
        def attn_fwd():
            pass
    except AtiSourceError as e:
        assert 'does not exist' in str(e)
        return
    raise AssertionError('expected AtiSourceError for missing file')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} source tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
