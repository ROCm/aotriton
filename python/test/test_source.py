# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Modular migration Step 4: @ati.source resolves a Triton source by path and returns
a KernelStub describing the kernel matching the placeholder def, so the stacked @ati.*
decorators attach to it without copying the kernel (agent-plans/ati_modular_rev0.md
§5a). The source is AST-parsed, NOT imported — the generator needs no triton package
(agent-plans/ati_triton-free_exec0.md)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.decorators import SourceError, KernelStub

KSRC = '../../modules/flash/kernel/fwd_kernel.py'   # relative to THIS test file dir (python/test/)


def test_source_returns_stub():
    @ati.source(KSRC)
    def attn_fwd():
        pass
    # The decorator replaced the placeholder with a KernelStub for the parsed kernel.
    assert isinstance(attn_fwd, KernelStub)
    assert attn_fwd.__name__ == 'attn_fwd'
    assert attn_fwd.source_path.endswith('fwd_kernel.py')
    # AST-extracted ARGUMENTS order (no import): the first params of attn_fwd.
    assert attn_fwd.params[:4] == ['Q', 'K', 'V', 'B']


def test_source_name_override():
    @ati.source(KSRC, name='attn_fwd')
    def some_other_local_name():
        pass
    assert isinstance(some_other_local_name, KernelStub)
    assert some_other_local_name.__name__ == 'attn_fwd'
    assert some_other_local_name.params[:3] == ['Q', 'K', 'V']


def test_stacked_specs_attach_to_stub():
    # Stack a spec above @ati.source (NOT @ati.start — that would run the full
    # completeness validation, out of scope here). The spec must accumulate onto the
    # KernelStub returned by source(), proving the stack targets it.
    @ati.scalar('Sm_scale', 'fp32')
    @ati.source(KSRC)
    def attn_fwd():
        pass
    assert isinstance(attn_fwd, KernelStub)
    pending = getattr(attn_fwd, '__ati_pending__', None)
    assert pending is not None and any('Sm_scale' in s.arg_names for s in pending)


def test_placeholder_string_annotations_captured():
    # STRING type annotations on the placeholder def are captured for the finalizer to
    # turn into ScalarSpecs (the terse `def k(x: 'fp32')` form).
    @ati.source(KSRC)
    def attn_fwd(dropout_p: 'fp32', philox_seed: '*u64'):
        pass
    assert isinstance(attn_fwd, KernelStub)
    assert attn_fwd.annotations == {'dropout_p': 'fp32',
                                            'philox_seed': '*u64'}


def test_placeholder_nonstring_annotations_disregarded():
    # A non-string annotation (a stray object) is NOT captured: only ATI type strings
    # are a type source. (object() stands in for a triton dtype / tl.constexpr.)
    marker = object()
    @ati.source(KSRC)
    def attn_fwd(a: 'i32', b: marker):
        pass
    assert attn_fwd.annotations == {'a': 'i32'}


def test_missing_symbol_errors():
    try:
        @ati.source(KSRC, name='no_such_kernel')
        def whatever():
            pass
    except SourceError as e:
        assert 'no_such_kernel' in str(e)
        return
    raise AssertionError('expected SourceError for missing symbol')


def test_missing_file_errors():
    try:
        @ati.source('../kernel/does_not_exist.py')
        def attn_fwd():
            pass
    except SourceError as e:
        assert 'does not exist' in str(e)
        return
    raise AssertionError('expected SourceError for missing file')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} source tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
