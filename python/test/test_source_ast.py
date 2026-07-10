# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""@ati.source is triton-free: it AST-parses the kernel file for its parameter names
and never imports it, so the code generator runs without the triton package
(agent-plans/ati_triton-free_exec0.md). These tests prove the parse works even when
`import triton` would fail."""

import sys
import builtins
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.decorators import KernelStub
from aotriton.template_instantiation.introspect import kernel_params

KSRC = 'fakefamily/flash/kernel/fwd_kernel.py'   # relative to python/test/ (fake family)


def test_ast_params_match_full_signature():
    @ati.source(KSRC)
    def attn_fwd():
        pass
    names = attn_fwd.params
    # The fwd kernel's full 74-parameter ARGUMENTS order, extracted without import.
    assert len(names) == 74
    assert names[:8] == ['Q', 'K', 'V', 'B', 'A', 'Sm_scale', 'L', 'Out']
    # constexpr-annotated params are present by NAME (types come from @ati.* later).
    assert 'BLOCK_DMODEL' in names
    assert 'philox_offset2' in names


def test_kernel_params_from_stub_has_no_annotations():
    @ati.source(KSRC)
    def attn_fwd():
        pass
    ps = kernel_params(attn_fwd)
    assert [p.name for p in ps] == attn_fwd.params
    # The stub path carries no triton annotations (types are explicit in @ati.*).
    assert all(not p.has_annotation for p in ps)
    assert all(p.is_constexpr is False for p in ps)


def test_source_does_not_import_triton():
    # Simulate triton being unavailable: block the import outright. @ati.source must
    # still succeed (it only reads the file's AST).
    real_import = builtins.__import__

    def no_triton(name, *a, **k):
        if name == 'triton' or name.startswith('triton.'):
            raise ModuleNotFoundError("No module named 'triton'")
        return real_import(name, *a, **k)

    saved = {m: sys.modules.pop(m) for m in list(sys.modules)
             if m == 'triton' or m.startswith('triton.')}
    builtins.__import__ = no_triton
    try:
        @ati.source(KSRC)
        def attn_fwd():
            pass
        assert isinstance(attn_fwd, KernelStub)
        assert attn_fwd.params[0] == 'Q'
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} source-AST tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
