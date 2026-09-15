# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collector unification, KernelSpec -> KernelDecl rename: the record attached
to a Triton kernel as `fn.__ati_node__` used to be the one passive Stage-2
"object file" not named `*Decl` (AffineDecl / OperatorDecl / FlycDecl already
were). Once cite resolution writes only a dedicated `resolved_disables` field
on a per-link clone -- rather than mutating the declared record's own fields --
the declared record is passive exactly like the other three, so the name
asymmetry no longer reflects anything real. This test pins the rename so a
future edit cannot reintroduce a bare `KernelSpec` (by hand, or by copy-paste
from history) without failing CI.

`AffineKernelSpec` / `FlycKernelSpec` are a DIFFERENT thing -- the stacked-@
marker record for '@ati.affine(...)' / '@ati.flyc(...)' itself, one rung down
in the *Spec-record / *Decl-collection vocabulary (specs/bundle.py's module
docstring) -- and must NOT be swept up by a blind rename."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PYTHON_ROOT = Path(__file__).resolve().parent.parent
_BARE_KERNEL_SPEC = re.compile(r'\bKernelSpec\b')
_AFFINE_MARKER = re.compile(r'\bAffineKernelSpec\b')
_FLYC_MARKER = re.compile(r'\bFlycKernelSpec\b')


_THIS_FILE = Path(__file__).resolve()


def _source_files():
    # Excludes __pycache__, build output under pytest-gpu-lease/build (a
    # vendored plugin's own build tree, not this project's source), and this
    # file itself -- which necessarily spells out the very identifiers it is
    # checking for/against in its docstrings, regexes, and assertions.
    return [p for p in PYTHON_ROOT.rglob('*.py')
            if '__pycache__' not in p.parts
            and 'pytest-gpu-lease' not in p.parts
            and p.resolve() != _THIS_FILE]


def test_no_bare_kernel_spec_remains():
    hits = []
    for path in _source_files():
        # encoding= is required, not decorative: this walks EVERY .py under
        # python/, nearly all of which carry a `©` in the copyright header, so
        # under a non-UTF-8 locale (LC_ALL=C, PYTHONUTF8=0) the platform
        # default decoder raises UnicodeDecodeError and this test fails for a
        # reason that has nothing to do with the rename it checks.
        text = path.read_text(encoding='utf-8')
        for i, line in enumerate(text.splitlines(), start=1):
            if _BARE_KERNEL_SPEC.search(line):
                hits.append(f'{path}:{i}: {line.strip()}')
    assert not hits, 'bare KernelSpec found (should be KernelDecl now):\n' + '\n'.join(hits)


def test_affine_and_flyc_markers_survive_the_rename():
    """The two stack markers are a DIFFERENT layer than KernelDecl (they are
    `*Spec` records, not the `*Decl` collection) and the rename must not have
    swept them up. Pin their count equal and nonzero rather than a fixed
    number: what matters is that they still exist as distinct symbols the
    common `partition()`/`describe()` machinery still isinstance-checks
    against, not the exact occurrence count, which grows harmlessly whenever
    a docstring gains another cross-reference to them."""
    affine_hits = sum(len(_AFFINE_MARKER.findall(p.read_text(encoding='utf-8')))
                      for p in _source_files())
    flyc_hits = sum(len(_FLYC_MARKER.findall(p.read_text(encoding='utf-8')))
                    for p in _source_files())
    assert affine_hits > 0, 'AffineKernelSpec marker was swept up by the KernelSpec rename'
    assert flyc_hits > 0, 'FlycKernelSpec marker was swept up by the KernelSpec rename'


def test_kernel_decl_importable_and_kernel_spec_is_not():
    from aotriton.template_instantiation.specs import kernel as kernel_module
    assert hasattr(kernel_module, 'KernelDecl')
    assert not hasattr(kernel_module, 'KernelSpec')


def test_every_ati_node_collection_is_named_decl():
    """Every collection type attached as `fn.__ati_node__` -- the thing the
    linker's `_node_kind` dispatches on -- is named `*Decl`. This is what the
    rename was for: one consistent vocabulary across all four stack kinds."""
    from aotriton.template_instantiation.specs.kernel import KernelDecl
    from aotriton.template_instantiation.specs.affine import AffineDecl
    from aotriton.template_instantiation.specs.operator import OperatorDecl
    from aotriton.template_instantiation.specs.flyc import FlycDecl
    for cls in (KernelDecl, AffineDecl, OperatorDecl, FlycDecl):
        assert cls.__name__.endswith('Decl'), f'{cls.__name__} does not end in Decl'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} kernel-decl-rename tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
