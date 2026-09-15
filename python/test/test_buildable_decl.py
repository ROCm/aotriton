# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`BuildableDecl`: the surface resolve_cites()/build_kernel() read.

A kind of description is a SIBLING of `KernelDecl`, never a subtype of it: the
two share exactly what the builder pipeline consumes, and each is free to carry
whatever else its own backend needs.

These tests guard that contract, because the alternative shape -- one record
CAST into another before building -- is easy to reach for and lossy every time:
it asserts a subtype relationship that does not hold, silently discards whatever
the source record carries beyond the target's fields, and needs a new line each
time either type gains one. So: every attribute in `BUILDABLE_ATTRS` must be
declared by every record, and `clone()` must keep returning the concrete type it
was called on.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.specs.node import (
    BUILDABLE_ATTRS, BuildableDecl,
)

# The fake family, not the real modules/ tree: these tests only need one real
# cite-resolved decl, and linking the shipping descriptions inside a unit test
# leaves their module-level records visible to every later test.
_MODULES = Path(__file__).resolve().parent / 'fakefamily'
from aotriton.template_instantiation.specs.kernel import KernelDecl

_IMPLS = (KernelDecl,)


def _a_kernel_decl():
    """A real, cite-resolved decl off the linker -- not a hand-built one, so the
    clone under test is the object the builder is actually handed."""
    from aotriton.codegen.linker import Linker
    kernels, _o, _a = Linker(_MODULES).link_all_families()
    return kernels[0].kernel_decl


def test_every_decl_declares_every_buildable_attr():
    for cls in _IMPLS:
        names = {f.name for f in dataclasses.fields(cls)}
        missing = [a for a in BUILDABLE_ATTRS if a not in names]
        assert not missing, f'{cls.__name__} is missing {missing}'


def test_every_decl_is_buildable_and_cloneable():
    for cls in _IMPLS:
        assert issubclass(cls, BuildableDecl), f'{cls.__name__} is not a BuildableDecl'
        assert callable(getattr(cls, 'clone', None)), f'{cls.__name__} has no clone()'
        assert isinstance(getattr(cls, 'param_names', None), property)


def test_clone_returns_the_same_concrete_type():
    """The anti-cast test. A clone that returned the BASE would be a conversion
    wearing a copy's name."""
    decl = _a_kernel_decl()
    assert type(decl.clone()) is type(decl)


def test_clone_gives_fresh_containers():
    """resolve_cites appends into these, so a shared list would leak resolution
    output back into the module-level record every description reads."""
    orig = _a_kernel_decl()
    copy = orig.clone()
    for attr in ('tensors', 'scalars', 'overrides', 'dtype_vars', 'cites'):
        assert getattr(copy, attr) is not getattr(orig, attr), f'{attr} is shared'
        assert getattr(copy, attr) == getattr(orig, attr), f'{attr} lost content'


def test_clone_copies_every_field():
    """The reflection gate. `clone()` walks `dataclasses.fields()`, so a field
    added to any record is copied for free -- this asserts it, because the
    per-class argument list that preceded it would have dropped a new field
    silently and the linker builds from clones."""
    orig = _a_kernel_decl()
    copy = orig.clone()
    for f in dataclasses.fields(orig):
        assert getattr(copy, f.name) == getattr(orig, f.name), f'{f.name} not copied'
    assert {f.name for f in dataclasses.fields(orig)} >= set(BUILDABLE_ATTRS)


def test_source_path_is_the_one_name_for_the_kernel_file():
    """One fact, one field: the kernel file is `source_path` and nothing else.
    A second record spelling it `module_path` is how the two drift."""
    decl = _a_kernel_decl()
    assert not hasattr(decl, 'module_path')
    assert decl.source_path == decl.kernel.source_path


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} BuildableDecl tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
