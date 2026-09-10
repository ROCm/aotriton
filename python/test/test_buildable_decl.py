# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`BuildableDecl`: the surface resolve_cites()/build_kernel() read.

`KernelDecl` and `FlycDecl` are SIBLINGS. They share what the builder pipeline
consumes and nothing else -- a flyc record additionally carries `desc_path`,
`hints_cls` and the deferred builder `fn`, and the two are expected to diverge
further.

These tests exist because that sharing used to be expressed as a CAST: the flyc
path converted its record into a `KernelDecl` before building. That asserted a
subtype relationship which does not hold, discarded the flyc-only fields, and
needed a new line each time either type gained one. The contract replaced it, so
the contract needs guarding: an attribute added to `BUILDABLE_ATTRS` must be
supplied by both, and `clone()` must keep returning the concrete type it was
called on.
"""

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.specs.node import (
    BUILDABLE_ATTRS, BuildableDecl,
)

_MODULES = Path(__file__).resolve().parents[2] / 'modules'
from aotriton.template_instantiation.specs.kernel import KernelDecl
from aotriton.template_instantiation.specs.flyc import FlycDecl

_IMPLS = (KernelDecl, FlycDecl)


def test_both_decls_declare_every_buildable_attr():
    for cls in _IMPLS:
        names = {f.name for f in dataclasses.fields(cls)}
        missing = [a for a in BUILDABLE_ATTRS if a not in names]
        assert not missing, f'{cls.__name__} is missing {missing}'


def test_both_decls_are_buildable_and_cloneable():
    for cls in _IMPLS:
        assert issubclass(cls, BuildableDecl), f'{cls.__name__} is not a BuildableDecl'
        assert callable(getattr(cls, 'clone', None)), f'{cls.__name__} has no clone()'
        assert isinstance(getattr(cls, 'param_names', None), property)


def test_clone_returns_the_same_concrete_type():
    """The anti-cast test. A clone that returned KernelDecl for a FlycDecl would
    be the old conversion wearing a new name."""
    from aotriton.codegen.linker import Linker
    _k, _o, _a, flycs = Linker(_MODULES).link_all_families()
    decl = flycs[0].kernel_decl
    assert type(decl) is FlycDecl, (
        f'the flyc build path produced a {type(decl).__name__}; it must stay a '
        f'FlycDecl rather than being cast into a KernelDecl')
    assert type(decl.clone()) is FlycDecl


def test_clone_gives_fresh_containers():
    """resolve_cites appends into these, so a shared list would leak resolution
    output back into the module-level record every description reads."""
    from aotriton.codegen.linker import Linker
    _k, _o, _a, flycs = Linker(_MODULES).link_all_families()
    orig = flycs[0].kernel_decl
    copy = orig.clone()
    for attr in ('tensors', 'scalars', 'overrides', 'dtype_vars', 'cites'):
        assert getattr(copy, attr) is not getattr(orig, attr), f'{attr} is shared'
        assert getattr(copy, attr) == getattr(orig, attr), f'{attr} lost content'


def test_clone_copies_every_field():
    """The reflection gate. `clone()` walks `dataclasses.fields()`, so a field
    added to either record is copied for free -- this asserts it, because the
    per-class argument lists that preceded it would have dropped a new field
    silently and the linker builds from clones."""
    from aotriton.codegen.linker import Linker
    _k, _o, _a, flycs = Linker(_MODULES).link_all_families()
    orig = flycs[0].kernel_decl
    copy = orig.clone()
    for f in dataclasses.fields(orig):
        assert getattr(copy, f.name) == getattr(orig, f.name), f'{f.name} not copied'
    assert {f.name for f in dataclasses.fields(orig)} >= set(BUILDABLE_ATTRS)


def test_flyc_only_fields_survive_the_clone():
    """The anti-cast assertion in field terms: converting to KernelDecl would
    have dropped these three, which is what made the conversion lossy."""
    from aotriton.codegen.linker import Linker
    _k, _o, _a, flycs = Linker(_MODULES).link_all_families()
    copy = flycs[0].kernel_decl.clone()
    for attr in ('desc_path', 'hints_cls', 'fn'):
        assert getattr(copy, attr) is not None, f'{attr} lost'


def test_source_path_is_the_one_name_for_the_vendored_directory():
    """`FlycDecl` used to carry the vendored kernel path twice -- once as
    `module_path`, once inside an eagerly-resolved KernelStub. One fact, one
    field -- still the rule here, just a narrower fact: `source_path` is the
    vendored flyc DIRECTORY (e.g. `modules/flash/flyc/`), not a specific
    kernel FILE, because the real per-arch file/def name is not knowable at
    link time (see specs/flyc.py's `_synth_param_order` and
    `collect_flyc_decl`). `decl.kernel` stays `None` for a `FlycDecl` --
    there is no eager KernelStub to resolve `source_path` from, so unlike a
    Triton `KernelDecl` the two cannot be cross-checked against each
    other."""
    from aotriton.codegen.linker import Linker
    _k, _o, _a, flycs = Linker(_MODULES).link_all_families()
    decl = flycs[0].kernel_decl
    assert not hasattr(decl, 'module_path')
    assert decl.kernel is None
    assert decl.source_path is not None
    assert Path(decl.source_path).name == 'flyc'
    assert Path(decl.source_path).is_dir()


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} BuildableDecl tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
