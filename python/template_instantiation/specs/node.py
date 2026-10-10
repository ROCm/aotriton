# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AtiNode / BuildableDecl — the bases for all ATI passive description records.

Every def object the ATI pipeline finalises carries exactly one `__ati_node__`
attribute pointing to a concrete AtiNode subclass instance:

  KernelDecl   (@ati.source / describe)  — kernel argument spec
  AffineDecl   (@ati.affine.aiter_asm)   — slim affine-kernel description
  OperatorDecl (@ati.operator)           — operator backend list
  MetroSpec    (@ati.metro_kernel)       — metro sub-kernel wiring plan

The hierarchy lets dispatch use isinstance() rather than string tags:

  node = fn.__ati_node__
  if isinstance(node, MetroSpec):   ...
  elif isinstance(node, OperatorDecl): ...
  # etc.
"""

import dataclasses
from dataclasses import dataclass, field


class AtiNode:
    """Marker base for ATI passive description records ('object files').

    Has __slots__ = () so @dataclass(slots=True) subclasses can declare their
    own __slots__ without conflict, and MetroSpec (with explicit __slots__) also
    inherits cleanly via multiple inheritance alongside StackedSpec."""
    __slots__ = ()


#: Every field a `BuildableDecl` carries. `resolve_cites()` (ir/ops/cite.py) and
#: `build_kernel()` (builder/kernel.py) read from this set; `source_path` is
#: additionally read by the parser and the linker. Kept as data so
#: `python/test/test_buildable_decl.py` can assert `clone()` covers all of it --
#: the failure mode a hand-written field list has.
BUILDABLE_ATTRS = (
    'name', 'kernel', 'params', 'tensors', 'scalars', 'overrides',
    'tune', 'disable', 'resolved_disables', 'dtype_vars', 'cites',
    'source_path',
)


@dataclass(kw_only=True)
class BuildableDecl(AtiNode):
    """A description the builder pipeline can lower: `resolve_cites()` then
    `build_kernel()`.

    Holds every field the builder pipeline reads, which today is all of
    `KernelDecl`'s -- the Triton record adds nothing of its own. A second kind of
    description is a SIBLING of `KernelDecl`, never a subtype of it: converting
    one record into the other before building asserts a relationship that does
    not hold, silently drops whatever the other kind adds, and hides every
    subsequent divergence behind the conversion.

    **These are data structures, not interfaces.** No behaviour at construction:
    there is deliberately no `__post_init__` anywhere in the hierarchy. A field
    that has to be derived is derived by the COLLECTOR that builds the record --
    a collector holds the kernel object, so it can compute
    `name`/`source_path`/`resolved_disables` and pass them in
    (`derived_decl_fields` below). A base class reaching into fields declared by
    its subclasses is duck-typing in an inheritance costume: invisible at the
    point of use and undetectable until construction fails.

    `kw_only=True` throughout. With shared fields on the base, positional order
    would be fixed by the base and every subclass field would have to follow it;
    keyword-only makes required and defaulted fields interleave freely and reads
    better for a record this wide.
    """

    name: str
    kernel: object = None
    params: list = field(default_factory=list)      # signature order
    tensors: list = field(default_factory=list)
    scalars: list = field(default_factory=list)
    overrides: list = field(default_factory=list)
    # Named dtype/choice variables from @ati.type_var / @ati.scalar_var. Ones
    # passed inline by object are NOT here -- they ride on the spec's dtype.
    dtype_vars: list = field(default_factory=list)
    cites: list = field(default_factory=list)       # @ati.cite targets, source order
    tune: object = None                             # TuneSpec | None
    # DECLARED cardinality: at most one, enforced by specs/bundle.py's
    # partition() at decoration time.
    disable: object = None
    # RESOLVED cardinality: 0-N. Seeded from `disable` by the collector, then
    # rewritten by resolve_cites -- a whole-metro cite can expand one declared
    # disable into several inherited ones. Downstream consumers (build_kernel,
    # BuiltKernel.disables) read this, never `disable`.
    resolved_disables: list = field(default_factory=list)
    # The file the kernel def lives in. ONE name for one fact, so a second kind
    # of record cannot spell the same string a second way.
    source_path: str | None = None

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def clone(self):
        """A copy with FRESH mutable containers, returning the SAME concrete type.

        `resolve_cites` appends into tensors/scalars/overrides/dtype_vars and
        assigns tune/resolved_disables, so it must work on a copy; the
        module-level record every description and test reads stays untouched.
        That is what makes linking idempotent.

        Reflective over `dataclasses.fields()` rather than a hand-written
        argument list: a field added to any record is copied automatically, where
        a per-class list would have dropped it silently. `type(self)(...)` is
        what keeps a subclass's clone its own type -- a clone that returned the
        base would be a cast wearing a copy's name.
        """
        kw = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            kw[f.name] = list(v) if isinstance(v, list) else v
        return type(self)(**kw)


def derived_decl_fields(kernel, disable, name=None):
    """The record fields a collector computes rather than reads off the stack.

    Lives here so every collector shares one implementation without the record
    itself growing construction behaviour. `name` defaults to the kernel object's
    `__name__`, which is right wherever the description and the GPU entry symbol
    are the same identifier -- true of every Triton kernel, and not guaranteed
    for a description whose kernel def is located in a vendored file by
    decorator; such a collector passes `name` explicitly.
    """
    from ..decorators.source import KernelStub
    return {
        'name': name or getattr(kernel, '__name__', 'kernel'),
        'kernel': kernel,
        'disable': disable,
        'resolved_disables': [disable] if disable is not None else [],
        'source_path': kernel.source_path if isinstance(kernel, KernelStub) else None,
    }
