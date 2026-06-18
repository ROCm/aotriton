# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AtiNode — the abstract base for all ATI passive description records.

Every def object the ATI pipeline finalises carries exactly one `__ati_node__`
attribute pointing to a concrete AtiNode subclass instance:

  KernelSpec   (@ati.source / describe)  — kernel argument spec
  AffineDecl   (@ati.affine.aiter_asm)   — slim affine-kernel description
  OperatorDecl (@ati.operator)           — operator backend list
  MetroPlan    (@ati.metro_kernel)       — metro sub-kernel wiring plan

The hierarchy lets dispatch use isinstance() rather than string tags:

  node = fn.__ati_node__
  if isinstance(node, MetroPlan):   ...
  elif isinstance(node, OperatorDecl): ...
  # etc.
"""


class AtiNode:
    """Marker base for ATI passive description records ('object files').

    Has __slots__ = () so @dataclass(slots=True) subclasses can declare their
    own __slots__ without conflict, and MetroPlan (with explicit __slots__) also
    inherits cleanly via multiple inheritance alongside StackedSpec."""
    __slots__ = ()
