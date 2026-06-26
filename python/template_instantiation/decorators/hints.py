# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI author hints (`ati.hints.*`).

Hints are OPTIONAL author annotations that resolve ambiguities the linker cannot
decide on its own. Unlike the core @ati.* decorators they carry no kernel semantics
— they only steer merge/precedence decisions.

  * union_precedence(kernels): on a @ati.metro_kernel, the priority order of the
    metro's sub-kernels when their argument bindings collide. Two collaborating
    sub-kernels may name the SAME operand's strides differently (e.g. the bwd metro's
    preprocess kernels call dO's 4th stride `stride_don` while the key kernels call it
    `stride_dok`); when a sub-kernel @ati.cites the whole metro, the gap donor must be
    the KEY kernel, not whichever sub-kernel happens to come first in call order. This
    hint lists the sub-kernels (by def or name) highest-priority first, so the linker
    picks the key kernel's binding. It also fixes the operator params-struct union
    order (build_merged_struct_cfields) to the same priority.
"""


from ..specs.base import StackedSpec


class UnionPrecedenceSpec(StackedSpec):
    """The @ati.hints.union_precedence record: an ordered list of sub-kernel NAMEs
    (highest priority first). Accumulates onto the metro def's pending list via the
    standard StackedSpec.__call__; @ati.start reads it when finalising the metro."""

    __slots__ = ('names',)

    def __init__(self, kernels):
        names = []
        for k in kernels:
            name = k if isinstance(k, str) else getattr(k, '__name__', None)
            assert name, (
                f'@ati.hints.union_precedence entries must be sub-kernel defs or '
                f'name strings, got {k!r}')
            names.append(name)
        self.names = names

    def __repr__(self):
        return f'UnionPrecedenceSpec({self.names!r})'


def union_precedence(kernels):
    """Declare the metro sub-kernels' merge priority (highest first). See module doc."""
    return UnionPrecedenceSpec(kernels)
