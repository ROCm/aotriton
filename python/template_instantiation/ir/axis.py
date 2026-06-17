# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Axis — a free dimension of a kernel's instantiation space (ATI executive plan
Step 1.2; see agent-plans/ati+newbinds_rev1.md §2.1, §4, §5).

One Axis per choice variable (tensor_dtype / choice_set / enumerated scalar). It
owns the variable name (the public identity used by f.choices.<var_name>), the
arguments it instantiates, the ordered concrete TypedChoices, and per-argument shape
(rank, contiguous). Overrides are NOT axes — only choice variables create
combinatorial dimensions.

Godel numbering is mixed-radix, big-endian: the earliest-anchored axis is the
most significant digit, mirroring the C++-side ladder that recomputes the number
from the params struct. Both orderings must be reproducible and identical, so the
canonical axis order is ascending `anchor` (earliest signature argument first).
"""

from .typed_choice import TypedChoice


class Axis:
    __slots__ = ('var_name', 'arg_names', 'choices', 'anchor',
                 'ranks', 'contiguous', 'godel_stride', 'kind', 'stride_of',
                 'signature_name')

    # kind values: 'tensor' | 'scalar' | 'stride' | 'stride_unit'
    def __init__(self, var_name, arg_names, choices, anchor,
                 ranks=None, contiguous=None, kind='scalar', stride_of=None,
                 signature_name=None):
        self.var_name = var_name
        self.arg_names = tuple(arg_names)
        self.choices = tuple(choices)
        assert all(isinstance(c, TypedChoice) for c in self.choices), \
            'Axis.choices must be TypedChoice objects'
        self.anchor = anchor
        # per-arg shape; only meaningful for tensor axes
        self.ranks = dict(ranks) if ranks else {}
        self.contiguous = dict(contiguous) if contiguous else {}
        self.godel_stride = None   # assigned by assign_godel over multi-choice axes
        self.kind = kind
        # for stride axes: (tensor_arg, dim_index) the stride belongs to
        self.stride_of = stride_of
        # the argument recording this axis in persisted artifacts (compact
        # signature, aks2/zip entry name, DB row key); explicit for shared
        # multi-choice variables, else the first argument.
        self.signature_name = signature_name or self.arg_names[0]

    @property
    def repr_arg(self) -> str:
        """The representative REAL argument of this axis — the first-appearing
        kernel argument it instantiates. This is the key used to look up the axis's
        resolved value (all members of an axis share one choice per functional,
        modulo per-arg overrides). It is ALWAYS a real argument, unlike
        `signature_name`, which is only the LABEL recorded in persisted artifacts
        and may be set to anything (e.g. 'dtype'). Never use `signature_name` to
        index the resolved table."""
        return self.arg_names[0]

    @property
    def is_stride(self) -> bool:
        return self.kind in ('stride', 'stride_unit')

    @property
    def radix(self) -> int:
        return len(self.choices)

    @property
    def is_trivial(self) -> bool:
        """A single-choice axis: pins a type but adds no combinations, so it is
        excluded from godel digits (a digit that is always 0)."""
        return self.radix <= 1

    def choice_for_arg(self, nth: int, arg_name: str) -> TypedChoice:
        """The nth choice, specialized to a concrete rank for a tensor arg."""
        c = self.choices[nth]
        if c.is_tensor and arg_name in self.ranks:
            return c.with_rank(self.ranks[arg_name])
        return c

    def __repr__(self):
        return (f'Axis({self.var_name!r}, args={self.arg_names}, '
                f'radix={self.radix}, anchor={self.anchor})')


def assign_godel(axes_multi) -> int:
    """Assign big-endian mixed-radix strides to the multi-choice axes (already in
    canonical anchor order) and return TOTAL = product of radices = number of
    functionals per arch.

    stride_{n-1} = 1 ; stride_j = product of all less-significant radices.
    """
    strides = [0] * len(axes_multi)
    acc = 1
    for j in reversed(range(len(axes_multi))):
        strides[j] = acc
        acc *= axes_multi[j].radix
    for axis, s in zip(axes_multi, strides):
        axis.godel_stride = s
    return acc


def godel_of(axes_multi, selection) -> int:
    """godel number for a selection (one nth per multi-choice axis, in the same
    order). Requires assign_godel to have run."""
    return sum(nth * axis.godel_stride
               for nth, axis in zip(selection, axes_multi))
