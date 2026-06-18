# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Named choice variables: `ati.type_var` / `ati.scalar_var`.

A ChoiceVar is the `template<typename T>` / typing.TypeVar analogue — one choice set
shared by several arguments (the element-type variable a group of tensors binds, or a
scalar choice variable). See the package docstring for the authoring surface.
"""

from ..specs.base import StackedSpec


class ChoiceVar(StackedSpec):
    """A named choice variable shared by several arguments — the
    `template<typename T>` / typing.TypeVar analogue. `type_var` and `scalar_var`
    both produce one; `kind` records which authoring word created it (purely for
    diagnostics — TypedChoice.parse handles tensor vs scalar literals uniformly).

    `signature_name` is the argument under which this variable is recorded in all
    PERSISTED forms — the compact signature string, which becomes the aks2 / zip
    entry name and the tuning-database row key. A multi-choice variable spanning
    several arguments MUST give it explicitly (it cannot be silently derived from
    spec order, or stored artifacts would shift); single-choice variables are
    trivial and exempt. `None` when not given — the builder resolves it."""

    __slots__ = ('name', 'choices', 'kind', 'signature_name')

    def __init__(self, name, choices, kind, signature_name=None):
        assert isinstance(name, str) and name, 'choice variable needs a name'
        self.name = name
        self.choices = list(choices)
        self.kind = kind          # 'type' | 'scalar'
        self.signature_name = signature_name

    def __repr__(self):
        return (f'ChoiceVar({self.name!r}, kind={self.kind}, '
                f'signature_name={self.signature_name!r}, choices={self.choices})')


def type_var(name, dtype, signature_name=None):
    """Declare a named tensor element-type variable (the typing.TypeVar analogue).
    `dtype` is the choice set. `signature_name` is the argument that records this
    variable in persisted artifacts (compact signature, aks2/zip entry name, DB row
    key); required for multi-choice variables shared across several tensors."""
    return ChoiceVar(name, dtype, kind='type', signature_name=signature_name)


def scalar_var(name, options, signature_name=None):
    """Declare a named scalar choice variable. `options` is the choice set.
    `signature_name` is the argument that records this variable in persisted
    artifacts (compact signature, aks2/zip entry name, DB row key)."""
    return ChoiceVar(name, options, kind='scalar', signature_name=signature_name)
