# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ChoiceView — the interface for reading one instantiation's pinned choices.

`ChoiceView` was one concrete class (`ir/functional.py`), constructed from a
`Functional` and reading its `choice`/`resolved` tables. That works for the
generator, which always has a linked `Functional` in hand. It does not work
for a build-time driver, which has only the plain `{name: literal}` dict
parsed out of a signature string: there is no IR behind it, and no `Functional`
to build one from.

The tempting shortcut is to let such a driver subscript the bare dict and be
done. That gives two spellings for "read this kernel's pinned choices" -- a
mapping on one side and an object on the other -- so any code that wants to
work with both (a kernel description that must be readable by the generator
*and* by a driver) has to branch on which it was handed. Make the shared
surface explicit instead: `ChoiceView` is an ABC, and each side supplies its
own backing.

`FunctionalChoiceView` (`ir/functional.py`) is the `Functional`-backed one, and
is what `Functional.choices` returns. A mapping-backed implementation is
declared by whoever parses the wire text, next to the parsing.

**The interface is `arg(aname)` + attribute access, and nothing else.**
`tc`/`arg_tc` -- which hand back the raw `TypedChoice` -- are deliberately NOT
on the ABC. A `TypedChoice` is a `Functional`-side object; a parsed dict never
carried one, because only the literal survives the wire format. Putting them on
the interface would force a mapping-backed implementation to declare two
methods whose only possible body is a raise, which states "this view has these
operations" and then denies it. They live on `FunctionalChoiceView` alone, so a
caller that needs one is asking for a Functional and finds that out from the
type.
"""

from abc import ABC, abstractmethod


class ChoiceVarAbsent(AttributeError):
    """Raised by a ChoiceView when a predicate reads a choice variable (or, for
    a mapping-backed view, a key) it does not have. Subclasses AttributeError
    so getattr/hasattr duck-typing still behaves, but
    KernelDescription.is_functional_disabled catches it specifically to emit a
    write-your-own-@ati.disable diagnostic (a cited disable predicate that reads
    a variable absent from the citing kernel)."""


class ChoiceView(ABC):
    """Ergonomic accessor over one instantiation's pinned choices.

    Attribute access is keyed by *choice-variable name*: `choices.T_io` returns
    the variable's signature. `.arg(aname)` reads a resolved argument by its
    real (kernel-signature) name -- the form to use for an operand whose axis
    variable is named differently, e.g. `choices.arg('Q')` rather than
    `choices.T_io` (see `ir/axis.py`'s `Axis.signature_name` for why the two
    can differ)."""

    # A slotted class is dict-free only if EVERY base is slotted, so an ABC
    # without this silently re-adds `__dict__` to `FunctionalChoiceView` and
    # undoes its `__slots__`. One view is cached per Functional and functionals
    # are enumerated in bulk, so that is a per-instantiation dict on the widest
    # object in the generator.
    __slots__ = ()

    @abstractmethod
    def arg(self, aname):
        """The resolved (post-override) signature for a real argument name."""

    @abstractmethod
    def __getattr__(self, var):
        """The signature for a choice variable, by attribute access."""
