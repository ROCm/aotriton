# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The stacked-@ spec-record base (pipeline Stage 1->2).

Almost every @ati.* decorator is a passive spec record that doubles as a stacked-@
decorator: written as `@ati.tensor(...)` above the kernel, calling it accumulates
the record onto the def's pending list (the stacked-@ Mode A); passed to
ati.describe(...), it is consumed as plain data (Mode B). The accumulate-on-call
behavior is identical for all of them, so it lives here once.

The `accumulate_spec` import is function-scoped on purpose: specs/finalize.py imports
the decorator classes at module top, so a top-level import here would close a cycle.
"""


class StackedSpec:
    """Base for passive @ati.* spec records usable as stacked-@ decorators.

    Calling the record on the def below it (`spec(kernel)`) appends it to the def's
    pending spec list and returns the def, so decorator stacking composes. Subclasses
    that must inspect/mutate state against the decorated def first (e.g. defaulting a
    name to `def.__name__`) override `__call__` and delegate via `super().__call__`."""

    __slots__ = ()

    def __call__(self, target):
        from .finalize import accumulate_spec
        return accumulate_spec(self, target)
