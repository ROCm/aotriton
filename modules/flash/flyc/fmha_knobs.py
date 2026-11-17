# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Knob base classes shared by every flyc attention kernel.

flydsl-free on purpose: the code generator imports the knobs classes (through
the tuning modules) to record them, and it runs with no flydsl installed.

`FlycKnobs` is the must-have set. It is what ATI and the C++ shim read back
from the recorded knob dict, whatever the arch: where the kernel lives, which
`@flyc.kernel` def it is, how the launch grid is ordered, and the Philox width.
"""

from dataclasses import dataclass, fields, replace
from typing import ClassVar

# Which quantity lands on which grid axis, the `GRID_AXIS_ORDER` knob:
#
#   0  HEAD_FASTEST  grid = (head, tile_blocks, batch_or_seq)
#   1  TILE_FASTEST  grid = (tile_blocks, head, batch_or_seq)
#
# "head" is the q head for the forward and dQ, the **kv** head for dK/dV (the
# GQA fold made it so). AOTriton dispatches the hsaco directly and computes the
# grid in C++, never running the `@flyc.jit` launcher, so this decision has to
# travel in the knob set rather than as an `if (arch == ...)` on the C++ side.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1

# Triton's Philox-4x32-10. The backward must reproduce the forward's mask, and
# both must match Triton's stream, so this is not a per-arch choice.
PHILOX_WIDTH = 32


@dataclass(frozen=True)
class FlycKnobs:
    """The knobs every flyc kernel records. `None` means "policy decides".

    The concrete classes name their kernel through the three `FLYC_*` class
    constants, and `resolve` stamps them into the fields with
    `with_flyc_fields()`. A caller never sets these four: they are properties of
    the kernel, not choices.
    """

    # The kernel file relative to `modules/<family>/flyc/`, and the
    # `@flyc.kernel` def's own name inside it.
    flyc_source: str | None = None
    flyc_kernel_name: str | None = None
    # Upper case against the convention here: the field name is the wire key the
    # C++ side reads back as `perf().get_int("GRID_AXIS_ORDER")`.
    GRID_AXIS_ORDER: int | None = None
    philox_width: int | None = None

    FLYC_SOURCE: ClassVar[str]
    FLYC_KERNEL_NAME: ClassVar[str]
    FLYC_GRID_AXIS_ORDER: ClassVar[int]

    def merge(self, other):
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)

    def with_flyc_fields(self):
        """Fill the four must-have fields from the class's own constants."""
        return replace(self,
                       flyc_source=self.FLYC_SOURCE,
                       flyc_kernel_name=self.FLYC_KERNEL_NAME,
                       GRID_AXIS_ORDER=self.FLYC_GRID_AXIS_ORDER,
                       philox_width=PHILOX_WIDTH)


@dataclass(frozen=True)
class FlycQKnobs(FlycKnobs):
    """Kernels whose grid walks Q tiles: the forward and dQ."""

    block_m: int | None = None


@dataclass(frozen=True)
class FlycKVKnobs(FlycKnobs):
    """Kernels whose grid walks KV tiles: dK/dV."""

    block_kv: int | None = None


def pinned(cls, overrides, **pins):
    """A `cls` from `overrides` (a dict, a `cls` or None) with `pins` on top."""
    if isinstance(overrides, dict):
        overrides = cls(**overrides)
    return (overrides or cls()).merge(cls(**pins))
