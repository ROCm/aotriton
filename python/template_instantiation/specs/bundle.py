# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
SpecBundle: the common partition every Stage-2 collector shares.

Every stacked-`@ati.*` block (triton/describe(), affine, operator, metro) is, at
bottom, a list of spec-records that must be sorted by kind before a
`*Decl` can be built from them. `partition(specs)` does the part of that sort
common to every stack -- tensors/scalars/overrides/dtype_vars/tune records, plus
the singular `disable` -- and parks anything it does not recognise in
`bundle.unrecognized`. It does not raise on an unclaimed spec: this layer has
no way to know whether a given stack additionally accepts it (a stack marker
like AffineKernelSpec/OperatorSpec, or a stack-specific record like an
`@ati.affine.directories`). Only the caller -- `partition_kernel` /
`partition_affine` / `partition_operator` (specs/finalize.py +
specs/{affine,operator}.py) -- knows its stack's full accepted vocabulary, so
only it may call `forbid()` (restrict a common kind this stack does not accept)
and `reject_remaining()` (raise on whatever is still unclaimed once its own
additional kinds are pulled out of `unrecognized`).

`cites` stays a plain list, in source order (`partition` must not reorder it):
stacking several `@ati.cite` on one kernel is a deliberate pattern ("citation
mode (b)") for pulling merged operand vocabulary out of several sub-kernels --
see `modules/flash/aot/bwd_kernel_fuse.py`. `resolve_cites`
(ir/ops/cite.py) resolves a contested operand first-wins over `spec.cites` in
order, so the OUTERMOST (topmost in source) `@ati.cite` wins; reordering here
would silently change which cited kernel's practices apply.

`disable` is singular, not a one-element list: a description may write at most
one, and this is the earliest point that rule can be enforced -- the specs are
in hand and nothing has been built yet. This is DECLARED cardinality, and is
distinct from RESOLVED cardinality (a whole-metro cite can expand one declared
disable into several inherited ones) -- see specs/node.py's
`resolved_disables`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..decorators.tensor import TensorSpec
    from ..decorators.scalar import ScalarSpec
    from ..decorators.choicevar import ChoiceVar
    from ..decorators.cite import CiteSpec
    from ..decorators.disable import DisableSpec
    from ..ir.override import Override


@dataclass
class SpecBundle:
    """The common spec vocabulary, partitioned by kind. `forbid()` and
    `reject_remaining()` are how a specialised partition states its stack's
    restrictions explicitly instead of leaving them as an absent `elif`."""

    tensors: list[TensorSpec] = field(default_factory=list)
    scalars: list[ScalarSpec] = field(default_factory=list)
    overrides: list[Override] = field(default_factory=list)
    dtype_vars: list[ChoiceVar] = field(default_factory=list)
    tune_records: list = field(default_factory=list)
    cites: list[CiteSpec] = field(default_factory=list)
    disable: 'DisableSpec | None' = None      # AT MOST ONE (declared cardinality)
    unrecognized: list = field(default_factory=list)

    def forbid(self, *field_names, what):
        """Raise, naming `what` (the stack, e.g. '@ati.affine') and the
        offending field(s), if any of `field_names` is non-empty/non-None.
        Called by a specialised partition for every common kind its stack does
        NOT accept, so the restriction is a line that has to be deleted to
        change the rule rather than the absence of one."""
        offenders = [n for n in field_names if getattr(self, n)]
        if offenders:
            raise AssertionError(
                f'{what} does not accept: {", ".join(offenders)}')

    def reject_remaining(self, what):
        """Raise if anything is still unclaimed in `unrecognized` -- called
        last by a specialised partition, once it has pulled its own additional
        kinds out of `unrecognized`, since only it knows its stack's full
        accepted vocabulary."""
        if self.unrecognized:
            raise AssertionError(
                f'unexpected spec(s) in an {what} stack: {self.unrecognized!r}')


def partition(specs) -> SpecBundle:
    """Claim every COMMON spec kind out of `specs`, in order. Anything
    unclaimed lands in `bundle.unrecognized` -- not an error here; see the
    module docstring."""
    from ..decorators import TensorSpec, ScalarSpec, ChoiceVar, DisableSpec, CiteSpec
    from ..ir import Override
    from .tune import PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec
    tune_types = (PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec)

    b = SpecBundle()
    for s in specs:
        if isinstance(s, TensorSpec):
            b.tensors.append(s)
        elif isinstance(s, ScalarSpec):
            b.scalars.append(s)
        elif isinstance(s, ChoiceVar):
            b.dtype_vars.append(s)
        elif isinstance(s, CiteSpec):
            b.cites.append(s)          # plural, source order preserved
        elif isinstance(s, Override):
            b.overrides.append(s)
        elif isinstance(s, DisableSpec):
            assert b.disable is None, 'multiple @ati.disable on one stack'
            b.disable = s
        elif isinstance(s, tune_types):
            b.tune_records.append(s)
        else:
            b.unrecognized.append(s)
    return b
