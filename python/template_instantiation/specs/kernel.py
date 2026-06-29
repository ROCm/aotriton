# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The KernelSpec passive record (pipeline Stage 2).

`@ati.describe` / the stacked-@ finalizer (specs/finalize.py) attaches one of these
to a kernel as `kernel.__ati_node__`: the introspected signature plus the partitioned
spec-records (tensors / scalars / overrides / tune / disables / dtype_vars / cites).
It is a passive "object file" — no Axis/Override IR is built here; the linker
(codegen) resolves cites then the builder lowers it to a KernelDescription.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from .node import AtiNode

if TYPE_CHECKING:
    from ..introspect import ParamSpec
    from ..decorators.tensor import TensorSpec
    from ..decorators.scalar import ScalarSpec
    from ..decorators.choicevar import ChoiceVar
    from ..decorators.cite import CiteSpec
    from ..decorators.disable import DisableSpec
    from ..ir.override import Override
    from .tune import TuneSpec


@dataclass(slots=True)
class KernelSpec(AtiNode):
    """The ATI sidecar attached to a kernel as `kernel.__ati_node__`.

    This is the kernel's passive Stage-2 "object file" — the same role that
    OperatorDecl and AffineDecl play for operators and affine kernels. There is no
    separate KernelDecl because KernelSpec must be CLONED AND MUTATED during linking:
    cite resolution (ir/ops/cite.py) appends gap tensors/scalars/overrides onto a
    per-link mutable copy of this record. OperatorDecl / AffineDecl carry no unresolved
    cross-kernel references, so the linker reads them verbatim (no clone needed)."""

    kernel: object
    params: list[ParamSpec]            # signature order
    tensors: list[TensorSpec]
    scalars: list[ScalarSpec]
    overrides: list[Override]
    tune: TuneSpec | None = None       # tune specs, attached later (Phase 3)
    disables: list[DisableSpec] | None = None
    # Named dtype/choice variables declared via @ati.type_var / ati.scalar_var as
    # standalone records (the stacked-@ / string-ref form); tensor/scalar specs
    # refer to them by name. ChoiceVars passed inline by object (the older form)
    # are NOT here — they ride on the spec's dtype.
    dtype_vars: list[ChoiceVar] | None = None
    cites: list[CiteSpec] | None = None    # @ati.cite targets
    # The kernel's own Triton source file (set by @ati.source on the kernel object,
    # copied here in __post_init__). The linker reads it instead of the family file
    # passing source_path to a builder. NOT a constructor arg — derived from kernel.
    source_path: str | None = field(init=False, default=None)

    def __post_init__(self):
        self.disables = self.disables or []
        self.dtype_vars = self.dtype_vars or []
        self.cites = self.cites or []
        from ..decorators.source import KernelStub
        self.source_path = self.kernel.source_path if isinstance(self.kernel, KernelStub) else None

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'KernelSpec({getattr(self.kernel, "__name__", self.kernel)!r}, '
                f'{len(self.tensors)} tensors, {len(self.scalars)} scalars, '
                f'{len(self.overrides)} overrides)')
