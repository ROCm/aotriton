# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The KernelSpec passive record (pipeline Stage 2).

`@ati.describe` / the stacked-@ finalizer (specs/finalize.py) attaches one of these
to a kernel as `kernel.__ati__`: the introspected signature plus the partitioned
spec-records (tensors / scalars / overrides / tune / disables / dtype_vars / cites).
It is a passive "object file" — no Axis/Override IR is built here; the linker
(codegen) resolves cites then the builder lowers it to a KernelDescription.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
class KernelSpec:
    """The ATI sidecar attached to a kernel as `kernel.__ati__`."""

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
        self.source_path = getattr(self.kernel, '__ati_source_path__', None)

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'KernelSpec({getattr(self.kernel, "__name__", self.kernel)!r}, '
                f'{len(self.tensors)} tensors, {len(self.scalars)} scalars, '
                f'{len(self.overrides)} overrides)')
