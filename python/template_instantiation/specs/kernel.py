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


class KernelSpec:
    """The ATI sidecar attached to a kernel as `kernel.__ati__`."""

    __slots__ = ('kernel', 'params', 'tensors', 'scalars', 'overrides', 'tune',
                 'disables', 'dtype_vars', 'cites', 'source_path')

    def __init__(self, kernel, params, tensors, scalars, overrides, tune=None,
                 disables=None, dtype_vars=None, cites=None):
        self.kernel = kernel
        self.params = params           # list[ParamSpec], signature order
        self.tensors = tensors         # list[TensorSpec]
        self.scalars = scalars         # list[ScalarSpec]
        self.overrides = overrides     # list[Override]
        self.tune = tune               # tune specs, attached later (Phase 3)
        self.disables = disables or []  # list[DisableSpec]
        # The kernel's own Triton source file (set by @ati.source on the kernel
        # object, copied here in describe()). The linker reads it instead of the
        # family file passing source_path to a builder.
        self.source_path = getattr(kernel, '__ati_source_path__', None)
        # Named dtype/choice variables declared via @ati.type_var /
        # ati.scalar_var as standalone records (the stacked-@ / string-ref form);
        # tensor/scalar specs refer to them by name. ChoiceVars passed inline by
        # object (the older form) are NOT here — they ride on the spec's dtype.
        self.dtype_vars = dtype_vars or []   # list[ChoiceVar]
        self.cites = cites or []             # list[CiteSpec] (@ati.cite targets)

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'KernelSpec({getattr(self.kernel, "__name__", self.kernel)!r}, '
                f'{len(self.tensors)} tensors, {len(self.scalars)} scalars, '
                f'{len(self.overrides)} overrides)')
