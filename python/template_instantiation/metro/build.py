# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI metro-kernel builder.

A metro is a LAUNCHER: it sequences its sub-kernels' contexts to implement one
operator functional (e.g. attn_fwd then the debug kernel; or preprocess + dk_dv +
dq). It owns no params struct or functional space of its own — the operator does.

`AtiMetroKernel` is the ATI form of `MetroKernel`: it bypasses the heavy
`Interface.__init__` / `_late_init` (which would build a functional space the metro
never uses) and keeps only the launcher surface the operator codegen reads
(`enum_name`, `list_kernels`, `iter_subkernels`, `get_kernel`,
`iter_kernel_slot_names`). `build_metro` lowers a transpiled @ati.metro_kernel plan
straight to one.

NOTE: this module imports `aotriton.op` (the description layer), so it is imported
ONLY where a metro is actually built (modules/flash/aot), NOT from metro/__init__ —
keeping `aotriton.op` off the base `template_instantiation` import path.
"""

from aotriton.op import MetroKernel, ConditionalKernel
from .transpile import lower_plan


class AtiMetroKernel(MetroKernel):
    """A metro launcher built from a transpiled @ati.metro_kernel plan. Skips
    MetroKernel.__init__ (the Interface functional machinery) — a metro has no
    functional space; it only sequences sub-kernel contexts."""

    def __init__(self, name, kernels, *, family='flash'):
        self.NAME = name
        self.FAMILY = family
        self._kernels = list(kernels)


def build_metro(plan, kernel_map, name, *, family='flash'):
    """Lower a transpiled MetroPlan to an AtiMetroKernel (a MetroKernel launcher).

    plan:        the @ati.metro_kernel transpiler output (fn.__ati_metro__).
    kernel_map:  {sub-kernel name -> built kdesc}.
    name:        the metro/backend NAME (-> kMetro_<Name> enum).
    """
    return lower_plan(plan, kernel_map,
                      lambda steps: AtiMetroKernel(name, steps, family=family),
                      ConditionalKernel)
