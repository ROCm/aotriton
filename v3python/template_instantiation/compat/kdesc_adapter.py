# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelDescription adapter over the ATI IR (executive plan Step 4.1).

Goal: drive the EXISTING code generator from the new Axis/Override/Functional IR,
without going through the legacy TYPE_CHOICES / FEAT_CHOICES / PERF_CHOICES /
ARGUMENTS dictionaries. Those dicts exist only to power the old conditional
type-inference and gen_functionals; the new enumeration (enumerate_functionals)
replaces both, so the adapter sources everything the generator reads directly
from the BuiltKernel.

Per ati+newbinds_rev1.md §6.2 the params-struct field type comes from the axis
(the ABI), and overrides change only per-functional values, never the struct.

This step (4.1) provides the enumeration + identity surface and proves
count/godel parity with the legacy description. The remaining codegen-facing
surface (func_cfields, translate_dataframe, the Functional signature helpers) is
filled in Step 4.2 while running the real generator, where the exact requirements
surface.
"""

from ..builder import build_kernel
from ..describe import get_kernel_spec
from ..ir import assign_godel, enumerate_functionals


class AtiFunctional:
    """A functional produced by the adapter. Wraps an IR Functional and carries
    the identity the generator keys on. Codegen-facing signature/name helpers are
    added in Step 4.2."""

    __slots__ = ('_ir', '_kdesc', '_optimized_for')

    def __init__(self, ir_functional, kdesc, optimized_for):
        self._ir = ir_functional
        self._kdesc = kdesc
        self._optimized_for = list(optimized_for)

    @property
    def meta_object(self):
        return self._kdesc

    @property
    def arch(self):
        return self._ir.arch

    @property
    def arch_number(self):
        return self._ir.arch_number

    @property
    def godel_number(self):
        return self._ir.godel_number

    @property
    def choices(self):
        return self._ir.choices

    @property
    def resolved(self):
        return self._ir.resolved

    @property
    def optimized_for(self):
        return self._optimized_for

    @property
    def noptimized_for(self):
        return len(self._optimized_for)

    @property
    def family(self):
        return self._kdesc.FAMILY

    @property
    def name(self):
        return self._kdesc.NAME

    def __repr__(self):
        return (f'AtiFunctional({self._kdesc.NAME!r}, arch={self.arch!r}, '
                f'godel={self.godel_number})')


class AtiKernelDescription:
    """KernelDescription-compatible facade backed by a BuiltKernel."""

    CODEGEN_MODULE = 'triton'
    TUNE_NAME = 'autotune'
    FILE_PFX = 'shim'
    SHARED_IFACE = None

    def __init__(self, built, *, family, source_path=None, triton_kernel_name=None):
        self._built = built
        self.NAME = built.name
        self.FAMILY = family
        self._source_path = source_path
        self._triton_kernel_name = triton_kernel_name or built.name
        # Canonical (anchor-ordered) axes; assign godel strides to multi-choice.
        self._axes_all = sorted(built.axes, key=lambda a: a.anchor)
        self._axes_multi = [a for a in self._axes_all if not a.is_trivial]
        assign_godel(self._axes_multi)
        self._godel_number = 1
        for a in self._axes_multi:
            self._godel_number *= a.radix

    # --- identity ---

    @property
    def ARGUMENTS(self):
        return self._built.arguments

    @property
    def godel_number(self):
        return self._godel_number

    @property
    def unique_path(self):
        from pathlib import Path
        return Path(self.FAMILY) / self.CODEGEN_MODULE / self.NAME

    # --- enumeration (replaces legacy gen_functionals) ---

    def gen_functionals(self, target_arch):
        for ir_f in enumerate_functionals(self._built.axes, self._built.overrides,
                                          target_arch):
            yield AtiFunctional(ir_f, self,
                                optimized_for=target_arch[ir_f.arch])

    # --- tuning passthrough ---

    @property
    def tune(self):
        return self._built.tune

    @property
    def is_tunable(self):
        return self._built.tune is not None and self._built.tune.is_tunable

    def is_functional_disabled(self, functional):
        return False


def build_kernel_description(kernel, *, family, source_path=None,
                             triton_kernel_name=None):
    """Build an AtiKernelDescription from a kernel already described via
    ati.describe() / the stacked-@ form."""
    spec = get_kernel_spec(kernel)
    assert spec is not None, (
        f'kernel {getattr(kernel, "__name__", kernel)!r} has no ATI description; '
        f'call ati.describe(...) or use the stacked-@ form first')
    built = build_kernel(spec)
    return AtiKernelDescription(built, family=family, source_path=source_path,
                                triton_kernel_name=triton_kernel_name)
