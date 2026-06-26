# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test-only cite-resolution registry and single-kernel builder.

This module must NOT be part of the aotriton package — it exists solely to
support test fixtures that build one kernel at a time and register it so
subsequent cite-resolution in the same test can find it.

Production cite resolution is handled by the linker (codegen/linker.py),
which constructs its own lookup/metro_lookup closures over its local compiled-
family dicts and never touches this module.

InterfaceRegistry
-----------------
Named after the Interface ABC: every ATI description that can be a @ati.cite
target — KernelDescription, MetroKernel, Operator — is an Interface subclass.
This registry holds all of them keyed by family and name rather than having
separate per-kind registries. One instance per test (or fixture scope); the
caller holds all registry objects and constructs the appropriate lookup
closures before calling resolve_cites.
"""

from __future__ import annotations


class InterfaceRegistry:
    """Per-family cite-resolution registry for test fixtures."""

    def __init__(self):
        self._kernels: dict[str, dict] = {}   # family -> {name -> KernelDescription}
        self._ops:     dict[str, dict] = {}   # family -> {name -> Operator}

    # --- kernel table ---

    def register_kernel(self, kdesc) -> None:
        self._kernels.setdefault(kdesc.FAMILY, {})[kdesc.NAME] = kdesc

    def get_kernel(self, family: str, kernel_name: str):
        return self._kernels.get(family, {}).get(kernel_name)

    # --- operator table ---

    def register_op(self, op) -> None:
        self._ops.setdefault(op.FAMILY, {})[op.NAME] = op

    def get_op(self, family: str, op_name: str):
        return self._ops.get(family, {}).get(op_name)

    # --- isolation ---

    def clear(self, family: str | None = None) -> None:
        if family is None:
            self._kernels.clear()
            self._ops.clear()
        else:
            self._kernels.pop(family, None)
            self._ops.pop(family, None)


def _testonly_build_kernel_description(kernel, *, family, registry: InterfaceRegistry,
                                       source_path=None, triton_kernel_name=None,
                                       register=True):
    """Build a KernelDescription for a single kernel and optionally register it.

    Test-only. The production path goes through codegen/linker.py which builds
    the whole family in topological order with its own lookup closures.

    The caller must supply an InterfaceRegistry and is responsible for constructing
    any cross-family lookup closures needed by resolve_cites.
    """
    from aotriton.template_instantiation.builder import build_kernel
    from aotriton.template_instantiation.ir.kdesc import KernelDescription
    from aotriton.template_instantiation.specs.finalize import get_kernel_spec
    from aotriton.template_instantiation.ir.ops.cite import resolve_cites

    spec = get_kernel_spec(kernel)
    assert spec is not None, (
        f'kernel {getattr(kernel, "__name__", kernel)!r} has no ATI description')

    def lookup(fam, kernel_name):
        return registry.get_kernel(fam, kernel_name)

    def op_lookup(fam, op_name):
        return registry.get_op(fam, op_name)

    resolve_cites(spec, family=family, lookup=lookup, op_lookup=op_lookup)
    built = build_kernel(spec)
    adapter = KernelDescription(built, family=family, source_path=source_path,
                                triton_kernel_name=triton_kernel_name)
    adapter.kernel_spec = spec
    if register:
        registry.register_kernel(adapter)
    return adapter
