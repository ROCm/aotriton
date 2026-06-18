# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Standalone KernelDescription builder (pipeline Stage 4 — LOWER).

`build_kernel_description` is the single-kernel build path used outside the linker
(the per-kernel registry tests + ad-hoc builds): resolve @ati.cite gaps, lower the
spec to a BuiltKernel, wrap it in the KernelDescription IR, and register it so later
kernels can cite it. The whole-family build goes through codegen/linker.py instead.
"""

from .kernel import build_kernel


def build_kernel_description(kernel, *, family, source_path=None,
                             triton_kernel_name=None, register=True):
    """Build an KernelDescription from a kernel already described via
    ati.describe() / the stacked-@ form.

    Before lowering, @ati.cite gaps are filled from kernels built earlier (the flat
    per-family registry). After building, the kdesc registers itself so later
    kernels can cite it. `register=False` skips registration (test isolation)."""
    from ..ir.ops.cite import resolve_cites
    from ..ir.kdesc import KernelDescription
    from ..specs.finalize import get_kernel_spec
    from .. import registry as _registry
    spec = get_kernel_spec(kernel)
    assert spec is not None, (
        f'kernel {getattr(kernel, "__name__", kernel)!r} has no ATI description; '
        f'call ati.describe(...) or use the stacked-@ form first')
    resolve_cites(spec, family=family)        # fill cited gaps before lowering
    built = build_kernel(spec)
    adapter = KernelDescription(built, family=family, source_path=source_path,
                                   triton_kernel_name=triton_kernel_name)
    adapter.kernel_spec = spec      # source KernelSpec (for --sancheck)
    if register:
        _registry.register_kernel(adapter)
    return adapter
