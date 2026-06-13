# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Per-family registries for @ati.cite resolution (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Steps 4 & 8).

@ati.cite("<op>.<metro>[.<kernel>]") refers to another kernel/metro by STRING to
avoid circular imports between sibling kernel modules. A string defers resolution
to build time; these registries are what the resolver looks the target up in.

Two registries, both per-family:
  * KERNELS (Step 4) — flat {kernel NAME -> KernelDescription}, populated as each
    kdesc is built. Kernel-level citing keys on the LAST cite segment (the Triton
    kernel name). A kernel can only cite one built earlier in module import order —
    the natural key-kernel-before-aux-kernel ordering.
  * OPS (Step 8) — {operator NAME -> operator IR}, populated as operators are built
    (after their backends). Metro/operator-level citing resolves the `<op>.<metro>`
    prefix through `ops[op].get_backend(metro)` and the optional `.kernel` suffix
    through `metro.get_kernel(kernel)`.
"""

# family -> { kernel NAME -> KernelDescription }
_KERNELS: dict[str, dict] = {}
# family -> { operator NAME -> operator IR }
_OPS: dict[str, dict] = {}


def register_kernel(kdesc):
    """Record a built KernelDescription so later kernels can @ati.cite it."""
    _KERNELS.setdefault(kdesc.FAMILY, {})[kdesc.NAME] = kdesc


def get_kernel(family: str, kernel_name: str):
    """The registered kdesc for (family, kernel_name), or None if not yet built."""
    return _KERNELS.get(family, {}).get(kernel_name)


def register_op(op):
    """Record a built operator so kernels can @ati.cite its metros/sub-kernels."""
    _OPS.setdefault(op.FAMILY, {})[op.NAME] = op


def get_op(family: str, op_name: str):
    """The registered operator for (family, op_name), or None if not yet built."""
    return _OPS.get(family, {}).get(op_name)


def clear(family: str | None = None):
    """Drop registrations (test isolation). All families when family is None."""
    if family is None:
        _KERNELS.clear()
        _OPS.clear()
    else:
        _KERNELS.pop(family, None)
        _OPS.pop(family, None)
