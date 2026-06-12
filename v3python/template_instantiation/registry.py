# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flat per-family kernel registry for @ati.cite resolution (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 4).

@ati.cite("<op>.<metro>.<kernel>") refers to another kernel by STRING to avoid
circular imports between sibling kernel modules. A string defers resolution to
build time; this registry is what the resolver looks the cited kernel up in.

Kernel-level citing (Step 4) keys on the LAST segment (the Triton kernel name).
The registry is FLAT and per-family: each AtiKernelDescription registers itself as
it is built (`build_kernel_description`), so a kernel can only cite one built
earlier in module import order — which is the natural key-kernel-before-aux-kernel
ordering. Metro/operator-level citing (a later step) will resolve the
`<op>.<metro>` prefix through the operator registry instead.
"""

# family -> { kernel NAME -> AtiKernelDescription }
_KERNELS: dict[str, dict] = {}


def register_kernel(kdesc):
    """Record a built AtiKernelDescription so later kernels can @ati.cite it."""
    _KERNELS.setdefault(kdesc.FAMILY, {})[kdesc.NAME] = kdesc


def get_kernel(family: str, kernel_name: str):
    """The registered kdesc for (family, kernel_name), or None if not yet built."""
    return _KERNELS.get(family, {}).get(kernel_name)


def clear(family: str | None = None):
    """Drop registrations (test isolation). All families when family is None."""
    if family is None:
        _KERNELS.clear()
    else:
        _KERNELS.pop(family, None)
