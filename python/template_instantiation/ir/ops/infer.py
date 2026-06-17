# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
SHARED_IFACE inference (executive plan Step 5.2).

A triton kernel never declares which param struct it participates in — that is the
operator → metro → kernel relationship, inferred here. The operator owns the param
struct (built from its DEFAULT backend, the feature superset); each triton
sub-kernel collaborating in that default backend's metro *borrows* it. The codegen
shim reads `param_class_name` / `FAMILY` / `CALL_OPTIONS_NAME` off the inferred
SHARED_IFACE.

This replaces the Step-4 scaffolding that hand-set `kernel.SHARED_IFACE` on the
ATI adapter; it is NOT part of the `@ati.*` surface.
"""


def _iter_subkernels(node):
    """Yield the concrete sub-kernels of a metro node, descending into
    ConditionalKernel branches."""
    # MetroKernel: a list of steps
    if hasattr(node, 'list_kernels'):
        for step in node.list_kernels():
            yield from _iter_subkernels(step)
        return
    # ConditionalKernel: if/else branches
    if hasattr(node, 'if_kernel'):
        yield from _iter_subkernels(node.if_kernel)
        if node.else_kernel is not None:
            yield from _iter_subkernels(node.else_kernel)
        return
    # a concrete kernel description
    yield node


def infer_shared_iface(operators):
    """Set SHARED_IFACE on every triton kernel that borrows an operator's param
    struct, walking operator -> default-backend metro -> sub-kernels.

    Only sets it on kernels that don't already have it (legacy kernels declare
    SHARED_IFACE on the class; ATI adapter kernels leave it None until here). The
    DEFAULT backend (backends[0]) is the param-struct owner; alternative backends
    are interchangeable and not the struct source.
    """
    for op in operators:
        backends = op.list_backends()
        if not backends:
            continue
        # Every backend's concrete kernels borrow this operator's param struct: the
        # default backend's metro sub-kernels AND any alternative backend that is a
        # kernel in its own right (e.g. the fused bwd_kernel_fuse). The default
        # backend is the struct OWNER, but all of them must reference it.
        for backend in backends:
            for sub in _iter_subkernels(backend):
                # Only fill in kernels whose SHARED_IFACE is unset (ATI adapter);
                # never override a legacy kernel's declared class attribute. Affine
                # backends have their own surface and are skipped (no SHARED_IFACE
                # attribute to set meaningfully).
                if hasattr(sub, 'SHARED_IFACE') and getattr(sub, 'SHARED_IFACE', None) is None:
                    sub.SHARED_IFACE = op
