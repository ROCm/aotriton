# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Backwards-compatible shim for the old `describe` module.

The describe() primitive, the stacked-@ finalizer, and the passive Stage-2 spec
models moved into the `specs/` package (specs/finalize.py + specs/{kernel,affine,
operator}.py) during the pipeline-stage reorg. This module re-exports their public
surface so existing `from ...template_instantiation.describe import ...` imports keep
working. New code should import from `specs` directly.
"""

from .specs.kernel import KernelSpec
from .specs.affine import AffineDecl, collect_affine_decl
from .specs.operator import OperatorDecl, collect_operator_decl
from .specs.finalize import (
    describe, start, accumulate_spec, get_kernel_spec,
    _validate_completeness, _partition, _build_tune_spec,
    _finalize_affine, _finalize_operator,
)

# Legacy private aliases (the collectors were _collect_* in this module).
_collect_affine_decl = collect_affine_decl
_collect_operator_decl = collect_operator_decl

__all__ = [
    'KernelSpec', 'AffineDecl', 'OperatorDecl',
    'describe', 'start', 'accumulate_spec', 'get_kernel_spec',
    'collect_affine_decl', 'collect_operator_decl',
]
