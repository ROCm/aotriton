# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI builder package (pipeline Stage 4 — LOWER).

Lowers the passive Stage-2 specs into the IR the code generators consume:
- kernel.py:   build_kernel + BuiltKernel (KernelSpec -> axes/overrides IR).
- metro.py:    lower_plan / build_metro (MetroPlan -> MetroKernel IR).
- errors.py:   DescriptionError (the front-end diagnostic).

build_merged_struct_cfields lives in ir/ops/union.py (it merges IR-level func_cfields
via union_params, not a lowering step) and is re-exported here for back-compat.
"""

from .errors import DescriptionError
from .kernel import build_kernel, BuiltKernel, _is_ati_type_string
from ..ir.ops.union import build_merged_struct_cfields
from .metro import lower_plan, build_metro

__all__ = [
    'DescriptionError', 'build_kernel', 'BuiltKernel',
    'build_merged_struct_cfields',
    'lower_plan', 'build_metro',
]
