# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Advanced Template Instantiation (ATI) — declarative front-end for describing
Triton kernels and operators.

This is the Step 0.2 skeleton: every documented public name resolves so the API
shape is fixed early, but each entry point raises NotImplementedError until its
implementing step lands. See agent-plans/ati_executive0.md.

Authoring surface (see agent-plans/ati_rev1.md):
    tensor_dtype, choice_set   - named choice variables (§3.1, §3.2)
    tensor, scalar             - bind args to a variable / literal type (§3.1, §3.2)
    overrides                  - conditional value rules (§3.3)
    eq, ne, lt, gt             - predicate builders for overrides / metro `if`
    describe                   - attach a spec to a pure-Triton kernel (§3.4 Mode B)
    operator                   - declare an operator over backends (§4)
    union_params               - order-preserving param merge (§4.2)
    metro_kernel               - if/else metro wiring (§5)
    tune                       - perf schema / autotune submodule (§6)
"""

from . import tune
from . import affine          # ati.affine.* decorator namespace (slim affine kernels)
from . import hints           # ati.hints.* author hints (union_precedence, ...)
# The operator-build helpers live in the `.ir.ops` subpackage (ir/ops/cite.py,
# infer.py, union.py) — under ir/ since they operate on the IR, and NOT named
# `operator`, so it never shadows the @ati.operator DECORATOR exported below.
from .ir.ops import union_params
from .describe import describe, kernel
from .builder import DescriptionError
from .metro import metro_kernel
from .decorators import (
    tensor_dtype, choice_set, tensor, scalar,
    derives, overrides, disable, cite, source, eq, ne, lt, gt,
    operator, backend,
)

__all__ = [
    'affine', 'hints',
    'tensor_dtype', 'choice_set', 'tensor', 'scalar',
    'derives', 'overrides', 'disable', 'cite', 'source', 'eq', 'ne', 'lt', 'gt',
    'describe', 'kernel', 'operator', 'backend', 'union_params', 'metro_kernel',
    'tune', 'DescriptionError',
]


def _stub(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f'ati.{name} is not implemented yet (Step 0.2 skeleton). '
            f'See agent-plans/ati_executive0.md.')
    _raise.__name__ = name
    _raise.__qualname__ = name
    return _raise


# --- choice-declaring surface (§3.1, §3.2): implemented in Step 2.1 ---
# tensor_dtype, choice_set, tensor, scalar imported from .decorators
# --- conditional overrides + predicate builders (§3.3): implemented in Step 2.2 ---
# overrides, eq, ne, lt, gt imported from .decorators

# --- description attachment (§3.4): describe + kernel imported from .describe ---
# --- operator + metro (§4, §5): operator/backend (§4), union_params (5.1),
#     metro_kernel (5.5) imported from .decorators / .operator / .metro ---
