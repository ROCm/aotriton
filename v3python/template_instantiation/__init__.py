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
from .decorators import (
    tensor_dtype, choice_set, tensor, scalar,
    derives, overrides, disable, eq, ne, lt, gt,
)
from .describe import describe, kernel
from .builder import AtiDescriptionError

__all__ = [
    'tensor_dtype', 'choice_set', 'tensor', 'scalar',
    'derives', 'overrides', 'disable', 'eq', 'ne', 'lt', 'gt',
    'describe', 'kernel', 'operator', 'union_params', 'metro_kernel',
    'tune', 'AtiDescriptionError',
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
# --- operator + metro (§4, §5): not yet implemented ---
operator = _stub('operator')
union_params = _stub('union_params')
metro_kernel = _stub('metro_kernel')
