# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Advanced Template Instantiation (ATI) — declarative front-end for describing
Triton kernels and operators.

Authoring surface:
    type_var, scalar_var       - named choice variables (§3.1, §3.2)
    tensor, scalar             - bind args to a variable / literal type (§3.1, §3.2)
    overrides                  - conditional value rules (§3.3)
    eq, ne, lt, gt             - predicate builders for overrides / metro `if`
    describe                   - attach a spec to a pure-Triton kernel (§3.4 Mode B)
    start                      - terminal stacked-@ decorator finalizing a kernel/
                                 operator/affine description (§3.4 Mode A)
    operator                   - declare an operator over backends (§4)
    union_params               - order-preserving param merge (§4.2)
    metro_kernel               - if/else metro wiring (§5)
    tune                       - perf schema / autotune submodule (§6)
"""

from .tune_facade import tune    # ati.tune.* namespace (decorators/tune + specs/tune)
from .decorators import affine   # ati.affine.* decorator namespace (slim affine kernels)
from .decorators import hints    # ati.hints.* author hints (union_precedence, ...)
# The operator-build helpers live in the `.ir.ops` subpackage (ir/ops/cite.py,
# infer.py, union.py) — under ir/ since they operate on the IR, and NOT named
# `operator`, so it never shadows the @ati.operator DECORATOR exported below.
from .ir.ops import union_params
# Import via the `.describe` shim (not `.specs.finalize` directly) so the `describe`
# submodule loads here FIRST — its package attribute is then overwritten by the
# `describe` FUNCTION binding below, so `ati.describe` stays the callable even after
# a later `from ...template_instantiation.describe import ...` re-touches the module.
from .describe import describe, start
from .builder import DescriptionError
from .decorators import metro_kernel
from .decorators import (
    type_var, scalar_var, tensor, scalar,
    derives, overrides, disable, no_disable, cite, source, eq, ne, lt, gt,
    operator, backend,
)

__all__ = [
    'affine', 'hints',
    'type_var', 'scalar_var', 'tensor', 'scalar',
    'derives', 'overrides', 'disable', 'no_disable', 'cite', 'source', 'eq', 'ne', 'lt', 'gt',
    'describe', 'start', 'operator', 'backend', 'union_params', 'metro_kernel',
    'tune', 'DescriptionError',
]


def _stub(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f'ati.{name} is not yet implemented.')
    _raise.__name__ = name
    _raise.__qualname__ = name
    return _raise


# type_var, scalar_var, tensor, scalar, overrides, eq/ne/lt/gt from .decorators
# describe + start from .describe; operator/backend, union_params, metro_kernel
# from .decorators
