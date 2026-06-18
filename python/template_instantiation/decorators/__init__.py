# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI choice-declaring decorators (executive plan Step 2.1; agent-plans/ati_rev1.md
§3.1–§3.2).

These produce *spec records* describing how an argument is instantiated; they are
not yet attached to a kernel (that is Step 2.3 — describe() + the @-sugar) and do
not yet build Axis/Override IR (Step 2.4).

This is a PACKAGE: one decorator family per submodule. The public surface (the names
re-exported below) is unchanged — `from ...decorators import TensorSpec` etc. work as
before.

Surface:
    T   = ati.type_var('T_io', dtype=[...])          # named element-type variable
    S   = ati.scalar_var('Seqlen', options=[...])    # named scalar choice variable
    ati.tensor('Q', T, strides='stride_q?', contiguous=-1)
    ati.tensor('L', '*fp32:16', rank=2)              # literal dtype
    ati.scalar('Sm_scale', 'fp32')                   # plain runtime scalar
    ati.scalar('CAUSAL_TYPE', options=[0, 3])        # enumerated (former feature)
    ati.scalar('Max_seqlen_q', S)                    # bound to a shared variable
"""

from .choicevar import ChoiceVar, type_var, scalar_var
from .tensor import TensorSpec, tensor
from .scalar import ScalarSpec, scalar
from .derive import derives, overrides, eq, ne, lt, gt, le, ge
from .disable import DisableSpec, disable, no_disable
from .cite import CiteSpec, cite
from .source import KernelStub, SourceError, source
from .operator import BackendSpec, backend, OperatorSpec, operator
from .metro import metro_kernel

__all__ = [
    'ChoiceVar', 'type_var', 'scalar_var',
    'TensorSpec', 'ScalarSpec', 'tensor', 'scalar',
    'derives', 'overrides', 'eq', 'ne', 'lt', 'gt', 'le', 'ge',
    'DisableSpec', 'disable', 'no_disable',
    'CiteSpec', 'cite',
    'KernelStub', 'SourceError', 'source',
    'BackendSpec', 'backend', 'OperatorSpec', 'operator',
    'metro_kernel',
]
