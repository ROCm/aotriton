# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The OperatorDecl passive record + its collector (pipeline Stage 2).

The stacked-@ operator finalizer (specs/finalize.py) partitions an @ati.operator
stack into one OperatorDecl, attached to the def as `fn.__ati_node__`. NO build —
the codegen linker constructs the Operator (ir/operator.py) from this record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from .node import AtiNode
from .bundle import partition

if TYPE_CHECKING:
    from ..decorators.operator import OperatorSpec, BackendSpec
    from .tune import BinningSelector


@dataclass(slots=True)
class OperatorDecl(AtiNode):
    """Passive record of an @ati.operator stack (the operator's "object file"): the
    OperatorSpec marker, the (index-sorted) BackendSpecs, and operator-level tune
    (binning -> OPTUNE_KEYS, fallback -> PARTIALLY_TUNED). NO build — the linker
    (codegen) consumes this. Attached to the def as `fn.__ati_node__`."""

    opspec: OperatorSpec
    backends: list[BackendSpec]            # index-sorted
    binning: dict[str, BinningSelector]    # operator backend-selection keys
    fallback: dict[str, object]            # {key -> value}

    @property
    def name(self):
        return self.opspec.name


def partition_operator(specs):
    """Common vocabulary, restricted: an operator stack is the dispatch layer
    above a kernel, not a kernel itself, so it has no use for a kernel's own
    ABI (tensors/scalars/overrides/dtype-vars), citation graph (cites), or
    disable predicate -- explicit `forbid()` for all five, rather than an
    absent `elif`, so e.g. an @ati.tensor on an @ati.operator stack fails
    naming the kind instead of silently landing in b.unrecognized.

    Tune records are NOT forbidden wholesale: an operator stack does have its
    own narrower tune vocabulary (backend-selection binning, its own partial
    fallback), just not the per-kernel one (perf configs). `b.tune_records`
    is left for collect_operator_decl below to sort out, since telling
    BinningSpec/FallbackSpec/ConfigsSpec apart needs more than a single
    forbid() line."""
    b = partition(specs)
    b.forbid('tensors', 'scalars', 'dtype_vars', 'overrides', 'cites', 'disable',
             what='@ati.operator')
    return b


def collect_operator_decl(placeholder, specs):
    """Partition an @ati.operator stack into a passive OperatorDecl (no build).
    `placeholder` is unused -- accepted only so every collect_*_decl shares the
    same (placeholder, specs) signature for specs/finalize.py's start()
    dispatch table."""
    import warnings
    from ..decorators import OperatorSpec, BackendSpec
    from .tune import BinningSpec, FallbackSpec, ConfigsSpec

    b = partition_operator(specs)
    opspec = None
    backends = []
    remaining = []
    for s in b.unrecognized:
        if isinstance(s, OperatorSpec):
            assert opspec is None, 'multiple @ati.operator markers in one stack'
            opspec = s
        elif isinstance(s, BackendSpec):
            backends.append(s)
        else:
            remaining.append(s)
    b.unrecognized = remaining
    b.reject_remaining('@ati.operator')

    # The common bucket lumps PerfSchema/ConfigsSpec/BinningSpec/FallbackSpec
    # together, but an operator's tune vocabulary is narrower than a kernel's:
    # it selects a BACKEND (binning) and states its own partial-tune
    # (fallback); it does not generate perf configs of its own (PerfSchema/
    # ConfigsSpec describe a single kernel's tuning), so PerfSchema is still
    # rejected here exactly as it was before the common `partition()` existed.
    binning = {}
    fallback = {}
    for s in b.tune_records:
        if isinstance(s, BinningSpec):
            binning.update(s.keys)            # operator backend-selection keys
        elif isinstance(s, FallbackSpec):
            fallback.update(s.values)         # operator's OWN partial-tune (default {})
        elif isinstance(s, ConfigsSpec):
            warnings.warn(
                f'@ati.tune.configs on operator {opspec.name if opspec else "?"!r} '
                f'is ignored: operator tuning selects a backend (binning) and does '
                f'not generate perf configs. Move configs to the kernel/tune module.',
                stacklevel=3)
        else:
            raise AssertionError(
                f'unexpected spec {s!r} in an @ati.operator stack; operators accept '
                f'only @ati.backend and operator-level @ati.tune.binning/fallback')
    assert opspec is not None, '@ati.start operator path without an @ati.operator marker'
    assert backends, f'operator {opspec.name!r} declares no @ati.backend'
    backends.sort(key=lambda spec: spec.index)
    return OperatorDecl(opspec, backends, binning, fallback)
