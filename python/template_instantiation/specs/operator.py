# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The OperatorDecl passive record + its collector (pipeline Stage 2).

The stacked-@ operator finalizer (specs/finalize.py) partitions an @ati.operator
stack into one OperatorDecl, attached to the def as `fn.__ati_operator__`. NO build —
the codegen linker constructs the Operator (ir/operator.py) from this record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..decorators.operator import OperatorSpec, BackendSpec
    from .tune import BinningSelector


@dataclass(slots=True)
class OperatorDecl:
    """Passive record of an @ati.operator stack (the operator's "object file"): the
    OperatorSpec marker, the (index-sorted) BackendSpecs, and operator-level tune
    (binning -> OPTUNE_KEYS, fallback -> PARTIALLY_TUNED). NO build — the linker
    (codegen) consumes this. Attached to the def as `fn.__ati_operator__`."""

    opspec: OperatorSpec
    backends: list[BackendSpec]            # index-sorted
    binning: dict[str, BinningSelector]    # operator backend-selection keys
    fallback: dict[str, object]            # {key -> value}

    @property
    def name(self):
        return self.opspec.name


def collect_operator_decl(specs):
    """Partition an @ati.operator stack into a passive OperatorDecl (no build)."""
    import warnings
    from ..decorators import OperatorSpec, BackendSpec
    from .tune import BinningSpec, FallbackSpec, ConfigsSpec

    opspec = None
    backends = []
    binning = {}
    fallback = {}
    for s in specs:
        if isinstance(s, OperatorSpec):
            assert opspec is None, 'multiple @ati.operator markers in one stack'
            opspec = s
        elif isinstance(s, BackendSpec):
            backends.append(s)
        elif isinstance(s, BinningSpec):
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
    backends.sort(key=lambda b: b.index)
    return OperatorDecl(opspec, backends, binning, fallback)
