# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The AffineDecl passive record + its collector (pipeline Stage 2).

The stacked-@ affine finalizer (specs/finalize.py) partitions an @ati.affine.*
stack into one AffineDecl, attached to the def as `fn.__ati_node__`. NO build —
the codegen linker constructs the AffineKernel (ir/affine/kdesc.py) from this record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from .node import AtiNode
from .bundle import partition

if TYPE_CHECKING:
    from ..decorators.disable import DisableSpec


@dataclass(slots=True, kw_only=True)
class AffineDecl(AtiNode):
    """Passive record of an @ati.affine stack (the affine kernel's "object file"):
    the marker + @ati.affine.* metadata + optional @ati.disable. NO build — the
    linker (codegen) consumes this. Attached to the def as `fn.__ati_node__`."""

    name: str
    co_dir: str
    cookie: str | None                     # the 3rd-party COOKIE_CLASS name
    headers: list[str]
    supported_arch: list[str]
    choice_filters: dict[str, object]      # {arg name -> predicate callable}
    shared_operator_name: str | None
    supplied_specs: list                   # list[TensorSpec | ScalarSpec]
    supplies_after: str | None             # neighbor operand name (union order)
    supplies_before: str | None
    disable: DisableSpec | None


def partition_affine(specs):
    """Common vocabulary, restricted: an affine kernel has no Python signature
    to bind gap arguments or dtype-vars against, and no perf-tuning concept of
    its own, so it accepts none of @ati.tensor/@ati.scalar/@ati.type_var/
    @ati.derives, @ati.cite, or the tune-record specs -- explicit `forbid()`
    calls rather than the absence of an `elif`, so an @ati.tensor on an affine
    stack fails naming the kind, instead of silently landing in
    b.unrecognized. Only @ati.disable is shared with the common vocabulary;
    the AffineKernelSpec marker and its @ati.affine.* metadata are
    affine-specific and claimed out of b.unrecognized by collect_affine_decl
    below."""
    b = partition(specs)
    b.forbid('tensors', 'scalars', 'dtype_vars', 'overrides', 'cites',
             'tune_records', what='@ati.affine')
    return b


def collect_affine_decl(placeholder, specs):
    """Partition an @ati.affine stack into a passive AffineDecl (no build).
    `placeholder` is unused -- accepted only so every collect_*_decl shares the
    same (placeholder, specs) signature for specs/finalize.py's start()
    dispatch table."""
    from ..decorators.affine import (
        AffineKernelSpec, SharedOperatorSpec, ArchSpec,
        LimitationsSpec, StructuresSpec, DirectoriesSpec, SuppliesSpec,
    )

    b = partition_affine(specs)
    marker = None
    shared_op = None
    arches = []
    filters = {}
    cookie = None
    co_dir = None
    headers = []
    supplied = []
    supplies_after = None
    supplies_before = None
    remaining = []
    for s in b.unrecognized:
        if isinstance(s, AffineKernelSpec):
            assert marker is None, 'multiple @ati.affine.aiter_asm markers in one stack'
            marker = s
        elif isinstance(s, SharedOperatorSpec):
            shared_op = s.op_name
        elif isinstance(s, ArchSpec):
            arches = s.arches
        elif isinstance(s, LimitationsSpec):
            filters.update(s.filters)
        elif isinstance(s, StructuresSpec):
            cookie = s.cookie
        elif isinstance(s, DirectoriesSpec):
            co_dir, headers = s.co_dir, s.headers
        elif isinstance(s, SuppliesSpec):
            supplied.extend(s.specs)
            if s.after is not None:
                assert supplies_after is None or supplies_after == s.after, (
                    f'conflicting after= anchors in @ati.affine.supplies: '
                    f'{supplies_after!r} vs {s.after!r}')
                supplies_after = s.after
            if s.before is not None:
                assert supplies_before is None or supplies_before == s.before, (
                    f'conflicting before= anchors in @ati.affine.supplies: '
                    f'{supplies_before!r} vs {s.before!r}')
                supplies_before = s.before
        else:
            remaining.append(s)
    b.unrecognized = remaining
    b.reject_remaining('@ati.affine')
    assert marker is not None, '@ati.start affine path without an @ati.affine marker'
    assert co_dir is not None, f'affine kernel {marker.name!r} missing @ati.affine.directories'
    return AffineDecl(name=marker.name, co_dir=co_dir, cookie=cookie, headers=headers,
                      supported_arch=arches, choice_filters=filters,
                      shared_operator_name=shared_op, supplied_specs=supplied,
                      supplies_after=supplies_after, supplies_before=supplies_before,
                      disable=b.disable)
