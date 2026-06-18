# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The AffineDecl passive record + its collector (pipeline Stage 2).

The stacked-@ affine finalizer (specs/finalize.py) partitions an @ati.affine.*
stack into one AffineDecl, attached to the def as `fn.__ati_node__`. NO build —
the codegen linker constructs the AffineKernel (ir/affine.py) from this record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from .node import AtiNode

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


def collect_affine_decl(specs):
    """Partition an @ati.affine stack into a passive AffineDecl (no build)."""
    from ..decorators.affine import (
        AffineKernelSpec, SharedOperatorSpec, ArchSpec,
        LimitationsSpec, StructuresSpec, DirectoriesSpec, SuppliesSpec,
    )
    from ..decorators import DisableSpec

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
    disable = None
    for s in specs:
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
            supplies_after = s.after if s.after is not None else supplies_after
            supplies_before = s.before if s.before is not None else supplies_before
        elif isinstance(s, DisableSpec):
            disable = s
        else:
            raise AssertionError(
                f'unexpected spec {s!r} in an @ati.affine stack; affine kernels '
                f'accept @ati.affine.* and @ati.disable only')
    assert marker is not None, '@ati.start affine path without an @ati.affine marker'
    assert co_dir is not None, f'affine kernel {marker.name!r} missing @ati.affine.directories'
    return AffineDecl(name=marker.name, co_dir=co_dir, cookie=cookie, headers=headers,
                      supported_arch=arches, choice_filters=filters,
                      shared_operator_name=shared_op, supplied_specs=supplied,
                      supplies_after=supplies_after, supplies_before=supplies_before,
                      disable=disable)
