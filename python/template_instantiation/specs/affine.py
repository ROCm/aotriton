# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The AffineDecl passive record + its collector (pipeline Stage 2).

The stacked-@ affine finalizer (specs/finalize.py) partitions an @ati.affine.*
stack into one AffineDecl, attached to the def as `fn.__ati_affine__`. NO build —
the codegen linker constructs the AffineKernel (ir/affine.py) from this record.
"""


class AffineDecl:
    """Passive record of an @ati.affine stack (the affine kernel's "object file"):
    the marker + @ati.affine.* metadata + optional @ati.disable. NO build — the
    linker (codegen) consumes this. Attached to the def as `fn.__ati_affine__`."""

    __slots__ = ('name', 'co_dir', 'cookie', 'headers', 'supported_arch',
                 'choice_filters', 'shared_operator_name', 'supplied_specs',
                 'supplies_after', 'supplies_before', 'disable')

    def __init__(self, *, name, co_dir, cookie, headers, supported_arch,
                 choice_filters, shared_operator_name, supplied_specs,
                 supplies_after, supplies_before, disable):
        self.name = name
        self.co_dir = co_dir
        self.cookie = cookie
        self.headers = headers
        self.supported_arch = supported_arch
        self.choice_filters = choice_filters
        self.shared_operator_name = shared_operator_name
        self.supplied_specs = supplied_specs
        self.supplies_after = supplies_after
        self.supplies_before = supplies_before
        self.disable = disable


def collect_affine_decl(specs):
    """Partition an @ati.affine stack into a passive AffineDecl (no build)."""
    from ..decorators.affine import (
        AffineMarkerSpec, SharedOperatorSpec, ArchSpec,
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
        if isinstance(s, AffineMarkerSpec):
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
    assert marker is not None, '@ati.kernel affine path without an @ati.affine marker'
    assert co_dir is not None, f'affine kernel {marker.name!r} missing @ati.affine.directories'
    return AffineDecl(name=marker.name, co_dir=co_dir, cookie=cookie, headers=headers,
                      supported_arch=arches, choice_filters=filters,
                      shared_operator_name=shared_op, supplied_specs=supplied,
                      supplies_after=supplies_after, supplies_before=supplies_before,
                      disable=disable)
