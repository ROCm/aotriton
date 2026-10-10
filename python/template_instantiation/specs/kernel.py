# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The KernelDecl passive record (pipeline Stage 2).

`@ati.describe` / the stacked-@ finalizer (specs/finalize.py) attaches one of these
to a kernel as `kernel.__ati_node__`: the introspected signature plus the partitioned
spec-records (tensors / scalars / overrides / tune / disable / dtype_vars / cites).
It is a passive "object file" — no Axis/Override IR is built here; the linker
(codegen) resolves cites then the builder lowers it to a KernelDescription.
"""

from __future__ import annotations

from dataclasses import dataclass

from .node import BuildableDecl


@dataclass(kw_only=True)
class KernelDecl(BuildableDecl):
    """The ATI sidecar attached to a Triton kernel as `kernel.__ati_node__`.

    Adds NOTHING to `BuildableDecl` (specs/node.py) — a Triton description's
    fields are exactly the set the builder pipeline reads, which is why the base
    holds all of them. The class stays a distinct type because dispatch and
    identity depend on it: `get_kernel_decl()` isinstance-checks it, and which
    concrete record `__ati_node__` holds is what tells the parser which shell to
    build.

    Cloned during linking (`clone()`, inherited): cite resolution appends gap
    tensors/scalars/overrides/dtype_vars onto a per-link copy, so the
    module-level record every test and description reads is never touched.
    """

    def __repr__(self):
        return (f'KernelDecl({self.name!r}, {len(self.tensors)} tensors, '
                f'{len(self.scalars)} scalars, {len(self.overrides)} overrides)')
