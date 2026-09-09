# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The FlycDecl passive record + its collector (pipeline Stage 2).

The stacked-@ flyc finalizer (specs/finalize.py) partitions an `@ati.flyc.*`
stack into one FlycDecl, attached to the def as `fn.__ati_node__`. NO build,
and -- unlike a Triton kernel -- NOT routed through `describe()`.

That last point is the substantive one. `describe()` validates that the specs
claim every parameter of a known, AST-parsed signature exactly once. A flyc
description has no such signature: the def it decorates is the BUILDER, whose
parameters are `(arch, choices, hints)`, and the kernel-argument list being
described belongs to a vendored FlyDSL kernel somewhere else entirely. So the
`@ati.tensor` / `@ati.scalar` stack IS the declaration rather than a claim
against one, and there is nothing for `describe()` to check it against.
Collected passively instead, the way `AffineDecl` is.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .node import AtiNode
from .bundle import partition
from ..ast_params import find_one_function, collect_params

if TYPE_CHECKING:
    from ..decorators.disable import DisableSpec


@dataclass(slots=True, kw_only=True)
class FlycDecl(AtiNode):
    """Passive record of an `@ati.flyc` stack (a flyc kernel's "object file").

    `source_path` is the vendored flyc DIRECTORY (e.g. `modules/flash/flyc/`),
    not a kernel file: the description does not name one until its builder has
    resolved `arch`. `desc_path` is the description module's own file, which
    is what an out-of-process driver is pointed at.

    `cites` is plural: stacking several `@ati.cite` is a deliberate pattern.
    `disable` is singular."""

    name: str
    source_path: str                       # the vendored flyc DIRECTORY
    desc_path: Path                        # the DESCRIPTION module's own file
    fn: object                             # the builder def itself
    hints_cls: type | None = None          # the @ati.flyc.hints dataclass
    tensors: list = field(default_factory=list)
    scalars: list = field(default_factory=list)
    overrides: list = field(default_factory=list)
    dtype_vars: list = field(default_factory=list)
    cites: list = field(default_factory=list)
    disable: DisableSpec | None = None

    def hints(self):
        """A default-constructed instance of the registered hints dataclass, or
        None if `@ati.flyc.hints` was never applied."""
        if self.hints_cls is None:
            return None
        return self.hints_cls()


def _flyc_kernel_stub(module_path, kernel_name):
    """AST-parse `module_path` (a vendored kernel file) for the function named
    EXACTLY `kernel_name` and wrap it as a `KernelStub` -- the same
    non-importing stand-in `@ati.source` builds for a Triton kernel.

    Selected BY NAME rather than as "the unique `@flyc.kernel`-decorated
    function in this file": the name is what the description itself declares
    (`build.flyc_kernel_name`, set by the builder once `arch` is known), and
    matching it exactly is robust to a vendored file someday holding more than
    one kernel, which a uniqueness assumption is not. A flyc kernel def may sit
    in a nested scope, unlike a top-level-only Triton kernel, so this walks
    every scope."""
    from ..decorators.source import KernelStub

    path = Path(module_path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    what = f'flyc kernel {kernel_name!r}: {path}'
    fn = find_one_function(tree, lambda n: n.name == kernel_name,
                           walk=True, what=what)
    params = collect_params(fn, what=what)
    return KernelStub(fn.name, params, str(path))


def partition_flyc(specs):
    """Common vocabulary, restricted: a flyc kernel has the same kernarg-ABI
    concept as a Triton kernel (tensors/scalars/overrides/dtype-vars) and the
    same @ati.cite(s)/@ati.disable, but no tune-record concept -- its knobs
    come from the builder, not from a perf space. The @ati.flyc.* markers
    (FlycKernelSpec/FlycHintsSpec) are flyc-specific and are claimed out of
    `b.unrecognized` by `collect_flyc_decl` below.

    No `forbid('cites', ...)` here: unlike specs/affine.py, flyc accepts the
    common vocabulary's plural `cites` outright. Multi-cite ("citation mode
    (b)", see specs/bundle.py) is as legitimate for a flyc kernel as for a
    Triton one."""
    b = partition(specs)
    b.forbid('tune_records', what='@ati.flyc')
    return b


def collect_flyc_decl(placeholder, specs):
    """Partition an `@ati.flyc` stack into a passive FlycDecl (no build, no
    `describe()` validation).

    Does NOT resolve a real kernel stub: `FlycKernelSpec` carries no path, so
    there is no vendored file to AST-parse yet. Only the builder, once called
    with a concrete `arch`, knows which file and def to use."""
    from ..decorators.flyc import FlycKernelSpec, FlycHintsSpec

    b = partition_flyc(specs)
    marker = None
    hints_cls = None
    remaining = []
    for s in b.unrecognized:
        if isinstance(s, FlycKernelSpec):
            assert marker is None, 'multiple @ati.flyc.kernel markers in one stack'
            marker = s
        elif isinstance(s, FlycHintsSpec):
            assert hints_cls is None, 'duplicate @ati.flyc.hints on one kernel'
            hints_cls = s.hints_cls
        else:
            remaining.append(s)
    b.unrecognized = remaining
    b.reject_remaining('@ati.flyc')
    assert marker is not None, '@ati.start flyc path without an @ati.flyc.kernel marker'

    # The DESCRIPTION module's own file, e.g. modules/flash/aot/flyc_attn_fwd.py.
    desc_path = Path(inspect.getfile(placeholder)).resolve()
    # The vendored flyc directory, a fixed sibling of the description family's
    # own package: <family>/aot/<desc>.py -> <family>/flyc/.
    vendored_dir = desc_path.parent.parent / 'flyc'
    return FlycDecl(name=placeholder.__name__,
                    source_path=str(vendored_dir),
                    desc_path=desc_path,
                    fn=placeholder,
                    hints_cls=hints_cls,
                    tensors=b.tensors, scalars=b.scalars, overrides=b.overrides,
                    dtype_vars=b.dtype_vars, cites=b.cites,
                    disable=b.disable)
