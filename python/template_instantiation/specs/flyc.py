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

`aotriton.flyc_compile` reads `source_path` and `hints()` off the record; the
linker (`codegen/linker.py:_build_flycs`) reads the rest.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .node import BuildableDecl, derived_decl_fields
from .bundle import partition
from ..ast_params import find_one_function, collect_params

if TYPE_CHECKING:
    from ..decorators.disable import DisableSpec
    from ..decorators.cite import CiteSpec


@dataclass(kw_only=True)
class FlycDecl(BuildableDecl):
    """Passive record of an @ati.flyc stack (a flyc kernel's "object file").

    Adds three fields to `BuildableDecl` (specs/node.py), and they are the
    three places flyc genuinely differs from Triton -- all of them build- or
    tune-time, none of them about how the kernel is DESCRIBED:

      desc_path       the description module's own file, for the Fly.compile
                      DESC column (build plumbing)
      hints_cls       the @ati.flyc.hints dataclass (tune)
      fn              the deferred builder the FlyDSL compiler calls (build)

    Everything else -- the kernarg ABI specs, cites, disable, params -- is the
    shared vocabulary. `source_path` is ALSO shared by name, but for a
    `FlycDecl` it means something narrower than for a Triton `KernelDecl`: the
    vendored flyc DIRECTORY (e.g. `modules/flash/flyc/`), not a specific kernel
    file, because the description does not name one until its builder resolves
    `arch` (see decorators/flyc.py).

    `cites` is plural (inherited): stacking several `@ati.cite` is a deliberate
    pattern, "citation mode (b)". `disable` stays singular, enforced by
    `partition_flyc`.
    """

    desc_path: Path = None             # the DESCRIPTION module's own file
    hints_cls: type | None = None      # the @ati.flyc.hints dataclass, or None
    fn: object = None                  # the placeholder def itself (the builder)

    def hints(self):
        """A default-constructed instance of the registered hints dataclass, or
        None if @ati.flyc.hints was never applied."""
        if self.hints_cls is None:
            return None
        return self.hints_cls()


def _flyc_kernel_stub(module_path, kernel_name):
    """AST-parse `module_path` (the vendored kernel file) for the function
    named EXACTLY `kernel_name` and wrap it as a `KernelStub` -- the same
    non-importing stand-in `@ati.source` builds for a Triton kernel
    (decorators/source.py).

    Selected by NAME, not by "the unique `@flyc.kernel`-decorated function in
    this file": an explicit name is what the description itself declares (`build.flyc_kernel_name`, set by the builder once `arch`
    is known -- see decorators/flyc.py), and matching it exactly is robust to
    a vendored file someday holding more than one `@flyc.kernel` def, which a
    uniqueness assumption is not. The flyc kernel def may still sit in a
    nested scope (unlike a top-level-only Triton kernel), so this walks every
    scope (`walk=True`)."""
    from ..decorators.source import KernelStub

    path = Path(module_path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    what = f'flyc kernel {kernel_name!r}: {path}'
    fn = find_one_function(tree, lambda n: n.name == kernel_name,
                           walk=True, what=what)
    params = collect_params(fn, what=what)
    return KernelStub(fn.name, params, str(path))


def _synth_param_order(tensors, scalars):
    """A stable declaration-order parameter list, used ONLY to seed
    `build_kernel`'s axis-anchor computation (`builder/kernel.py`'s
    `_build_axes`, which needs SOME `{name: index}` map covering every
    declared `@ati.tensor`/`@ati.scalar` `arg_name` or it raises `KeyError`).

    Deliberately NOT the real, AST-parsed flyc kernel signature. The real
    per-arch kernel file and def name live on the `build` closure the
    description's builder function returns, which cannot be
    called this early: link time (`codegen/linker.py:_build_flycs`, where
    `build_kernel` runs) has no `Functional` yet, and every flyc builder seen
    so far reads real `choices` values before setting
    `build.flyc_source`/`build.flyc_kernel_name` -- so resolving the real
    stub is only possible once a concrete functional exists, i.e. at
    generate time (`codegen/flytune.py`'s `_gen_signatures`, per functional).
    That is where `ir/flyc/kdesc.py`'s `iter_launch_arguments` gets ITS real,
    per-arch parameter order from -- this function's output never reaches
    codegen (`BuiltKernel.arguments`/`.axes` are both dead weight for flyc,
    see kdesc.py's `iter_launch_arguments` and `_axes_overrides`).

    Sound as an anchor order because every flyc description already keeps its
    `@ati.tensor`/`@ati.scalar` stack in the real kernel's argument order by
    convention (see flyc_attn_fwd.py's "the kernarg ABI, in
    flash_attn_func_aiw_kernel order" comment): `_build_axes` only needs a
    stable RELATIVE order to sort axes it never emits, not the literal
    AST-derived names or the stride parameters this list omits."""
    from ..introspect import ParamSpec

    seen = []
    for spec in (*tensors, *scalars):
        for name in spec.arg_names:
            if name not in seen:
                seen.append(name)
    return [ParamSpec(name=n, is_constexpr=False, annotation=ParamSpec.EMPTY)
            for n in seen]


def partition_flyc(specs):
    """Common vocabulary, restricted: a flyc kernel has the same kernarg-ABI
    concept as a Triton kernel (tensors/scalars/overrides/dtype-vars) and the
    same @ati.cite(s)/@ati.disable, but no tune-record concept of its own yet
    -- `tune` on FlycDecl is an inert placeholder nothing populates. The
    @ati.flyc.* markers (FlycKernelSpec/FlycHintsSpec) are flyc-specific and
    claimed out of b.unrecognized by collect_flyc_decl below.

    No `forbid('cites', ...)` here: unlike specs/affine.py, flyc accepts the
    common vocabulary's plural `cites` outright -- there used to be a
    `assert cite is None` here restricting a flyc stack to one @ati.cite, but
    that was an artifact of this collector being written narrowly before the
    common vocabulary existed, not a rule flyc actually needs; multi-cite
    ("citation mode (b)", see specs/bundle.py) is just as legitimate for a
    flyc kernel as for a Triton one."""
    b = partition(specs)
    b.forbid('tune_records', what='@ati.flyc')
    return b


def collect_flyc_decl(placeholder, specs):
    """Partition an @ati.flyc stack into a passive FlycDecl (no build, no
    describe() validation).

    Does NOT resolve a real kernel stub: `FlycKernelSpec` carries no path, so
    there is no vendored file to AST-parse yet -- only
    the description's builder function, once called with a concrete `arch`,
    knows which file/def to use (`build.flyc_source`/`build.flyc_kernel_name`).
    That resolution happens later, per arch, in `codegen/flytune.py`; see
    `_synth_param_order` for why `params` does not need it either."""
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
    # The DESCRIPTION module's own file (e.g. modules/flash/aot/flyc_attn_fwd.py).
    # This is what codegen/root.py's Fly.compile DESC column feeds to
    # `aotriton.flyc_compile <desc_path> --kernel_name ...`.
    desc_path = Path(inspect.getfile(placeholder)).resolve()
    # The vendored flyc directory, a fixed sibling of the description
    # family's own package (modules/flash/aot/flyc_attn_fwd.py's parent's
    # parent, plus 'flyc').
    vendored_dir = desc_path.parent.parent / 'flyc'
    fields = derived_decl_fields(None, b.disable, name=placeholder.__name__)
    fields['source_path'] = str(vendored_dir)
    # `name` passed explicitly: the flyc DESCRIPTION's identity, not the
    # (no longer eagerly known) AST-located @flyc.kernel def's name in the
    # vendored file.
    return FlycDecl(desc_path=desc_path,
                    hints_cls=hints_cls, fn=placeholder,
                    params=_synth_param_order(b.tensors, b.scalars),
                    tensors=b.tensors,
                    scalars=b.scalars, overrides=b.overrides,
                    dtype_vars=b.dtype_vars, cites=b.cites,
                    **fields)
