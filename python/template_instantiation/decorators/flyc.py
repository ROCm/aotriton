# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `ati.flyc.*` decorator surface (FlyDSL-compiled kernels).

A flyc kernel is a THIRD kind of backend, between triton (`@ati.source`,
compiled during the build, ATI owns the perf space) and affine
(`@ati.affine.*`, prebuilt `.co`, no perf space): it is compiled during the
build from a FlyDSL description, it inherits and filters the operator's
functional axes rather than owning a space, and it dispatches an hsaco whose
kernel-argument list is NOT the operator's -- so that argument list has to be
declared, which nothing else can supply. Its description uses the stacked-@
form:

    @ati.start
    @ati.tensor('Q', 'T_io', rank=4, strides='stride_q_*', wires_to='Q')
    ...
    @ati.scalar('varlen_bits', 'i32', wires_to='Varlen_bits')
    @ati.flyc.hints(FlycFwdHints)                 # optimization-input dataclass
    @ati.flyc.kernel()                            # innermost marker
    def flyc_attn_fwd(arch, choices, hints):
        ...
        build.flyc_source = 'flash_attn_func_gfx1201_aiw.py'   # in modules/flash/flyc/
        build.flyc_kernel_name = 'flash_attn_func_aiw_kernel'  # the @flyc.kernel def
        return build, knobs

These produce passive spec records; `specs/flyc.py` collects them into a
`FlycDecl`. See `docs/FlyDSL.md`.
"""

from __future__ import annotations

from ..specs.base import StackedSpec


# --- spec records (callable -> accumulate onto the placeholder def) ----------


class FlycKernelSpec(StackedSpec):
    """@ati.flyc.kernel: the innermost marker that makes the def a
    flyc-kernel description (the flyc analogue of @ati.affine.aiter_asm /
    @ati.source).

    Carries NO path. The vendored kernel FILE the description's builder
    actually drives can vary by `arch` -- one description serving two
    architectures is the point -- so it cannot be known here, at decoration
    time, before any arch is chosen. Instead the builder sets it, once it has
    resolved `arch`, as two attributes on the `build` closure it returns:
    `build.flyc_source` (the vendored file, relative to the family's `flyc/`
    directory) and `build.flyc_kernel_name` (the `@flyc.kernel` def's own name
    inside that file). Consumers read those two strings off `build` -- never
    anything stashed here eagerly.

    The operator whose functionals this kernel inherits is NOT declared here
    either: which operator a kernel serves is the operator's fact, not the
    kernel's, and it is declared on the operator side like every other
    backend."""

    __slots__ = ()

    def __repr__(self):
        return 'FlycKernelSpec()'


class FlycHintsSpec(StackedSpec):
    """@ati.flyc.hints(Dataclass): the optimization-input dataclass a flyc
    builder may read -- sequence lengths and the like, NOT functional axes.
    A default-constructed instance is what a caller with nothing better to say
    passes in.

    Lives here rather than in `decorators/tune.py`: `ati.tune.*` is the SHARED
    tuning vocabulary feeding the lookup tables and the tuning database, while
    this feeds one description's builder and nothing else. `ati.affine.*` is
    the precedent for a backend-specific namespace."""

    __slots__ = ('hints_cls',)

    def __init__(self, hints_cls):
        self.hints_cls = hints_cls

    def __repr__(self):
        return f'FlycHintsSpec({self.hints_cls!r})'


# --- public decorator namespace (ati.flyc.*) --------------------------------


def kernel():
    """@ati.flyc.kernel(): innermost marker, no path. See `FlycKernelSpec` for
    where the vendored file and def name come from instead."""
    return FlycKernelSpec()


def hints(hints_cls):
    """@ati.flyc.hints(Dataclass): register the builder's optimization-input
    dataclass. See `FlycHintsSpec`."""
    return FlycHintsSpec(hints_cls)
