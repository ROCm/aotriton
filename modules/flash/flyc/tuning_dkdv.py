# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention dK/dV kernel.

Same contract as `tuning_fwd`.
"""

from tuning_fwd import arch_module, require_head_dim, require_tile

_ARCH_MODULES = {
    'gfx950': 'gfx950.tuning_dkdv',
    'gfx1201': 'gfx1201.tuning_dkdv',
}


def resolve_knobs(arch, meta, *, overrides=None):
    """The arch's knobs for `meta`; `overrides` (dict or knobs) passes through."""
    require_tile(meta)
    return arch_module(_ARCH_MODULES, arch).entry_resolve(meta, overrides=overrides)


def plan(arch, meta, *, overrides=None):
    """JIT: `(meta, knobs)`, with `meta`'s real head dims rounded up the ladder.

    Sets the tiles and `padded_head` that AOT takes from the functional.
    """
    require_head_dim(meta)
    return arch_module(_ARCH_MODULES, arch).entry_plan(meta, overrides=overrides)


def arches():
    """The arches this kernel has a row for."""
    return tuple(_ARCH_MODULES)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module; imports flydsl."""
    require_tile(meta)
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
