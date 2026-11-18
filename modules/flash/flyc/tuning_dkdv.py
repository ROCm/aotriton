# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention dK/dV kernel.

Same contract as `tuning_fwd`.
"""

from tuning_fwd import arch_module

_ARCH_MODULES = {
    'gfx950': 'gfx950.tuning_dkdv',
    'gfx1201': 'gfx1201.tuning_dkdv',
}


def resolve_knobs(arch, meta, *, overrides=None):
    """The arch's knobs for `meta`; `overrides` (dict or knobs) passes through."""
    return arch_module(_ARCH_MODULES, arch).entry_resolve(meta, overrides=overrides)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module. Imports flydsl: build time only."""
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
