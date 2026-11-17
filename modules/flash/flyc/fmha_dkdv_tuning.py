# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention dK/dV kernel.

Same contract as `fmha_fwd_tuning`, whose `FmhaInputMetadata` it takes.
"""

from fmha_fwd_tuning import FmhaInputMetadata, arch_module  # noqa: F401

_ARCH_MODULES = {
    'gfx950': 'fmha_tuning_bwd_dkdv_gfx950',
    'gfx1201': 'fmha_tuning_bwd_dkdv_gfx1201',
}


def resolve_knobs(arch, meta, *, overrides=None):
    """The arch's knobs for `meta`; `overrides` (dict or knobs) passes through."""
    return arch_module(_ARCH_MODULES, arch).entry_resolve(meta, overrides=overrides)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module. Imports flydsl: build time only."""
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
