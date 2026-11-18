# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention forward.

flydsl-free: the code generator calls `resolve_knobs`. `builder` imports the
kernel, so only the build driver calls it. Every caller describes the problem
with `fmha_meta.FmhaInputMetadata`; each arch's tuning module takes it as is
(`entry_resolve`, `entry_build`) and owns its pins.
"""

import importlib

_ARCH_MODULES = {
    'gfx950': 'gfx950.tuning_fwd',
    'gfx1201': 'gfx1201.tuning_fwd',
}


def arch_module(table, arch):
    """The tuning module `table` names for `arch`."""
    if arch not in table:
        raise ValueError(f'no flyc kernel for arch {arch!r}; known: {sorted(table)}')
    return importlib.import_module(table[arch])


def resolve_knobs(arch, meta, *, overrides=None):
    """The arch's knobs for `meta`; `overrides` (dict or knobs) passes through."""
    return arch_module(_ARCH_MODULES, arch).entry_resolve(meta, overrides=overrides)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module. Imports flydsl: build time only."""
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
