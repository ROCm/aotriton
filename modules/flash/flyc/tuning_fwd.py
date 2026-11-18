# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention forward.

Every caller describes the problem with `fmha_meta.FmhaInputMetadata`, and
each arch's tuning module takes it as is and owns its pins:

- AOT: the code generator calls `resolve_knobs` (the functional fixes the tiles);
- JIT: `plan` derives the tiles from the real head dims, then resolves;
- both: `builder` constructs the module. It imports flydsl, so the code
  generator never calls it.

Everything here but `builder` is flydsl-free.
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


def plan(arch, meta, *, overrides=None):
    """JIT: `(meta, knobs)`, with `meta`'s real head dims rounded up the ladder.

    Sets the tiles and `padded_head` that AOT takes from the functional.
    """
    return arch_module(_ARCH_MODULES, arch).entry_plan(meta, overrides=overrides)


def arches():
    """The arches this kernel has a row for."""
    return tuple(_ARCH_MODULES)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module; imports flydsl."""
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
