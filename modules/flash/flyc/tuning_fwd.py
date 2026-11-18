# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention forward.

flydsl-free: the code generator calls `resolve_knobs`. `builder` imports the
kernel, so only the build driver calls it. Descriptions describe the problem
with the arch-neutral `FmhaInputMetadata`; each arch's tuning module adapts it
(`entry_resolve`, `entry_build`) and owns its pins.
"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class FmhaInputMetadata:
    """One attention problem, as every flyc kernel (fwd, dQ, dK/dV) sees it."""

    num_heads: int
    # The functional's BLOCK_DMODEL and PADDED_HEAD: the compiled tile, and
    # whether calls may bring a smaller head dim.
    block_dmodel: int
    padded_head: bool
    # AOTriton's CAUSAL_TYPE: 0 none, 1 top-left, 2 bottom-right, 3 window.
    causal_type: int
    dtype_str: str
    bias: bool
    dropout: bool


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
