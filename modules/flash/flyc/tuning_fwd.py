# SPDX-License-Identifier: Apache-2.0

"""Unified entry for every arch's flyc attention forward.

Every caller describes the problem with `fmha_meta.FmhaInputMetadata`, and
each arch's tuning module takes it as is and owns its pins. Its schedule has
two layers: per functional, `resolve` (inside `entry_resolve`/`entry_plan`);
per call, the `fmha_variants` rules its optional `variants(meta)` yields. Both
paths read both layers:

- AOT: `aot_builds` (the functional fixes the tiles), one build per distinct
  override set of the rules;
- JIT: `plan` derives the tiles from the real head dims and applies the rules
  that match the call's runtime values;
- both: `builder` constructs the module. It imports flydsl, so the code
  generator never calls it.

Everything here but `builder` is flydsl-free.
"""

import importlib
from dataclasses import asdict, fields

from fmha_variants import builds, overrides_at

_ARCH_MODULES = {
    'gfx950': 'gfx950.tuning_fwd',
    'gfx1201': 'gfx1201.tuning_fwd',
}


def arch_module(table, arch):
    """The tuning module `table` names for `arch`."""
    if arch not in table:
        raise ValueError(f'no flyc kernel for arch {arch!r}; known: {sorted(table)}')
    return importlib.import_module(table[arch])


def require_tile(meta):
    """`resolve_knobs` and `builder` take a tile: AOT's, or the one `plan` chose."""
    if meta.block_dmodel is None or meta.padded_head is None:
        raise ValueError("meta has no tile: AOT sets block_dmodel and padded_head, JIT gets them from plan()")


def require_head_dim(meta):
    """`plan` derives the tile from the caller's real head dim."""
    if meta.head_dim is None:
        raise ValueError("plan derives the tile from the caller's head_dim: set it")


def variants(arch, meta):
    """The per-call rules of `meta`'s functional (none for an arch without them)."""
    rules = getattr(arch_module(_ARCH_MODULES, arch), 'variants', None)
    return list(rules(meta)) if rules else []


def resolve_knobs(arch, meta, *, overrides=None):
    """The arch's per-functional knobs for `meta`, without the per-call rules.

    For sweeps and tools pinning a build; AOT goes through `aot_builds`.
    """
    require_tile(meta)
    return arch_module(_ARCH_MODULES, arch).entry_resolve(meta, overrides=overrides)


def aot_builds(arch, meta):
    """AOT: what a description's builder function returns for `meta`.

    `(build, knobs)` when one build serves every call, as before the rules
    existed; otherwise a generator of `(boxes, build, knobs)`, one per distinct
    override set, `boxes` being the calls it serves (`fmha_variants.builds`).
    `knobs` is the `asdict` the code generator records beside the hsaco;
    `build` is deferred because it imports flydsl.
    """
    require_tile(meta)
    entry = arch_module(_ARCH_MODULES, arch)

    def one(overrides):
        knobs = entry.entry_resolve(meta, overrides=overrides or None)

        def build(knobs=knobs):
            return entry.entry_build(meta, knobs)
        return build, asdict(knobs)

    found = builds(variants(arch, meta))
    if len(found) == 1 and found[0][0] == [{}]:
        return one(found[0][1])
    return ((boxes, *one(overrides)) for boxes, overrides in found)


def plan(arch, meta, *, overrides=None, seqlen_q=None, seqlen_k=None):
    """JIT: `(meta, knobs)` for calls with these runtime values.

    Rounds `meta`'s real head dims up the ladder (setting the tiles and
    `padded_head` AOT takes from the functional), then applies the rules
    matching the call: `hdim_qk` is `meta.head_dim`, the sequence lengths are
    the call's maxima. `overrides` (dict or knobs) wins over the rules, so a
    rule whose targets it pins is moot; any other rule reading a value not
    passed raises.
    """
    require_head_dim(meta)
    entry = arch_module(_ARCH_MODULES, arch)
    if isinstance(overrides, dict) or overrides is None:
        user = dict(overrides or {})
    else:
        user = {f.name: getattr(overrides, f.name)
                for f in fields(overrides) if getattr(overrides, f.name) is not None}
    meta, _ = entry.entry_plan(meta, overrides=user or None)
    rules = [r for r in variants(arch, meta) if not r.overrides.keys() <= user.keys()]
    values = {'hdim_qk': meta.head_dim, 'seqlen_q': seqlen_q, 'seqlen_k': seqlen_k}
    pins = overrides_at(rules, values) | user
    return entry.entry_plan(meta, overrides=pins or None)


def arches():
    """The arches this kernel has a row for."""
    return tuple(_ARCH_MODULES)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module; imports flydsl."""
    require_tile(meta)
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
