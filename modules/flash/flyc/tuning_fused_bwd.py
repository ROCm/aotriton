# SPDX-License-Identifier: Apache-2.0

"""Unified entry for the single-pass fused flyc backward (dQ, dK and dV in one launch).

The `tuning_dkdv.py` pattern, with one more input: one workgroup owns every KV
row, so the build depends on the KV extent, which `FmhaInputMetadata` does not
carry. `plan` takes it as `max_seqlen_k` and the arch's policy turns it into
the `block_kv` knob; `resolve_knobs` wants `block_kv` pinned in `overrides`.

JIT only: no AOT description builds this kernel yet. Everything here but
`builder` is flydsl-free.
"""

from tuning_fwd import arch_module, require_head_dim, require_tile

_ARCH_MODULES = {
    'gfx950': 'gfx950.tuning_fused_bwd',
}


def resolve_knobs(arch, meta, *, overrides=None):
    """The arch's knobs for `meta`; `overrides` must pin `block_kv`."""
    require_tile(meta)
    return arch_module(_ARCH_MODULES, arch).entry_resolve(meta, overrides=overrides)


def plan(arch, meta, *, max_seqlen_k, overrides=None):
    """JIT: `(meta, knobs)` for calls whose sequences are at most `max_seqlen_k` keys long."""
    require_head_dim(meta)
    return arch_module(_ARCH_MODULES, arch).entry_plan(meta, max_seqlen_k=max_seqlen_k, overrides=overrides)


def prefer_fused(arch, **call):
    """Whether the arch's policy picks this kernel over the split path for `call`."""
    if arch not in _ARCH_MODULES:
        return False
    return arch_module(_ARCH_MODULES, arch).prefer_fused(**call)


def arches():
    """The arches this kernel has a row for."""
    return tuple(_ARCH_MODULES)


def builder(arch, meta, knobs):
    """Construct the FlyDSL module; imports flydsl."""
    require_tile(meta)
    return arch_module(_ARCH_MODULES, arch).entry_build(meta, knobs)
