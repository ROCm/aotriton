# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuning policy of the gfx942 backward dQ/dB kernel (`dq.py`): a narrow body
for head dims 32-128 and a wide one for 160-512.

flydsl-free, like `tuning_fwd.py`: the code generator imports it. The body
builders take their defaults from here, so the two cannot drift.
"""

from fmha_knobs import GRID_AXIS_HEAD_FASTEST, GRID_AXIS_TILE_FASTEST, knobs_of  # noqa: F401  (the kernels read the grid axes here)
from fmha_meta import FmhaInputMetadata
from .knobs import BwdDqKnobs
from .tuning_fwd import LADDER, check_tile, plan, resolve_fields, rung  # noqa: F401  (re-exported for the kernels and benches)


def default_block_n(head_dim):
    """KV rows per tile: 48 MFMAs per tile at 64 and 128."""
    return 64 if head_dim <= 64 else 32


def default_grid_axis_order(causal):
    """Head-fastest: tile-fastest measured no better (B2 H32 S8k d128: 518 either
    way; causal 315 vs 444)."""
    return GRID_AXIS_HEAD_FASTEST


def default_lpt(causal):
    """Longest q tiles first under causal masks (B2 H32 S8k causal: d128 444 -> 471,
    d64 367 -> 418 TFLOPS)."""
    return causal


def body(head_dim):
    """Which body serves a compiled head dim."""
    return 'narrow' if head_dim <= 128 else 'wide'


def policy(head_dim, causal):
    """The build of one functional, as a dict of knobs. `block_m` (rows per grid
    block) and `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 dQ ladder {LADDER}')
    if body(head_dim) == 'wide':
        # 32 query rows per wave up to 256, 16 above; a causal block pairs two q tiles.
        wave_m = 32 if head_dim <= 256 else 16
        return dict(block_m=4 * wave_m * (2 if causal else 1),
                    wave_m=wave_m,
                    num_waves=4,
                    block_n=wave_m,
                    waves_per_eu=1,
                    GRID_AXIS_ORDER=GRID_AXIS_TILE_FASTEST)
    return dict(block_m=256,
                num_waves=8,
                block_n=default_block_n(head_dim),
                waves_per_eu=2,
                lpt=default_lpt(causal),
                GRID_AXIS_ORDER=default_grid_axis_order(causal))


def resolve_knobs(meta: FmhaInputMetadata, overrides: BwdDqKnobs | None = None) -> BwdDqKnobs:
    """The complete configuration for `meta`'s tile; `overrides`' fields win."""
    check_tile(meta)
    k = overrides or BwdDqKnobs()
    defaults = dict(policy(meta.block_dmodel, meta.causal), body=body(meta.block_dmodel))
    return resolve_fields(k, defaults).with_flyc_fields()


# Unified entry (flyc/tuning_dq.py): the shared metadata.


def entry_resolve(meta, *, overrides=None) -> BwdDqKnobs:
    return resolve_knobs(meta, knobs_of(BwdDqKnobs, overrides))


def entry_plan(meta, *, overrides=None):
    meta = plan(meta)
    return meta, resolve_knobs(meta, knobs_of(BwdDqKnobs, overrides))


def entry_build(meta, knobs: BwdDqKnobs):
    from .dq import build_fmha_bwd_dq_gfx942_module_primary
    return build_fmha_bwd_dq_gfx942_module_primary(meta, knobs)
