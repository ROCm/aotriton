# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuning policy of the gfx942 forward (`fwd.py`): which body serves a rung,
and with which knobs.

flydsl-free: the code generator calls it through `flyc/tuning_fwd.py`, and it
runs with no flydsl. The body builders take their defaults from here, so the
two cannot drift.

Also the head-dim ladder, tile checks and `plan` that the dQ and dK/dV
policies share.
"""

from dataclasses import fields, replace

from fmha_knobs import GRID_AXIS_HEAD_FASTEST, GRID_AXIS_TILE_FASTEST, knobs_of  # noqa: F401  (the kernels read the grid axes here)
from fmha_meta import FmhaInputMetadata
from .knobs import FmhaKnobs

# The compiled head dims, served by three bodies with different register plans
# (fwd.py maps a body to its source):
#   narrow  <= 128: 8 waves x 32 rows, 2 waves/SIMD
#   wide    <= 256: 4 waves x 32 rows, 1 wave/SIMD
#   xwide   > 256:  8 waves x 16 rows, 16x16x16 MFMA
# 384 is JIT-only: AOT builds the shipped ladder (aot/_flyc_common.py).
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)


def rung(hdim_qk, hdim_vo):
    """The compiled head dim serving a call, or None past the ladder."""
    h = max(hdim_qk, hdim_vo)
    return next((r for r in LADDER if r >= h), None)


def check_tile(meta: FmhaInputMetadata):
    """Raise unless `meta`'s tile is a gfx942 rung that serves its head dims."""
    tile = meta.block_dmodel
    if tile not in LADDER:
        raise ValueError(f'block_dmodel {tile} is not on the gfx942 ladder {LADDER}')
    if meta.block_dmodel_v != tile:
        raise ValueError(f'gfx942 serves block_dmodel_v == block_dmodel only, '
                         f'got {meta.block_dmodel_v} != {tile}')
    if meta.head_dim is not None and not meta.padded_head:
        if meta.head_dim != tile or meta.head_dim_v != tile:
            raise ValueError(f'head dims ({meta.head_dim}, {meta.head_dim_v}) on an unpadded {tile} build')


def plan(request: FmhaInputMetadata) -> FmhaInputMetadata:
    """JIT: `request` with the tile serving its real head dims and `padded_head`."""
    tile = rung(request.head_dim, request.head_dim_v)
    if tile is None:
        raise ValueError(f'head dims ({request.head_dim}, {request.head_dim_v}) exceed '
                         f'the gfx942 ladder {LADDER}')
    padded = request.head_dim != tile or request.head_dim_v != tile
    return replace(request, block_dmodel=tile, block_dmodel_v=tile, padded_head=padded)


def resolve_fields(knobs, defaults):
    """`knobs` with every None field taken from the `defaults` dict."""
    return replace(knobs, **{f.name: defaults[f.name] for f in fields(knobs)
                             if getattr(knobs, f.name) is None and f.name in defaults})


def default_block_n(head_dim):
    """Keep 32 MFMAs per KV tile: small heads take a longer tile."""
    return 64 if head_dim <= 64 else 32


def default_grid_axis_order(causal):
    """Tile-fastest shares K/V in L2 across a head's q tiles; causal needs
    head-fastest to spread its uneven tiles, longest first (B2 H64 S32k d128: 523
    vs 495 TFLOPS non-causal; causal d128, head- vs tile-fastest: B4 H32 S8k 407
    vs 294, B1 H32 S32k 457 vs 419)."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def body(head_dim):
    """Which body serves a compiled head dim."""
    if head_dim <= 128:
        return 'narrow'
    return 'wide' if head_dim <= 256 else 'xwide'


def policy(head_dim, causal):
    """The build of one functional, as a dict of knobs. `block_m` (rows per grid
    block) and `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 ladder {LADDER}')
    if body(head_dim) == 'wide':
        # A causal block runs q tiles g and T-1-g, so it covers 2 * 128 rows; the
        # tile-fastest grid keeps K/V in L2 and the pairing evens out block lengths.
        return dict(block_m=256 if causal else 128,
                    num_waves=4,
                    block_n=32,
                    waves_per_eu=1,
                    rescale_threshold=0.0,
                    GRID_AXIS_ORDER=GRID_AXIS_TILE_FASTEST)
    if body(head_dim) == 'xwide':
        return dict(block_m=128,
                    num_waves=8,
                    block_n=16,
                    waves_per_eu=2,
                    rescale_threshold=0.0,
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
    return dict(block_m=256,
                num_waves=8,
                block_n=default_block_n(head_dim),
                waves_per_eu=2,
                rescale_threshold=0.0,
                GRID_AXIS_ORDER=default_grid_axis_order(causal))


def resolve_knobs(meta: FmhaInputMetadata, overrides: FmhaKnobs | None = None) -> FmhaKnobs:
    """The complete configuration for `meta`'s tile; `overrides`' fields win."""
    check_tile(meta)
    k = overrides or FmhaKnobs()
    defaults = dict(policy(meta.block_dmodel, meta.causal), body=body(meta.block_dmodel))
    return resolve_fields(k, defaults).with_flyc_fields()


# Unified entry (flyc/tuning_fwd.py): the shared metadata.


def entry_resolve(meta, *, overrides=None) -> FmhaKnobs:
    return resolve_knobs(meta, knobs_of(FmhaKnobs, overrides))


def entry_plan(meta, *, overrides=None):
    meta = plan(meta)
    return meta, resolve_knobs(meta, knobs_of(FmhaKnobs, overrides))


def entry_build(meta, knobs: FmhaKnobs):
    from .fwd import build_fmha_fwd_gfx942_module_primary
    return build_fmha_fwd_gfx942_module_primary(meta, knobs)
