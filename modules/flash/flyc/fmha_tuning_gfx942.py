# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 forward (fmha_fwd_gfx942.py).

flydsl-free: aot/flyc_attn_fwd.py calls it from the code generator, which has
no flydsl. The builder takes its defaults from here, so the two cannot drift.
"""

# The compiled head dims. Wider heads need a different register plan.
LADDER = (32, 64, 96, 128)

NUM_WAVES = 8
WAVE_M = 32          # query rows per wave
BLOCK_M = NUM_WAVES * WAVE_M

# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1


def default_block_n(head_dim):
    """Keep 32 MFMAs per KV tile: small heads take a longer tile."""
    return 64 if head_dim <= 64 else 32


def default_grid_axis_order(causal):
    """Tile-fastest shares K/V in L2 across a head's q tiles; causal needs
    head-fastest to spread its uneven tiles (B2 H64 S32k d128: 523 vs 495 TFLOPS
    non-causal; B4 H32 S8k d128 causal: 402 vs 295)."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def knobs(head_dim, causal):
    """The build of one functional, as a JSON-able dict. `block_m` and
    `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 ladder {LADDER}')
    return dict(block_m=BLOCK_M,
                num_waves=NUM_WAVES,
                block_n=default_block_n(head_dim),
                waves_per_eu=2,
                rescale_threshold=0.0,
                GRID_AXIS_ORDER=default_grid_axis_order(causal))
