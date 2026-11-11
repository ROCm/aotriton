# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 backward dQ/dB kernel (fmha_bwd_dq_gfx942.py).

flydsl-free, like fmha_tuning_gfx942.py: the code generator imports it.
The builder takes its defaults from here, so the two cannot drift.
"""

# Head dims served by fmha_bwd_dq_gfx942.py; wider rungs live in another body.
LADDER = (32, 64, 96, 128)

# (source file, @flyc.kernel name, builder) per body.
BODIES = {
    'narrow': ('fmha_bwd_dq_gfx942.py', 'fmha_bwd_dq_gfx942_kernel', 'build_fmha_bwd_dq_gfx942'),
}

# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1


def default_block_n(head_dim):
    """KV rows per tile: 48 MFMAs per tile at 64 and 128."""
    return 64 if head_dim <= 64 else 32


def default_grid_axis_order(causal):
    """Head-fastest everywhere until csrc/flyc_bwd_dq.cc serves TILE_FASTEST."""
    return GRID_AXIS_HEAD_FASTEST


def default_lpt(causal):
    """Reverse the q tiles (longest first) under causal masks."""
    return causal


def body(head_dim):
    """Which body serves a compiled head dim."""
    return 'narrow'


def rung(hdim_qk, hdim_vo):
    """The compiled head dim serving a call, or None past the ladder."""
    h = max(hdim_qk, hdim_vo)
    return next((r for r in LADDER if r >= h), None)


def build(head_dim, **kwargs):
    """The body's builder; flydsl is imported only here, never at module scope."""
    import importlib

    src, _, name = BODIES[body(head_dim)]
    return getattr(importlib.import_module(src[:-3]), name)(head_dim, **kwargs)


def knobs(head_dim, causal):
    """The build of one functional, as a JSON-able dict. `block_m` (rows per grid
    block) and `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 dQ ladder {LADDER}')
    return dict(block_m=256,
                num_waves=8,
                block_n=default_block_n(head_dim),
                waves_per_eu=2,
                GRID_AXIS_ORDER=default_grid_axis_order(causal))
