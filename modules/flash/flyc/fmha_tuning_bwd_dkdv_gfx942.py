# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 dK/dV backward (fmha_bwd_dkdv_gfx942.py), head dims 32-128.

flydsl-free: the AOT description calls it from the code generator, which has
no flydsl. The builder takes its defaults from here, so the two cannot drift.
"""

# The compiled head dims this body serves; an off-ladder head dim runs padded
# on the next rung.
LADDER = (32, 64, 96, 128)

# Source file and @flyc.kernel name, for the AOT description.
SOURCE = 'fmha_bwd_dkdv_gfx942.py'
KERNEL_NAME = 'fmha_bwd_dkdv_gfx942_kernel'

# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1

# Rows of one 16x16x16 MFMA block: the q tile per loop step and the kv unit.
BLOCK_ROWS = 16


def default_num_waves(head_dim):
    return 4


def default_kv_blocks(head_dim):
    """16-row kv blocks each wave keeps resident: K, V, dK^T and dV^T cost
    3 * 16 * kv_blocks * head_dim / 64 VGPRs, about 192 at the default."""
    return {32: 4, 64: 4, 96: 2, 128: 2}[head_dim]


def default_waves_per_eu(head_dim):
    return 2


def default_grid_axis_order(causal):
    return GRID_AXIS_TILE_FASTEST


def block_kv(num_waves, kv_blocks):
    """KV rows per workgroup: what the grid's tile axis divides max_seqlen_k by."""
    return num_waves * kv_blocks * BLOCK_ROWS


def rung(hdim_qk, hdim_vo):
    """The compiled head dim serving a call, or None past the ladder."""
    h = max(hdim_qk, hdim_vo)
    return next((r for r in LADDER if r >= h), None)


def knobs(head_dim, causal):
    """The build of one functional, as a JSON-able dict. `block_kv` and
    `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 dK/dV ladder {LADDER}')
    nw = default_num_waves(head_dim)
    kvb = default_kv_blocks(head_dim)
    return dict(num_waves=nw,
                kv_blocks=kvb,
                waves_per_eu=default_waves_per_eu(head_dim),
                block_kv=block_kv(nw, kvb),
                GRID_AXIS_ORDER=default_grid_axis_order(causal))
