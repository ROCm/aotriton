# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 dK/dV backward (fmha_bwd_dkdv_gfx942.py), head dims 32-128.

flydsl-free: the AOT description calls it from the code generator, which has
no flydsl. The builder takes its defaults from here, so the two cannot drift.
"""

# The compiled head dims this body serves; an off-ladder head dim runs padded
# on the next rung.
LADDER = (32, 64, 96, 128)

# Source file, @flyc.kernel name and builder, for the AOT description.
SOURCE = 'fmha_bwd_dkdv_gfx942.py'
KERNEL_NAME = 'fmha_bwd_dkdv_gfx942_kernel'
BUILDER = 'build_fmha_bwd_dkdv_gfx942'

# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1

# Rows of one 16x16x16 MFMA block: the q tile per loop step and the kv unit.
BLOCK_ROWS = 16


def default_num_waves(head_dim):
    return 4


def default_kv_blocks(head_dim):
    """16-row kv blocks each wave keeps resident: K, V, dK^T and dV^T cost
    3 * 16 * kv_blocks * head_dim / 64 registers. 2 is best or tied at every rung
    (B2 H32 S8k); 3 and 4 run out of registers at d128 and lose at d64 too."""
    return 2


def default_waves_per_eu(head_dim):
    """One wave per SIMD: at d128 the resident set alone is 192 of 256 registers,
    and two waves per SIMD at d64 measured the same."""
    return 1


def default_interleave_valu(head_dim):
    """VALU ops the scheduler is asked to place after each S/dP MFMA; 0 leaves it free."""
    return 3


def default_grid_axis_order(causal):
    """Causal: head-fastest launches every head's long low-kv tiles first (B2 H32 S8k
    d128: 434 vs 331 TFLOPS); otherwise tile-fastest shares Q/dO in L2 (448 vs 438)."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def block_kv(num_waves, kv_blocks):
    """KV rows per workgroup: what the grid's tile axis divides max_seqlen_k by."""
    return num_waves * kv_blocks * BLOCK_ROWS


def rung(hdim_qk, hdim_vo):
    """The compiled head dim serving a call, or None past the ladder."""
    h = max(hdim_qk, hdim_vo)
    return next((r for r in LADDER if r >= h), None)


def knobs(head_dim, causal):
    """The build of one functional, as a JSON-able dict: the builder's keyword
    arguments plus `block_kv` and `GRID_AXIS_ORDER`, which the C++ grid_calculator
    reads. `GRID_AXIS_ORDER` is passed to the builder as `grid_axis_order`."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 dK/dV ladder {LADDER}')
    nw = default_num_waves(head_dim)
    kvb = default_kv_blocks(head_dim)
    return dict(num_waves=nw,
                kv_blocks=kvb,
                waves_per_eu=default_waves_per_eu(head_dim),
                interleave_valu=default_interleave_valu(head_dim),
                block_kv=block_kv(nw, kvb),
                GRID_AXIS_ORDER=default_grid_axis_order(causal))


def build(head_dim, knob_dict=None, **kwargs):
    """The builder, with `knob_dict` (a `knobs()` result) applied; flydsl is imported
    only here, never at module scope."""
    import importlib

    kw = dict(kwargs)
    if knob_dict is not None:
        kw.update({k: knob_dict[k] for k in ('num_waves', 'kv_blocks', 'waves_per_eu', 'interleave_valu')})
        kw['grid_axis_order'] = knob_dict['GRID_AXIS_ORDER']
    return getattr(importlib.import_module(SOURCE[:-3]), BUILDER)(head_dim, **kw)
