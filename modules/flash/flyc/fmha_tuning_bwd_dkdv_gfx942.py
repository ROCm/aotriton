# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 dK/dV backward: fmha_bwd_dkdv_gfx942.py (head dims 32-128),
fmha_bwd_dkdv_gfx942_wide.py (160-192) and fmha_bwd_dkdv_gfx942_xwide.py (224-512).

flydsl-free: the AOT description calls it from the code generator, which has
no flydsl. The builder takes its defaults from here, so the two cannot drift.
"""

# The compiled head dims; an off-ladder head dim runs padded on the next rung.
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

# (source file, @flyc.kernel name, builder) per body, for the AOT description.
BODIES = {
    'narrow': ('fmha_bwd_dkdv_gfx942.py', 'fmha_bwd_dkdv_gfx942_kernel', 'build_fmha_bwd_dkdv_gfx942'),
    'wide': ('fmha_bwd_dkdv_gfx942_wide.py', 'fmha_bwd_dkdv_gfx942_wide_kernel', 'build_fmha_bwd_dkdv_gfx942_wide'),
    'xwide': ('fmha_bwd_dkdv_gfx942_xwide.py', 'fmha_bwd_dkdv_gfx942_xwide_kernel', 'build_fmha_bwd_dkdv_gfx942_xwide'),
}

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


def body(head_dim):
    """Which body serves a compiled head dim. xwide's role-split teams win from 224
    (B2 H32 S8k dense/causal TFLOPS, wide -> xwide): 224 333/328 -> 448/438,
    256 371/366 -> 464/444; they lose at 192 (446 -> 423) and 160 (418 -> 403)."""
    if head_dim <= 128:
        return 'narrow'
    return 'wide' if head_dim <= 192 else 'xwide'


def default_kv_subs(head_dim):
    """16-key columns per wave (wide) or per team (xwide).
    wide: 2 spills past head dim 224. xwide: a wave holds 0.375 * D * kv_subs
    registers of state; 4 at <= 256 (256 B2 H32 S8k: 464 vs 397 at 2), 2 above."""
    if body(head_dim) == 'xwide':
        return 4 if head_dim <= 256 else 2
    return 2 if head_dim <= 224 else 1


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
    if body(head_dim) == 'wide':
        subs = default_kv_subs(head_dim)
        return dict(kv_subs=subs,
                    waves_per_eu=1,
                    block_kv=block_kv(4, subs),
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
    if body(head_dim) == 'xwide':
        # Two teams of two waves, each team owning 16 * kv_subs keys.
        subs = default_kv_subs(head_dim)
        return dict(kv_subs=subs,
                    waves_per_eu=1,
                    block_kv=block_kv(2, subs),
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
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
        kw.update({k: v for k, v in knob_dict.items() if k not in ('block_kv', 'GRID_AXIS_ORDER')})
        kw['grid_axis_order'] = knob_dict['GRID_AXIS_ORDER']
    source, _, builder = BODIES[body(head_dim)]
    return getattr(importlib.import_module(source[:-3]), builder)(head_dim, **kw)
