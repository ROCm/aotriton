# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuning policy of the gfx942 dK/dV backward (`dkdv.py`): bodies narrow
(head dims 32-128), wide (160-192), xwide (224-256) and xxwide (384-512).

flydsl-free: the code generator imports it. The body builders take their
defaults from here, so the two cannot drift.
"""

from fmha_knobs import GRID_AXIS_HEAD_FASTEST, GRID_AXIS_TILE_FASTEST, knobs_of  # noqa: F401  (the kernels read the grid axes here)
from fmha_meta import FmhaInputMetadata
from .knobs import MASK_PROB, MASK_SCORE, P_DROP_AT_STORE, P_DROP_PER_ELEM, BwdDkDvKnobs
from .tuning_fwd import LADDER, check_tile, plan, resolve_fields, rung  # noqa: F401  (re-exported for the kernels and benches)

# Rows of one 16x16x16 MFMA block: the q tile per loop step and the kv unit.
BLOCK_ROWS = 16


def default_num_waves(head_dim):
    return 4


def default_kv_blocks(head_dim):
    """16-row kv blocks each wave keeps resident: K, V, dK^T and dV^T cost
    3 * 16 * kv_blocks * head_dim / 64 registers. 2 is best or tied at every rung
    (B2 H32 S8k); 3 and 4 run out of registers at d128 and lose at d64 too."""
    return 2


def default_waves_per_eu(head_dim, bias=False):
    """One wave per SIMD: at d128 the resident set alone is 192 of 256 registers,
    and two waves per SIMD at d64 measured the same. d96 takes two except with bias.
    B2 H32 d96, ms, one vs two waves:
        dense S4k 2.01 / 1.74, S8k 7.82 / 6.52; dropout 0.2 S8k 11.0 / 9.0
        causal S2k 0.32 / 0.40, S4k 1.08 / 1.17, S8k 3.84 / 3.61, S16k 14.9 / 13.1
        bias S4k 2.64 / 6.64, S8k 10.2 / 26.4
    Causal flips with seqlen (an input-time choice); two win where time is spent."""
    return 2 if head_dim == 96 and not bias else 1


def default_interleave_valu(head_dim):
    """VALU ops the scheduler is asked to place after each S/dP MFMA; 0 leaves it free."""
    return 3


def default_grid_axis_order(causal):
    """Causal: head-fastest launches every head's long low-kv tiles first (B2 H32 S8k
    d128: 434 vs 331 TFLOPS, d192: 431 vs 336); otherwise tile-fastest shares Q/dO
    in L2 (448 vs 438)."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def body(head_dim):
    """Which body serves a compiled head dim. xwide's role-split teams win from 224
    (B2 H32 S8k dense/causal TFLOPS, wide -> xwide): 224 333/328 -> 448/438,
    256 371/366 -> 464/444; they lose at 192 (446 -> 423) and 160 (418 -> 403).
    xxwide's teams also shard the contraction and win from 384 (xwide -> xxwide,
    dense/causal/dropout/bias): 384 436/417/398/412 -> 452/420/405/414,
    512 341/226/217/369 -> 423/390/372/382."""
    if head_dim <= 128:
        return 'narrow'
    if head_dim <= 192:
        return 'wide'
    return 'xwide' if head_dim <= 256 else 'xxwide'


def default_kv_subs(head_dim):
    """16-key columns per wave (wide) or per team (xwide, xxwide).
    xwide: a wave holds 0.375 * D * kv_subs registers of state; 4 at 256 (B2 H32
    S8k: 464 vs 397 at 2). xxwide: 0.1875 * D * kv_subs; the most that does not
    spill: 4 at 384 (452 vs 418 at 3), 3 at 512 (4 spills 462; 3: 423, 2: 390)."""
    if body(head_dim) == 'xwide':
        return 4 if head_dim <= 256 else 2
    if body(head_dim) == 'xxwide':
        return 4 if head_dim <= 384 else 3
    return 2


def body_kv_subs(body_name, head_dim):
    """kv_subs of a `body_name` build by default: the policy's on its own rungs, else 2.
    wide keeps 2: 32 keys halve the LDS and global traffic per FLOP, at 1.5 * D
    registers (which spills past 224)."""
    return default_kv_subs(head_dim) if body(head_dim) == body_name else 2


# Per body; dropout_scale_at by A/B at B4 H32 bf16, dropout 0.2.
_MASK_AT = {'narrow': MASK_SCORE, 'wide': MASK_PROB, 'xwide': MASK_PROB, 'xxwide': MASK_SCORE}
_DROPOUT_SCALE_AT = {'narrow': P_DROP_AT_STORE,
                     'wide': P_DROP_AT_STORE,
                     'xwide': P_DROP_AT_STORE,
                     'xxwide': P_DROP_PER_ELEM}


def default_mask_at(body_name):
    """Where `body_name`'s band mask lands: the score before exp2 or P after it."""
    return _MASK_AT[body_name]


def default_dropout_scale_at(body_name):
    """Where `body_name` applies dropout's 1/(1-p) to P for dV: per element or at the dV store."""
    return _DROPOUT_SCALE_AT[body_name]


def block_kv(num_waves, kv_blocks):
    """KV rows per workgroup: what the grid's tile axis divides max_seqlen_k by."""
    return num_waves * kv_blocks * BLOCK_ROWS


def policy(head_dim, causal, bias=False):
    """The build of one functional, as a dict of knobs: the builder's keyword
    arguments plus `block_kv` and `GRID_AXIS_ORDER`, which the C++ grid_calculator
    reads. `GRID_AXIS_ORDER` is passed to the builder as `grid_axis_order`."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 dK/dV ladder {LADDER}')
    name = body(head_dim)
    return dict(_body_policy(name, head_dim, causal, bias),
                mask_at=default_mask_at(name),
                dropout_scale_at=default_dropout_scale_at(name))


def _body_policy(name, head_dim, causal, bias):
    """`policy`'s body-specific knobs."""
    if name == 'wide':
        subs = default_kv_subs(head_dim)
        return dict(kv_subs=subs,
                    waves_per_eu=1,
                    block_kv=block_kv(4, subs),
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
    if name == 'xwide':
        # Two teams of two waves, each team owning 16 * kv_subs keys.
        subs = default_kv_subs(head_dim)
        return dict(kv_subs=subs,
                    waves_per_eu=1,
                    block_kv=block_kv(2, subs),
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
    if name == 'xxwide':
        # One team of four waves owning 16 * kv_subs keys.
        subs = default_kv_subs(head_dim)
        return dict(kv_subs=subs,
                    waves_per_eu=1,
                    block_kv=block_kv(1, subs),
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
    nw = default_num_waves(head_dim)
    kvb = default_kv_blocks(head_dim)
    return dict(num_waves=nw,
                kv_blocks=kvb,
                waves_per_eu=default_waves_per_eu(head_dim, bias),
                interleave_valu=default_interleave_valu(head_dim),
                block_kv=block_kv(nw, kvb),
                GRID_AXIS_ORDER=default_grid_axis_order(causal))


def resolve_knobs(meta: FmhaInputMetadata, overrides: BwdDkDvKnobs | None = None) -> BwdDkDvKnobs:
    """The complete configuration for `meta`'s tile; `overrides`' fields win."""
    check_tile(meta)
    k = overrides or BwdDkDvKnobs()
    defaults = dict(policy(meta.block_dmodel, meta.causal, meta.bias), body=body(meta.block_dmodel))
    return resolve_fields(k, defaults).with_flyc_fields()


# Unified entry (flyc/tuning_dkdv.py): the shared metadata.


def entry_resolve(meta, *, overrides=None) -> BwdDkDvKnobs:
    return resolve_knobs(meta, knobs_of(BwdDkDvKnobs, overrides))


def entry_plan(meta, *, overrides=None):
    meta = plan(meta)
    return meta, resolve_knobs(meta, knobs_of(BwdDkDvKnobs, overrides))


def entry_build(meta, knobs: BwdDkDvKnobs):
    from .dkdv import build_fmha_bwd_dkdv_gfx942_module_primary
    return build_fmha_bwd_dkdv_gfx942_module_primary(meta, knobs)
