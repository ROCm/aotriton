# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Knob fields of the gfx950 attention kernels (forward, dQ, dK/dV, fused backward).

Plain data and flydsl-free: the code generator imports them to record the
resolved knobs. The tuning modules (`fmha_tuning_*gfx950.py`) subclass these
with the policy that fills them (`resolve`, the geometry steps, `build_traits`).
"""

from dataclasses import dataclass

from fmha_knobs import (
    GRID_AXIS_HEAD_FASTEST,
    FlycKVKnobs,
    FlycQKnobs,
)

__all__ = ["BwdDkDvKnobFields", "BwdDqKnobFields", "BwdFusedKnobFields", "Gfx950KnobFields"]


@dataclass(frozen=True)
class Gfx950KnobFields(FlycQKnobs):
    """The forward's knobs, which dQ extends. `None` means "policy decides".

    `None` rather than a literal default on purpose: it is the only way
    `resolve` can tell "the caller wants 1" from "the caller did not say".
    """

    # A compile-time *lower* bound on the runtime `hdim_qk`, exclusive. The
    # build promises nothing about hdim at or below it, and the dispatcher
    # rejects a call that violates it, so the kernel may treat D columns below
    # it as real data and skip masking them.
    #
    # Derived, not a taste knob, but pinnable to 0 to force the old
    # mask-everything behaviour if a floor is ever suspected of hiding a bug.
    hdim_qk_floor: int | None = None

    # There is no `hdim_mode`. A `"runtime_qk_steps"` mode used to be declared
    # here -- shorten the QK reduction to `ceil(hdim_qk/16)` MFMA K-steps at
    # runtime, against `"zero_fill"`'s tile-shaped count -- and it was **never
    # implemented**: the flag threaded through three files into
    # `ctx.RUNTIME_QK_STEPS`, which nothing read. The two modes produced
    # byte-identical ISA.
    #
    # Deleted rather than implemented, because measurement says it was aimed at
    # the wrong term. At head_dim 32 in the 64 tile the padded build has the
    # same 96 MFMAs as an unpadded 64 -- the 2x waste is real -- but it also has
    # 565 `v_cndmask` against 145 and 3584 instructions against 2562. The
    # masking, not the MFMA count, is what makes head_dim 32 *slower in
    # wall-clock* than head_dim 64 (185 us against 155). Shortening QK would cut
    # 8 of the 16 MFMAs per tile and leave that untouched.
    #
    # The fix that would matter is a tile that does not pad: `PV_MFMA_N` is 32
    # and `K_STEP_QK` is 16, so rungs at multiples of 32 are legal in principle
    # and need only granule-32 staging (family S).

    # Whether the strides fold to literals. False is the parity ABI; True
    # exists so the fast path can be shown to emit unchanged code.
    strides_constexpr: bool | None = None

    return_lse: bool | None = None

    # Dual-wave schedule.
    waves_per_eu: int | None = None
    lazy_rescale: bool | None = None
    setprio: bool | None = None
    stagger: bool | None = None
    lpt_tile_order: bool | None = None
    # End the KV walk at the last live tile; see `EXACT_KV_TILES`. A speed
    # trade by sequence length, not accuracy: +69% at seqlen_k 64 and +33% at
    # 128, but 1.5-2% slower from four tiles up, so off unless `resolve` (or a
    # caller) opts a shape in.
    exact_kv_tiles: bool | None = None

    # Problem modes. Ordinary fields; see the class docstring.
    paged: bool | None = None
    kv_cache_layout: str | None = None
    num_kv_splits: int | None = None

    # Wave geometry. `None` means "the family for this tile width", which is
    # what `_with_wave_geometry` decides. Pinnable so a sweep can cross a
    # family boundary without editing the table.
    num_waves: int | None = None
    block_n: int | None = None
    head_dim_granule: int | None = None

    # GQA head packing: q heads of one GQA group per workgroup, with
    # `block_m` then meaning the Q tile *per head*. Not one of the four
    # geometry fields -- those decide the KV staging, which this leaves
    # alone -- and not derived by policy either: whether it is legal depends
    # on the *call's* group size, which no build knows, so the dispatcher picks
    # it (see `ParityKernelContext._pack_q_heads`) and the host rejects a group
    # it does not divide.
    q_heads_per_wg: int | None = None

    # The two D-axis splits. Deliberately *not* part of the four-field geometry
    # pinning rule above: those four are one indivisible choice (a family),
    # while these are per-width policy that a sweep wants to vary on its own.
    # `None` means "the width decides", as everywhere else here.
    d_stages: int | None = None
    qk_shards: int | None = None
    vo_shards: int | None = None

    # The four V-layout constants whose formula family A cannot pin down; see
    # `make_traits`. Exposed so a sweep can settle them by measurement instead
    # of picking one. `None` keeps the default formula.
    v_half_wave: int | None = None
    v_n_group: int | None = None
    v_k_substep: int | None = None
    v_dc_in_pair: int | None = None

    FLYC_SOURCE = "gfx950/fwd.py"
    FLYC_KERNEL_NAME = "flash_attn_func_gfx950_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST


@dataclass(frozen=True)
class BwdDqKnobFields(Gfx950KnobFields):
    """dQ's build axes, on top of the forward's knobs."""

    store_db: bool | None = None

    # 32 or 16. The MFMA's N extent, which is the query rows one wave owns --
    # `_with_wave_geometry` turns it into the shape and the KV tile together,
    # because at 16 rows `BLOCK_N` is not free.
    mfma_rows: int | None = None

    FLYC_SOURCE = "gfx950/dq.py"
    FLYC_KERNEL_NAME = "fmha_bwd_dq_gfx950_kernel"


@dataclass(frozen=True)
class BwdDkDvKnobFields(FlycKVKnobs):
    """The dK/dV knobs. `None` means "policy decides". The head here is the kv head."""

    hdim_qk_floor: int | None = None

    # Pinned as a set of four; see `_with_geometry`.
    num_waves: int | None = None
    block_q: int | None = None
    head_dim_granule: int | None = None

    # Per-width policy a sweep wants to vary on its own, so not part of the
    # four-field set above.
    dkv_shards: int | None = None
    mfma_rows: int | None = None
    num_stream_buffers: int | None = None
    waves_per_eu: int | None = None
    tight_registers: bool | None = None

    strides_constexpr: bool | None = None

    FLYC_SOURCE = "gfx950/dkdv.py"
    FLYC_KERNEL_NAME = "fmha_bwd_dkdv_gfx950_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST


@dataclass(frozen=True)
class BwdFusedKnobFields(FlycKVKnobs):
    """The fused backward's knobs. `None` means "policy decides".

    `block_kv` is the KV rung, which one workgroup owns whole; the rest is the
    dK/dV body's geometry, which the fused kernel is.
    """

    hdim_qk_floor: int | None = None

    mfma_rows: int | None = None
    num_waves: int | None = None
    block_q: int | None = None
    head_dim_granule: int | None = None
    num_stream_buffers: int | None = None
    waves_per_eu: int | None = None
    # The 32-row body's register-against-ILP switch; see
    # `BwdDkDvKnobFields.tight_registers`. Inert in the 16-row family.
    tight_registers: bool | None = None
    # Rows of `-log2e*LSE` and `delta` the LDS row arrays hold: the build's
    # `seqlen_q` ceiling, rounded up by `_with_rows_cap`.
    q_rows_cap: int | None = None

    strides_constexpr: bool | None = None

    FLYC_SOURCE = "gfx950/fused_bwd.py"
    FLYC_KERNEL_NAME = "fmha_bwd_fused_gfx950_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST
