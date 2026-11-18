# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Knob classes of the gfx1201 attention kernels (forward, dQ, dK/dV).

Plain data and flydsl-free: the code generator imports them to record the
resolved knobs. The policy that fills them is in the `fmha_tuning_*_gfx1201`
modules.
"""

from dataclasses import dataclass

from fmha_knobs import (
    GRID_AXIS_HEAD_FASTEST,
    GRID_AXIS_TILE_FASTEST,
    FlycKVKnobs,
    FlycQKnobs,
)

__all__ = ["BwdDkDvKnobs", "BwdDqKnobs", "FmhaKnobs"]


@dataclass(frozen=True)
class FmhaKnobs(FlycQKnobs):
    """How to compute it. Every field `None` means "policy decides".

    `None` rather than a literal default on purpose: it is the only way
    `resolve_knobs` can tell "the caller wants 1" from "the caller did not
    say", and that difference is the whole point of the overrides argument.
    """

    d_offset: int | None = None

    block_n: int | None = None
    k_prefetch_dist: int | None = None
    v_prefetch_dist: int | None = None
    v_lds_layout: str | None = None
    row_subtiles: int | None = None
    waves_per_eu: int | None = None
    kv_addr_hoist: bool | None = None

    # The two knobs `resolve_knobs` cannot fill. Both are derived from
    # NUM_WAVES, which the builder computes from the shard count and the tile
    # geometry, so filling them here would mean duplicating that derivation.
    # `None` means "derive it"; every other field is guaranteed resolved.
    flat_work_group_size: int | None = None
    shards: int | None = None
    sched_strategy: str | None = None

    # The explicit `arith.FastMathFlags` set put on the *softmax* operations
    # -- "fast", "noninf" or "safe". "fast" includes `ninf`, which silently
    # deleted the KV tail mask once. The whole-compilation options are build
    # options, not knobs: `fmha_utils.COMPILE_HINTS`.
    fp_mode: str | None = None
    strides_constexpr: bool | None = None

    lpt_tile_order: bool | None = None
    unsafe_no_kv_clamp: bool | None = None
    path_tag: str | None = None



# Defaults for the knobs the policy has no opinion about -- they are the same
# for every shape, so they are literals here rather than functions.

    FLYC_SOURCE = "gfx1201/fwd.py"
    FLYC_KERNEL_NAME = "flash_attn_func_aiw_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST


@dataclass(frozen=True)
class BwdDqKnobs(FlycQKnobs):
    """How to compute it. Every field `None` means "policy decides"."""

    block_n: int | None = None
    num_waves: int | None = None
    # Waves cooperating on one Q row sub-tile. `num_waves // shards` Q row
    # sub-tiles per workgroup, hence `block_m = 16 * num_waves // shards`.
    shards: int | None = None
    kt_lds_layout: str | None = None
    kv_addr_hoist: bool | None = None
    kv_prefetch_dist: int | None = None

    waves_per_eu: int | None = None
    flat_work_group_size: int | None = None
    sched_strategy: str | None = None

    # Per-op fastmath flags; see `FmhaKnobs.fp_mode`.
    #
    # "noninf" is load-bearing here for the same reason it is there and one
    # more: this kernel writes -inf into the scores of masked columns *and*
    # reads a +inf logsumexp for rows the forward found no keys for. `ninf`
    # licenses the compiler to assume neither exists.
    fp_mode: str | None = None

    lpt_tile_order: bool | None = None

    FLYC_SOURCE = "gfx1201/dq.py"
    FLYC_KERNEL_NAME = "bwd_dq_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST


@dataclass(frozen=True)
class BwdDkDvKnobs(FlycKVKnobs):
    """How to compute it. Every field `None` means "policy decides"."""

    block_m: int | None = None
    num_waves: int | None = None
    lpt_tile_order: bool | None = None
    split_head_dim: bool | None = None

    # Read only by `fmha_bwd_dkdv512_gfx1201_kernel`, the wide-head_dim variant.
    # They live here rather than in a parallel dataclass so that `plan()` and
    # `resolve_knobs` serve both kernels; the primary kernel ignores them, and
    # their defaults are what it would have done anyway.
    #
    #   contraction_shards  ways the S / dP contraction is split across a team.
    #                       1 is the primary kernel's role-only pair.
    #   transposed_source   "tile" stages Q^T/dO^T in LDS; "derived" reads the
    #                       same operands strided out of the row-major tiles and
    #                       does not allocate them.
    contraction_shards: int | None = None
    transposed_source: str | None = None

    # Function attributes and floating-point latitude. `fp_mode` is the
    # forward's: the explicit flag set on the softmax-ish arithmetic.
    waves_per_eu: int | None = None
    sched_strategy: str | None = None
    fp_mode: str | None = None

    # Whether the Q/dO address hoists `row * stride_seq` out of the loop. See
    # `kv_off` in `utils.make_addr_pair`.
    addr_hoist: bool | None = None

    FLYC_SOURCE = "gfx1201/dkdv.py"
    FLYC_KERNEL_NAME = "bwd_dkdv_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_TILE_FASTEST
