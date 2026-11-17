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

    # Compile-time widths, baked into the binary. BLOCK_DMODEL is the maximum
    # head_dim the resulting hsaco can serve; the *real* extent travels as a
    # runtime argument and may be smaller, which is what `padded_head` records.
    # They are knobs and not input metadata because only the ladder decides
    # them, and the ladder is this module's.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
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

    # Three floating-point knobs, which looks like two too many until you see
    # that they act at three different levels:
    #
    #   fp_mode        the explicit `arith.FastMathFlags` set put on the
    #                  *softmax* operations -- "fast", "noninf" or "safe".
    #                  This is the one that has been measured; "fast" includes
    #                  `ninf`, which silently deleted the KV tail mask once.
    #   fast_fp_math   the *ambient* fastmath for traced operations that do not
    #                  carry their own flags (`effective_fastmath_hint`).
    #   unsafe_fp_math a ROCm *backend* option, `unsafe-math`, applied to the
    #                  whole compilation (`compiler/backends/rocm.py`).
    #
    # So they are per-op, per-op-default and whole-compilation respectively.
    # Only `fp_mode` is ever varied in practice; the other two have been True
    # for every build ever made here, which means their effect is measured only
    # in combination and collapsing them would need its own experiment.
    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None
    strides_constexpr: bool | None = None

    # Derived, not chosen: true when the caller's head_dim is not itself a
    # compiled tile width, so the kernel computes ceil-to-tile columns and
    # masks the difference. It lives here rather than with the caller's inputs
    # because deciding it needs the ladder -- the set of every width that
    # exists -- which is knowledge only this module has.
    padded_head: bool | None = None
    lpt_tile_order: bool | None = None
    unsafe_no_kv_clamp: bool | None = None
    path_tag: str | None = None



# Defaults for the knobs the policy has no opinion about -- they are the same
# for every shape, so they are literals here rather than functions.

    FLYC_SOURCE = "flash_attn_func_gfx1201_aiw.py"
    FLYC_KERNEL_NAME = "flash_attn_func_aiw_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST


@dataclass(frozen=True)
class BwdDqKnobs(FlycQKnobs):
    """How to compute it. Every field `None` means "policy decides"."""

    # Compile-time width baked into the binary. The *real* extent travels as a
    # runtime argument and may be smaller, which is what `padded_head` records.
    block_dmodel: int | None = None

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

    # Three floating-point knobs acting at three levels; see the forward tuning
    # module's note. Only `fp_mode` is ever varied in practice.
    #
    # "noninf" is load-bearing here for the same reason it is there and one
    # more: this kernel writes -inf into the scores of masked columns *and*
    # reads a +inf logsumexp for rows the forward found no keys for. `ninf`
    # licenses the compiler to assume neither exists.
    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None

    padded_head: bool | None = None
    lpt_tile_order: bool | None = None

    FLYC_SOURCE = "fmha_bwd_dq_gfx1201_kernel.py"
    FLYC_KERNEL_NAME = "bwd_dq_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_HEAD_FASTEST


@dataclass(frozen=True)
class BwdDkDvKnobs(FlycKVKnobs):
    """How to compute it. Every field `None` means "policy decides"."""

    # Compile-time widths baked into the binary; the real extents ride along as
    # runtime `hdim_qk` / `hdim_vo` arguments and `padded_head` records whether
    # they differ.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None

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

    # Function attributes and floating-point latitude. Same three-level split
    # as the forward's: `fp_mode` is the explicit flag set on the softmax-ish
    # arithmetic, `fast_fp_math` the ambient default, `unsafe_fp_math` a
    # whole-compilation backend option.
    waves_per_eu: int | None = None
    sched_strategy: str | None = None
    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None

    # Whether the Q/dO address hoists `row * stride_seq` out of the loop. See
    # `kv_off` in `fmha_common_gfx1201.make_addr_pair`.
    addr_hoist: bool | None = None

    # Derived, not chosen: true when the caller's head_dim is not itself a
    # compiled tile width.
    padded_head: bool | None = None

    FLYC_SOURCE = "fmha_bwd_dkdv_gfx1201_kernel.py"
    FLYC_KERNEL_NAME = "bwd_dkdv_kernel"
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_TILE_FASTEST
