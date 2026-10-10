# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 backward-dQ kernel: which knobs, for which shape.

Split out of ``fmha_bwd_dq_gfx1201_kernel.py`` the way
``fmha_tuning_gfx1201.py`` is split out of the forward kernel, and for the same
reason: the kernel file is about correctness and this one is about speed. They
change on different evidence -- a number here moves when a sweep says so, and
nothing here can make a build *wrong*, only slow.

It imports nothing from ``flydsl``, so it is trivially stable-only and can be
read (or edited) without a GPU in the room. The forward tuning module has the
same property; keep it.

**Lightly tuned.** Four knobs have been swept, all at B=1 H=8 N=4096 f16 in
both masking modes: the wave count and ``kt_lds_layout`` on a single
un-interleaved run, and ``shards`` and ``sched_strategy`` interleaved over
3 reps at the two sharded widths. Those four tables are measurements.
``kv_addr_hoist`` is an unswept default and says so at its definition; so is
``block_n``, except at head_dim 512 where it is arithmetic rather than tuning.
Treat the rest as hypotheses.

One caveat on the un-interleaved numbers: they come from one run each, and
``sdpa_lore_gfx1201.md`` records that this board drifts about 5% -- measured
here at up to 15% between whole-script runs of the *same* build -- so
alternatives closer than that are not separated. The effects kept below are
much larger than that where they matter at all.
"""

from dataclasses import dataclass, fields, replace

# ---------------------------------------------------------------------------
# The compiled tile widths.
#
# The ladder now matches the forward's top. A dQ wave carries, per lane and per
# head-dim shard:
#
#   q packs    slice / 4  VGPRs   (slice/16 operands of v8f16)
#   dO packs   slice / 4  VGPRs
#   dq accs    slice / 2  VGPRs   (slice/16 accumulators of v8f32)
#
# with `slice = head_dim / shards`, i.e. `slice` VGPRs before the S and dP
# accumulators, the addressing, and the LDS staging registers. Unsharded that
# is `head_dim`, which is why 256 already sits exactly on the 256-VGPR wall
# (measured: 256 VGPRs, 132 bytes of scratch, 133 spill instructions) and why
# 512 -- which would need 544 -- is not merely slow but inexpressible.
#
# `shards` is the whole answer. It is the same mechanism the forward uses and
# it applies more cleanly here, because GEMM1's reduction axis and GEMM3's
# output axis are *the same axis*: a wave that reduces over head-dim columns
# [s*slice, (s+1)*slice) also owns exactly those columns of dQ, so one offset
# drives Q, dO, K, V, K^T and the output store, and the shards' output stores
# are disjoint. The price is a cross-shard LDS reduction of the partial S and
# dP -- twice the forward's, which reduces S only.
# ---------------------------------------------------------------------------

_BLOCK_DMODEL_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512)

MAX_HEAD_DIM = _BLOCK_DMODEL_LADDER[-1]


def _round_to_ladder(head_dim: int) -> int:
    """Smallest compiled tile width covering `head_dim`."""
    for w in _BLOCK_DMODEL_LADDER:
        if w >= head_dim:
            return w
    raise ValueError(f"head_dim {head_dim} exceeds the largest compiled tile ({MAX_HEAD_DIM})")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Q rows a single wave owns. Fixed by the WMMA tile, not a choice: one 16x16
# WMMA per (Q row block, KV column block), and this kernel does not implement
# the forward's ROW_SUBTILES knob.
_ROWS_PER_WAVE = 16

# Waves per workgroup, i.e. BLOCK_M / 16.
#
# The trade is *not* the forward's. Here a wave's register cost does not depend
# on the wave count at all -- every wave owns 16 Q rows whatever BLOCK_M is --
# so more waves is purely: fewer workgroups, hence less K/V global traffic and
# fewer LDS stagings per KV tile, against more waves sharing one workgroup's
# LDS bandwidth and a coarser Q tail. The measured curve is accordingly much
# flatter than the forward's, with two cliffs.
#
# TFLOPS, B=1 H=8 N=4096 f16, non-causal / causal:
#
#   head_dim    w2           w4           w8           w16          chosen
#   16       51.3/34.5    48.6/38.9    53.9/45.5    54.0/51.2       16
#   32       62.7/62.4    70.6/60.7    75.5/78.5    70.2/66.8        8
#   48          --        79.6/71.7    82.9/83.3    77.2/65.8        8
#   64       75.6/74.5    85.6/89.9    89.8/88.0    83.2/74.3        8
#   80          --        86.1/80.5    92.8/86.5    94.4/86.4        8
#   96          --        86.6/83.9    90.2/87.4    85.6/78.3        8
#   128      35.8/39.2    93.9/90.8    99.5/93.2    94.5/83.7        8
#   160         --        50.0/52.2    90.5/84.1    87.3/76.8        8
#   192      41.4/42.0    75.3/81.8    26.7/23.1    92.8/83.6       16
#   224         --        46.4/44.4    26.0/20.8    91.2/44.8       16
#   256      13.9/15.2    39.9/40.6    65.0/58.8    72.0/60.9       16
#
# The two collapses -- 128 at w2, and 192/224 at w8 -- are three to four times
# slower than their neighbours, far outside the board's drift, and they are the
# reason this is a table rather than a constant. They are almost certainly
# spills (the 192/224 pair spills at 8 waves and *recovers* at 16, which is the
# same non-monotone shape the forward's `_KV_ADDR_HOIST_HEAD_DIMS` comment
# describes: whether a hoist or a wider tile survives is decided by the
# register allocator one width at a time). Not confirmed against an ISA dump.
#
# 80 is a coin toss (w16 is +1.7% non-causal and -0.1% causal, inside the
# drift) and stays at the default rather than gaining an entry for noise.
#
# 384 and 512 are the sharded widths, where `num_waves` no longer means
# BLOCK_M: a workgroup is `num_waves / shards` Q row sub-tiles, so 8 waves at
# 4 shards is BLOCK_M 32. Interleaved, 3 reps, same shape, TFLOPS non-causal /
# causal:
#
#   head_dim   4 waves (m16)   8 waves (m32)   16 waves (m64)
#   384             --          35.1/36.1       31.8/31.9
#   512        10.8/11.9        24.1/28.5       LDS: 65792 B
#
# The 4-wave collapse at 512 is 2.2x, far outside the drift, and it is not a
# register effect -- per-wave pressure is identical at every wave count. It is
# K/V global traffic: BLOCK_M 16 quadruples the workgroup count over BLOCK_M
# 64 and every workgroup re-reads the whole KV sequence. 16 waves does not
# help either (measured at 384, -10%) and at 512 does not fit: the reduction
# buffer scales with the wave count and stops aliasing V.
#
# 8 waves is therefore the answer at both, from opposite directions.
_NUM_WAVES_BY_HEAD_DIM: dict[int, int] = {16: 16, 192: 16, 224: 16, 256: 16, 384: 8, 512: 8}
_DEFAULT_NUM_WAVES = 8

# Head-dim shards: waves cooperating on one Q row sub-tile, each holding only
# `head_dim / shards` columns of Q, dO and the dQ accumulator.
#
# 1 everywhere it fits, because sharding is not free: it buys registers with
# two cross-shard LDS reductions per KV tile and three extra barriers. It is
# switched on only where the unsharded form does not exist -- 384 would need
# 384 VGPRs of operands and 512 would need 512, against a 256-VGPR file.
#
# 4 and not 2 at both widths: 2 shards leave a 192-wide slice at 384 and a
# 256-wide one at 512, which are the register profiles of the unsharded 192 and
# 256 builds -- and 256 unsharded already spills 132 bytes. 4 shards give
# 96- and 128-wide slices, and unsharded 128 measures 221 VGPRs with no spill.
# Measured, 4 shards: 384 comes out at 217 VGPRs and 512 at 254, both with
# **zero** spill.
#
# 8 is worse, and not for the reason the register table would suggest. It
# halves the slice again (512 would be 64 wide) but it also halves BLOCK_M at
# a fixed wave count, and BLOCK_M is what this kernel is sensitive to: 8
# shards at 8 waves measured 13.8/14.0 TFLOPS at 512 against 24.1/28.5 for 4.
# Going to 16 waves to recover BLOCK_M does not fit -- the reduction buffer
# scales with the wave count.
_SHARDS_BY_HEAD_DIM: dict[int, int] = {384: 4, 512: 4}
_DEFAULT_SHARDS = 1

# KV columns per tile.
#
# 32 everywhere except 512, and the exception is arithmetic rather than
# tuning. K and V are both staged row-major and padded by 4 elements, so the
# pair costs `2 * BLOCK_N * (head_dim + 4) * 2` bytes: at head_dim 512 and
# BLOCK_N 32 that is 66048, over the 64 KiB workgroup cap, and even removing
# the padding entirely lands on exactly 65536 with no room for the reduction
# buffer. BLOCK_N 16 halves it to 33024 and leaves the padding intact.
#
# Everywhere else 32 stands. BLOCK_N is the width of the S and dP
# accumulators, which are *dead* register pressure in this kernel -- unlike the
# forward, where a wider tile amortises the per-tile softmax. Widening it would
# add 2 * BLOCK_N/16 * 8 VGPRs against a head_dim-proportional budget that is
# already the binding constraint. Revisit only at head_dim <= 32.
_BLOCK_N_BY_HEAD_DIM: dict[int, int] = {384: 16, 512: 16}
_DEFAULT_BLOCK_N = 32

# How K reaches GEMM3 (`dq += K^T @ dS^T`).
#
#   "scalar"      read K^T out of the row-major K tile with 8 strided 16-bit
#                 LDS loads per operand. No extra LDS, no extra global traffic,
#                 head_dim scalar LDS reads per KV tile per wave.
#   "transposed"  stage a second, transposed copy K^T[d][kv] with
#                 `global_load_tr_b128`, exactly as the forward stages V^T, and
#                 read one vector per operand. Costs a second full K global
#                 load and `head_dim * (BLOCK_N + 4)` more elements of LDS.
#
# The forward measured the equivalent choice for V at +2.7% at N >= 4096, on a
# loop that reads V *once* per operand, and the expectation here was a larger
# win because GEMM3 does 8 scalar reads per operand. **The opposite happened.**
# TFLOPS, B=1 H=8 N=4096 f16, scalar -> transposed:
#
#   head_dim   non-causal        causal        ratio (nc / c)
#   64        86.7 -> 87.7    90.6 -> 88.4     1.01 / 0.98
#   128       93.5 -> 56.4    91.0 -> 55.9     0.60 / 0.61
#   256       39.9 -> 23.3    40.4 -> 20.5     0.59 / 0.51
#
# So the transposed arm is a wash at 64 and loses 40-50% from 128 up. The
# forward's V^T pays for itself because V is loaded once either way; here K^T
# is a **second** full K tile -- a second global load and
# `head_dim * (BLOCK_N + 4)` more LDS -- to serve a GEMM that already had the
# data resident. The LDS reads it saves are cheaper than the traffic it adds,
# and the gap grows with head_dim exactly as the extra tile does.
#
# Kept as a knob rather than deleted: the two arms are bitwise identical, so
# it is a free A/B for anything that changes the LDS or traffic budget (a KV
# prefetch, a wider BLOCK_N, a fused dK/dV that has K^T resident anyway).
_DEFAULT_KT_LDS_LAYOUT = "scalar"

# Distance-1 K/V prefetch, as the forward's `k_prefetch_dist`/`v_prefetch_dist`.
#
# 0 until measured. The loop today issues both global reads inside the barrier
# pair and waits on them immediately: at head_dim 128 the ISA puts 0..4
# instructions and **zero** WMMA between each `global_load` and the
# `s_wait_loadcnt` that consumes it, where the forward gets 35-36 instructions
# and 7 WMMA, plus two loads it does not wait on at all. So the latency is
# fully exposed here and there are three GEMMs per tile to hide it behind --
# one more than the forward has.
#
# The carry is cheap: 1-2 batches each for K and V at every head_dim on the
# ladder, so 8-16 VGPRs. dQ measures 150/221/231/256 VGPRs at head_dim
# 64/128/192/256, so it fits everywhere except 256, which already spills 132.
_DEFAULT_KV_PREFETCH_DIST = 0

# LLVM's `amdgpu-sched-strategy` function attribute. Unset ("" -> the default
# GCN scheduler) everywhere except head_dim 512.
#
# Interleaved, 3 reps, B=1 H=8 N=4096 f16, TFLOPS default -> max-memory-clause:
#
#   head_dim   non-causal        causal
#   384       35.1 -> 35.0    36.1 -> 35.0
#   512       23.8 -> 36.1    27.8 -> 35.9
#
# So it is worth +52% / +29% at 512 and nothing at 384, on two builds that
# differ only in width -- same 4 shards, same BLOCK_N 16, same 8 waves. The
# forward's copy of this knob is gated the same way and its comment gives the
# mechanism: the default scheduler sinks each `ds_load` next to its consuming
# WMMA and the WAR dependency then forces a full `s_wait_dscnt` drain between
# every pair. 512 is the width where this kernel's LDS reads per KV tile are
# densest, so it is where the drains cost the most.
#
# `kv_prefetch_dist=1` reaches the same place from the other side (36.2 / 34.3
# at 512, measured in the same run) and the two do not compose -- together they
# give 36.0 / 36.2, i.e. neither adds to the other. That is the signature of
# both fixing one bottleneck: exposed load latency. The scheduler attribute is
# preferred because it costs no registers and no loop-carried values.
_SCHED_STRATEGY_BY_HEAD_DIM: dict[int, str] = {512: "max-memory-clause"}


def default_sched_strategy(head_dim: int) -> str:
    """LLVM scheduling strategy at this head_dim; "" is the default scheduler."""
    return _SCHED_STRATEGY_BY_HEAD_DIM.get(head_dim, "")


def default_num_waves(head_dim: int) -> int:
    """Waves per workgroup at this head_dim."""
    return _NUM_WAVES_BY_HEAD_DIM.get(head_dim, _DEFAULT_NUM_WAVES)


def default_shards(head_dim: int) -> int:
    """Waves cooperating on one Q row sub-tile at this head_dim."""
    return _SHARDS_BY_HEAD_DIM.get(head_dim, _DEFAULT_SHARDS)


def default_block_m(head_dim: int) -> int:
    """Q rows per workgroup at this head_dim."""
    return _ROWS_PER_WAVE * (default_num_waves(head_dim) // default_shards(head_dim))


def default_block_n(head_dim: int, causal: bool) -> int:
    """KV columns per tile at this head_dim."""
    return _BLOCK_N_BY_HEAD_DIM.get(head_dim, _DEFAULT_BLOCK_N)


# Whether the KV address hoists `row * stride_seq` out of the loop; see
# `kv_off` in `fmha_common_gfx1201.make_addr_pair` for the two forms.
#
# The forward keys this off head_dim from a measured table. Copying its table
# would be borrowing a conclusion rather than a fact -- the two kernels have
# different loop bodies -- so this one was swept where it was needed, which is
# the two sharded widths. Interleaved, 4 reps, B=1 H=8 N=4096 f16, TFLOPS
# off -> on:
#
#   head_dim   non-causal        causal
#   384       35.0 -> 36.3    35.3 -> 36.3
#   512       36.3 -> 37.6    35.8 -> 37.5
#
# +3 to +5%, which is inside this board's between-run drift and would not be
# believable from one run. It is kept because it is *flat*: every rep after the
# warmup reads 37.6/37.5 against 36.3/35.8 to within 0.1 TFLOPS, which is the
# lore's stated signature of a real effect against drift. 16 to 256 remain
# unswept and off.
_KV_ADDR_HOIST_HEAD_DIMS: frozenset[int] = frozenset({384, 512})


def _kv_addr_hoist(head_dim: int, causal: bool) -> bool:
    return head_dim in _KV_ADDR_HOIST_HEAD_DIMS


# ---------------------------------------------------------------------------
# The two halves of a build request. Same split as the forward's, for the same
# reason: a caller states a problem, the tuning policy answers with a schedule.
# Both frozen so the pair can be an `lru_cache` key.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwdDqInputMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_heads: int
    head_dim: int
    # The V/O extent, when it differs from the QK one. `None` means "same".
    #
    # There is only ONE compiled tile here, unlike dK/dV which carries a second
    # `block_dmodel_v`, so the tile must cover the *wider* of the two and the
    # narrower axis rides as a masked runtime extent. That is what
    # `padded_head` has to reflect: the kernel's `vo_cols` is
    # `MaskedAxis(hdim_vo, active=PADDED_HEAD)`, so with the flag off the V/O
    # axis is not bounded at all and a narrower V is walked to the full tile.
    head_dim_v: int | None = None
    causal: bool = True
    dtype_str: str = "bf16"
    sm_scale: float | None = None
    causal_type: int | None = None
    dropout: bool = False
    philox_width: int | None = None
    # Attention bias, matching the forward's `BIAS_TYPE`: a (B, H, Sq, Sk)
    # matrix added to the scores after the scale and before the mask.
    #
    # It has to be an input, not just an output question. The forward folds the
    # bias into the score *and* into the logsumexp it stores, and this kernel
    # recomputes `P` from that logsumexp -- so without the bias term the
    # recomputed P is off by `exp(-bias)` and dQ, dK and dV are all wrong.
    #
    # A bias build emits dB too, always: dB is `dS`, which the kernel already
    # forms for its last GEMM, so gating it would save a store and cost a
    # build axis. `dbias=` is therefore required alongside `bias=`.
    bias: bool = False


@dataclass(frozen=True)
class BwdDqKnobs:
    """How to compute it. Every field `None` means "policy decides"."""

    # Compile-time width baked into the binary. The *real* extent travels as a
    # runtime argument and may be smaller, which is what `padded_head` records.
    block_dmodel: int | None = None

    block_m: int | None = None
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

    def merge(self, other: "BwdDqKnobs | None") -> "BwdDqKnobs":
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)


_KNOBS_FALLBACK = BwdDqKnobs(
    kt_lds_layout=_DEFAULT_KT_LDS_LAYOUT,
    kv_prefetch_dist=_DEFAULT_KV_PREFETCH_DIST,
    waves_per_eu=2,
    fp_mode="noninf",
    denormals_are_zero=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    lpt_tile_order=True,
)


def _head_dim_v(meta) -> int:
    """The V/O extent, defaulting to the QK one."""
    return meta.head_dim_v if meta.head_dim_v is not None else meta.head_dim


def resolve_knobs(meta: BwdDqInputMetadata, overrides: "BwdDqKnobs | None" = None) -> BwdDqKnobs:
    """The complete configuration for `meta`.

    `overrides` is applied *first*, so a pinned knob participates in deriving
    the ones downstream of it rather than being stamped on afterwards --
    pinning `num_waves` therefore also moves `block_m`.
    """
    s = _KNOBS_FALLBACK.merge(overrides)
    if s.block_dmodel is None:
        # The wider of the two axes: one tile serves both, so it must cover
        # whichever is larger and the other rides as a masked runtime extent.
        s = replace(s, block_dmodel=max(meta.head_dim, _head_dim_v(meta)))
    hd = s.block_dmodel

    # **Derived here rather than in `plan`.** `plan` is only one of the ways in
    # -- `build_bwd_dq_module`'s keyword front end and the AOT builder both call
    # `resolve_knobs` directly -- and a `padded_head` computed one level up is a
    # `padded_head` those two never see. It defaulted to False for them, which
    # left `vo_cols` unmasked whenever the V/O extent was the narrower one.
    if s.padded_head is None:
        s = replace(s, padded_head=(hd != meta.head_dim) or (hd != _head_dim_v(meta)))

    if s.shards is None:
        s = replace(s, shards=default_shards(hd))
    if s.sched_strategy is None:
        s = replace(s, sched_strategy=default_sched_strategy(hd))
    if s.num_waves is None:
        s = replace(s, num_waves=default_num_waves(hd))
    if s.block_m is None:
        s = replace(s, block_m=_ROWS_PER_WAVE * (s.num_waves // s.shards))
    if s.block_n is None:
        s = replace(s, block_n=default_block_n(hd, meta.causal))
    if s.kv_addr_hoist is None:
        s = replace(s, kv_addr_hoist=_kv_addr_hoist(hd, meta.causal))
    if s.flat_work_group_size is None:
        s = replace(s, flat_work_group_size=32 * s.num_waves)
    # `block_m`, `num_waves` and `shards` state one fact twice over, so they
    # must agree. A caller who pins only `block_m` gets the wave count that
    # goes with it, at whatever shard count is in force.
    if s.block_m * s.shards != _ROWS_PER_WAVE * s.num_waves:
        if overrides is not None and overrides.num_waves is None:
            _nw = (s.block_m // _ROWS_PER_WAVE) * s.shards
            s = replace(s, num_waves=_nw, flat_work_group_size=32 * _nw)
        else:
            raise ValueError(
                f"block_m ({s.block_m}) must be {_ROWS_PER_WAVE} * num_waves "
                f"({s.num_waves}) / shards ({s.shards}); one wave owns 16 rows "
                f"of one head-dim shard"
            )
    return s


@dataclass(frozen=True)
class BwdDqPlan:
    """Everything a host needs for one build, from one call."""

    meta: BwdDqInputMetadata
    knobs: BwdDqKnobs


def plan(request: BwdDqInputMetadata, overrides: BwdDqKnobs | None = None) -> BwdDqPlan:
    """The one entry point into tuning: the caller's inputs in, a full plan out.

    `request` is returned as `BwdDqPlan.meta` **unchanged**. The rounding up to
    a compiled tile lands in `BwdDqPlan.knobs.block_dmodel`, and
    `knobs.padded_head` records whether the two differ.
    """
    if request.bias and (request.causal or request.causal_type):
        raise ValueError(
            "bias and causal masking are mutually exclusive, as in the forward: a bias "
            "already is an additive mask, so the pair has no defined meaning. Fold the "
            "causal pattern into the bias tensor, or drop the bias"
        )
    head_dim = request.head_dim
    head_dim_v = request.head_dim_v if request.head_dim_v is not None else head_dim
    # One tile serves both axes, so it has to cover the wider one; whichever
    # axis is narrower than the tile is then a masked runtime extent.
    wide = max(head_dim, head_dim_v)
    if wide < 1 or wide > MAX_HEAD_DIM:
        raise ValueError(f"kernel requires 1 <= head_dim <= {MAX_HEAD_DIM}, got ({head_dim}, {head_dim_v})")
    block_dmodel = _round_to_ladder(wide)
    # `resolve_knobs` derives `padded_head` from the same two extents.
    knobs = resolve_knobs(request, (overrides or BwdDqKnobs()).merge(BwdDqKnobs(block_dmodel=block_dmodel)))
    return BwdDqPlan(request, knobs)
