# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 dK/dV backward kernel: which knobs, which shape.

Same split as `fmha_tuning_gfx1201.py` and for the same reason -- the kernel
file is about correctness and this one is about speed, and a number here moves
when a sweep says so. It imports nothing from `flydsl`, so it is trivially
stable-only and importable from anywhere.

`block_m`, `num_waves`, `split_head_dim`, `contraction_shards` and
`transposed_source` have each been swept and carry their tables below. The
floating-point knobs are one default each, carried over from the forward;
treat those as placeholders. `lds_bytes` / `_fits_lds` is not a placeholder --
it is a legality calculation and must stay correct.

Geometry the kernel and this module must agree on
-------------------------------------------------

The dK/dV kernel is the transpose of the forward's loop: it holds a K/V tile
and streams Q/dO past it. That inverts which tensor is register-resident and
which transits LDS, and it fixes the wave decomposition:

- **a team of waves owns 16 KV rows.** A team is `1` wave unsplit, `2` under
  `split_head_dim` (roles: K/S and V/dP), and `2 * contraction_shards` when
  the contraction is sharded too -- so `BLOCK_N = 16 * num_waves /
  TEAM_WAVES`. Teams own *disjoint* KV rows, which is what keeps the dK/dV
  accumulators team-private. The alternative -- every wave over the same KV
  tile, splitting the Q stream -- needs a `BLOCK_N x head_dim` f32 reduction
  through LDS per workgroup, which does not fit.
- **Q and dO are staged in LDS twice each**, row-major and transposed, because
  each is read once as a WMMA A operand along `d` (for S and dP) and once along
  `q` (for dK^T and dV^T). See the kernel's LDS-layout comment. That is what
  makes `lds_bytes` the binding constraint on `block_m` -- unless
  `transposed_source="derived"` drops the transposed pair, which is what makes
  head_dim 448 and 512 fit at all.

Register floor, per wave, in VGPRs. An f16 operand tile over 16 KV rows is
`head_dim/4` and an f32 accumulator `head_dim/2`, so with `C` contraction
shards and a `2C`-wave team:

    packs         head_dim / (4 * C)
    dk + dv accs  head_dim / (2 * C)
    state         0.75 * head_dim / C

Measured ISA gives `vgpr = state + 62`, and the model reproduces the spill
counts it can be checked against -- at head_dim 256 unsplit it predicts a
deficit of 190 and the ISA reports 203 spills. That is what the three
decomposition knobs are for, and the crossovers are all spill boundaries:

    head_dim   default            state   why
    <= 128     split off            1.5D  fits; splitting costs BLOCK_N
    192-256    split, C=1          0.75D  unsplit spills 79-203
    >= 320     split, C=2, derived 0.375D C=1 spills, four tiles overflow

`is_wide` is that last threshold. Nothing here is keyed on `causal`.
"""

from dataclasses import dataclass, fields, replace

# One wave owns one WMMA M tile of KV rows. Not a knob: it is the instruction.
ROWS_PER_WAVE = 16

# Workgroup LDS budget on gfx1201. The hardware allows 64 KiB per workgroup.
_LDS_LIMIT = 65536
_WAVE_LANES = 32  # gfx1201 is wave32; the exchange is sized per lane

# LDS row padding, in elements. Same value and same reason as the forward's
# `_LDS_PAD`: it moves consecutive rows off the same bank group, and a swizzle
# was measured a net loss there.
LDS_PAD = 4

# The compiled tile widths, mirroring the forward's ladder so a backward build
# can be requested for any head_dim the forward accepts. Widths above 128 are
# *buildable but spill*; see the register floor in the module docstring.
_BLOCK_DMODEL_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)

# At and above this the wide decomposition wins: `contraction_shards=2` and
# `transposed_source="derived"`. See `default_contraction_shards` for the
# measurement -- this is a *measured* crossover, not the legality boundary.
#
# The two are worth keeping apart. Legality alone would put the threshold at
# 448, which is the first width where the four staging tiles cannot fit at any
# `num_waves`; 320 and 384 admit all four combinations. The reason 320 routes
# wide is that `0.75 * head_dim` = 240 exceeds the ~194 register budget there
# and the unsharded build starts spilling, which costs more than the sharding
# does. The crossover falls between the 256 and 320 rungs, and there is no
# rung between them, so no finer threshold is expressible.
_WIDE_HEAD_DIM_MIN = 320

MAX_HEAD_DIM = _BLOCK_DMODEL_LADDER[-1]

# Waves per workgroup; `BLOCK_N = 16 * this`.
#
# The knob that matters most at large head_dim, and the reason is traffic
# rather than registers: every workgroup streams the *whole* Q/dO sequence past
# its KV tile, so that traffic scales as `1/BLOCK_N`. Widening the workgroup
# quarters it. Per-wave register state is untouched -- each wave still owns 16
# KV rows and its own dK/dV accumulators -- so this is orthogonal to the spills
# at head_dim >= 192 and composes with anything that fixes them.
#
# Interleaved, three reps, median, at two shapes. Milliseconds, lower better;
# the previous default 4 against the two alternatives:
#
#              B=4 H=16 N=2048           B=1 H=8 N=4096
#   hd  mask   nw4     nw8    nw16     nw4     nw8    nw16
#   64  no     2.03    2.04    2.09    1.04    1.01    0.99
#   64  yes    1.15    1.24    1.26    0.65    0.71    0.78
#   80  no     2.90    2.54    2.71    1.47    1.24    1.24
#   80  yes    1.63    1.47    1.53    0.95    0.86    0.97
#   128 no     4.73    4.23    4.62    2.32    1.84    2.01
#   128 yes    2.32    2.31    2.57    1.25    1.21    1.25
#   160 no     6.69    6.65    6.17    3.47    3.21    2.98
#   192 no    10.05    9.54    8.83    4.95    4.72    4.55
#   224 no    17.22   16.92   13.24      --      --      --
#   256 no    28.28   21.18   19.08   14.09   10.85    9.53
#   256 yes   15.39   11.69   11.11    7.86    5.89    5.46
#
# Three bands, and the same three at both shapes, which is why this is a table
# and not a constant:
#
#   <= 64    4    widening loses; the tile is small enough that fewer, fatter
#                 workgroups just cost occupancy
#   80-128   8    1.05-1.26x
#   >= 160  16    1.06-1.48x, growing with head_dim exactly as the Q/dO
#                 traffic per KV row does
#
# Deliberately **not** keyed on `causal`. The two masking modes disagree only
# at 128 and 160, by at most 4% -- inside the board's own drift
# (`sdpa_lore_gfx1201.md`) -- and at 192 causal the wider setting is faster
# anyway. A second axis would be two more numbers to maintain for noise.
#
# The previous default of 4 everywhere came from one un-interleaved sweep at a
# third shape, whose comment said "re-sweep interleaved before moving it".
# This is that re-sweep.
# Reverse the KV-tile axis of the grid, the forward's `lpt_tile_order`.
#
# **The direction is inverted here, which is why this defaults to off.** The
# forward reverses because its cost *rises* with the tile index -- query tile i
# attends keys j <= i, so late tiles are the expensive ones, and the natural
# order dispatches the cheap ones first. Longest-processing-time scheduling
# wants the opposite.
#
# dK/dV is the transpose of that loop: key j is attended by queries i >= j, so
# a *low* KV tile is visited by nearly every Q block and a high one by few. The
# natural order already dispatches longest-first, and reversing it would be
# anti-LPT. Kept as a knob rather than not ported at all, because that
# reasoning is a prediction about a 64-CU dispatcher, and the measurement is
# cheap. Measured, interleaved x3, median, natural against reversed:
#
#                       head_dim 64      head_dim 128     head_dim 256
#   B=4 H=16 N=2048  full   0.99x         0.97x            1.00x
#                  causal   1.00x         0.94x            0.99x
#   B=1 H=8  N=4096  full   1.05x         1.00x            0.99x
#                  causal   0.96x         0.91x            0.92x
#
# The prediction holds, and the shape of the result is the confirmation: the
# loss is **causal-only** -- 0.91-0.96x across both shapes -- while non-causal
# is a wash, because there every KV tile has the same cost and the order cannot
# matter. Reversing turns longest-first into shortest-first exactly where the
# cost is skewed.
#
# Kept rather than deleted: the reversal is bitwise identical (it is a grid
# permutation, verified), so it is a free A/B for anything that changes the
# per-tile cost profile -- a fused dK/dV/dQ, or a window that makes the tail
# tiles expensive instead of cheap.
_DEFAULT_LPT_TILE_ORDER = False


# Pair the waves and split the head dim of dK/dV between them.
#
# Not a schedule tweak but a different decomposition, and it wins only where
# the baseline spills -- see the kernel's `SPLIT_HEAD_DIM`
# section for the mechanism and the measurements. Summary, B=4 H=16 N=2048 f16,
# each arm at its own best num_waves:
#
#   head_dim   full     causal      baseline spill
#   128        0.86x    0.82x       0
#   192        1.06x    1.34x       91
#   256        2.39x    2.61x       251
#
# The crossover is the spill boundary, which is why this is keyed on head_dim
# and not measured per shape: below 192 the baseline holds everything in
# registers and the pair exchange plus the halved BLOCK_N buy nothing.
_SPLIT_HEAD_DIM_MIN = 192


def is_wide(head_dim: int, head_dim_v: int) -> bool:
    """Whether this build routes to the wide kernel rather than the primary one."""
    return max(head_dim, head_dim_v) >= _WIDE_HEAD_DIM_MIN


def default_contraction_shards(head_dim: int, head_dim_v: int) -> int:
    """Ways to split the S / dP contraction across a team.

    1 below the wide threshold and 2 at or above it, both halves measured.
    Sharding costs `BLOCK_N`: a team is `2 * shards` waves, so at fixed
    `num_waves` doubling the shard count halves the KV rows per workgroup and
    doubles how often each workgroup re-reads the Q/dO stream. It buys the
    register state back, at `0.75 * head_dim / C`.

    The full crossover sweep, B=1 H=8 N=4096 f16 nw=16, percent of the 212.8
    TFLOPS WMMA ceiling as full/causal, interleaved x3 and median. Blank where
    the four staging tiles do not fit at any `num_waves`:

        head_dim   C=1 tile     C=1 derived  C=2 tile     C=2 derived
        256        31.1/32.6    27.4/27.3    20.9/19.8    25.9/24.1
        320        22.4/21.3    19.0/19.2    17.0/16.2    24.2/25.8
        384             --      17.8/16.6         --      19.2/19.4
        448             --      13.7/11.6         --      19.9/20.2

    256 prefers the unsharded build by 17-26% and 320 prefers the sharded one
    by 8-21%, so the crossover is between those two rungs and
    `_WIDE_HEAD_DIM_MIN` sits on the lower edge of the sharded side. The
    margin widens with head_dim -- 45-74% by 448 -- because that is the spill
    curve: `0.75 * head_dim` passes the ~194 budget at 260 and keeps going.

    4 shards is worse again (15.5% at head_dim 512) despite being the only
    spill-free build there. `BLOCK_N` drops to 32 and the Q traffic swamps the
    registers saved -- the same trade `fmha_tuning_bwd_dq_gfx1201` records at
    its own BLOCK_M, where 16 cost it 2.2x.
    """
    return 2 if is_wide(head_dim, head_dim_v) else 1


def default_transposed_source(head_dim: int, head_dim_v: int) -> str:
    """Where the Q^T / dO^T WMMA operands come from.

    "derived" is what makes the wide widths fit -- it removes the two
    transposed tiles, which at head_dim 512 is 40960 of the 90496 B that
    otherwise overflows, and at 448 it is the difference between legal and not
    at any `num_waves`.

    It is not free, but it is far cheaper than the read counting suggests
    (1.0 -> ~4.5 LDS reads per WMMA, against dq measured LDS-bound at 3.3).
    Against "tile" at C=1, B=1 H=8 N=4096, full/causal percent of peak:

        head_dim   tile         derived
        192        26.0/25.3    25.2/26.3
        256        31.1/32.6    27.4/27.3
        320        22.4/21.3    19.0/19.2

    So roughly 12% at 256 and 320 and nothing measurable at 192. It is tied to
    `contraction_shards` in the policy only because the same threshold suits
    both, not because either needs the other: the knobs are independent, and
    the C=2 column of the table in `default_contraction_shards` shows "tile"
    losing to "derived" at 320 even though both fit.
    """
    return "derived" if is_wide(head_dim, head_dim_v) else "tile"


def default_split_head_dim(head_dim: int, head_dim_v: int) -> bool:
    """Whether to pair the waves and split the head dim of dK/dV.

    Keyed on head_dim because the crossover *is* the spill boundary -- the
    baseline holds everything in registers below 192 and the pair exchange
    plus the halved BLOCK_N buy nothing there. Not keyed on `causal`: the
    split wins by more under causal at both widths where it wins at all.

    Requires `head_dim == head_dim_v`. The split gives both waves one operand
    array and one LDS stride, which only works when K and V are the same
    width; the kernel asserts it too.
    """
    return head_dim >= _SPLIT_HEAD_DIM_MIN and head_dim == head_dim_v


_NUM_WAVES_SMALL_MAX_HEAD_DIM = 64
_NUM_WAVES_MEDIUM_MAX_HEAD_DIM = 128

# Head dims that stage 32 Q rows per pass rather than 16. Measured, and the
# effect is far outside the board's drift at the wide end.
#
# Two 16-row sub-tiles per staging pass amortise the pass's two barriers over
# twice the work, which is why 32 looks like the obvious default -- right up to
# the point where the extra live values spill. The per-wave register floor is
# `1.5 * head_dim` VGPRs before any transient, so that point arrives early.
#
# Measured B=2 H=12 N=4096 f16, best of two or three alternating reps
# (`bm32/bm16`, below 1.0 means 32 wins):
#
#   head_dim   non-causal   causal
#   16           0.930       0.903
#   32             --        0.937
#   48           1.022       1.032
#   64           0.989       0.947
#   80             --        1.300
#   96             --        1.378
#   128            --        1.528
#   160            --        2.211
#
# So the penalty grows monotonically from 80 up and there is no threshold to
# key on below it: **48 loses and 64 wins**, which is not an ordering any
# formula produces. 48 is the awkward width -- 6 threads per cooperative-load
# row against 64's 8 -- and the forward's tables record the same width class
# misbehaving (its head_dim 224, at 14 threads per row, spills 101 registers).
# Hence a set rather than a bound. 64's margin is 1-5% and 48's 2-3%, both
# close to the drift floor; re-measure interleaved before moving either.
#
# Note the sign is the *opposite* of the lesson the forward's tables record
# three times ("spill count is not a proxy for speed"). Here it is a good
# proxy, because what spills is an accumulator read and written by every WMMA
# rather than an operand preloaded once.
_BLOCK_M_32_HEAD_DIMS = frozenset({16, 32, 64})
_DEFAULT_BLOCK_M = 32


def _round_to_ladder(head_dim: int) -> int:
    """Smallest compiled tile width covering `head_dim`."""
    for w in _BLOCK_DMODEL_LADDER:
        if w >= head_dim:
            return w
    raise ValueError(f"head_dim {head_dim} exceeds the largest compiled tile ({MAX_HEAD_DIM})")


def lds_bytes(
    block_m: int,
    head_dim: int,
    head_dim_v: int,
    elem_bytes: int = 2,
    num_waves: int | None = None,
    split_head_dim: bool = False,
    transposed_source: str = "tile",
) -> int:
    """LDS a Q/dO staging pass needs, in bytes.

    Four tiles, because both tensors are needed in both orientations:

        Q  row-major   block_m         x (head_dim   + pad)
        Q  transposed  head_dim        x (block_m    + pad)
        dO row-major   block_m         x (head_dim_v + pad)
        dO transposed  head_dim_v      x (block_m    + pad)

    The transposed copies are not an optimisation. A WMMA A operand's eight
    per-lane elements run along the *contraction* axis, and the four GEMMs in
    this kernel contract over `d` twice and over `q` twice, so each tensor is
    read both ways.

    `transposed_source="derived"` drops the two transposed tiles and reads
    those operands strided out of the row-major pair instead -- eight
    `ds_read_u16` where the tile costs one `ds_read_b128`, since gfx1201 has no
    `ds_load_tr16_b128`. The forward measured the equivalent at 2.7% for a
    single tensor; it is what makes head_dim 512 fit, and only
    `fmha_bwd_dkdv512_gfx1201_kernel` implements it.
    """
    rm = block_m * (head_dim + LDS_PAD) + block_m * (head_dim_v + LDS_PAD)
    tr = head_dim * (block_m + LDS_PAD) + head_dim_v * (block_m + LDS_PAD)
    if transposed_source == "derived":
        tr = 0
    total = (rm + tr) * elem_bytes
    # Per-row LSE and delta for the Q tile in flight: two f32 runs of block_m,
    # staged once and read by every wave.
    total += 2 * block_m * 4
    if split_head_dim:
        # The team exchange: one f32 slot per lane per wave, eight values each.
        # Independent of how wide the team is -- two waves relaying S and dP, or
        # `2 * contraction_shards` waves reducing partials, is one slot per wave
        # either way.
        if num_waves is None:
            raise ValueError("split_head_dim needs num_waves to size the exchange")
        total += num_waves * _WAVE_LANES * 8 * 4
    return total


def _fits_lds(block_m, head_dim, head_dim_v, num_waves=None, split_head_dim=False, transposed_source="tile") -> bool:
    return lds_bytes(block_m, head_dim, head_dim_v, 2, num_waves, split_head_dim, transposed_source) <= _LDS_LIMIT


def default_block_m(
    head_dim: int, head_dim_v: int, num_waves=None, split_head_dim: bool = False, transposed_source: str = "tile"
) -> int:
    """Q rows per staging pass: the measured choice, reduced until it fits LDS.

    Walks *down* rather than failing, because block_m is a pure schedule
    parameter -- 16 is always legal (one WMMA M tile) and always correct.
    """
    want = _DEFAULT_BLOCK_M if max(head_dim, head_dim_v) in _BLOCK_M_32_HEAD_DIMS else 16
    for bm in (want, 16):
        if _fits_lds(bm, head_dim, head_dim_v, num_waves, split_head_dim, transposed_source):
            return bm
    return 16


def default_num_waves(head_dim: int, head_dim_v: int) -> int:
    """Waves per workgroup, hence `BLOCK_N = 16 * this`.

    Independent of LDS: the Q/dO tiles are shared by every wave, so widening
    the workgroup costs registers (each wave's own K/V and dK/dV) rather than
    LDS. See the table above for the three bands and the measurements.
    """
    wide = max(head_dim, head_dim_v)
    if wide <= _NUM_WAVES_SMALL_MAX_HEAD_DIM:
        return 4
    if wide <= _NUM_WAVES_MEDIUM_MAX_HEAD_DIM:
        return 8
    return 16


@dataclass(frozen=True)
class BwdDkDvMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_heads: int
    head_dim: int
    causal: bool = False
    dtype_str: str = "f16"
    head_dim_v: int | None = None
    sm_scale: float | None = None
    causal_type: int | None = None
    dropout: bool = False
    philox_width: int | None = None
    # Attention bias, matching the forward's `BIAS_TYPE`: a (B, H, Sq, Sk)
    # matrix added to the scores after the scale and before the mask.
    #
    # An input only. The forward folds the bias into the score *and* into the
    # logsumexp it stores, and this kernel recomputes `P` from that logsumexp,
    # so without the bias term dK and dV are wrong by `exp(-bias)`.
    #
    # **dB is not emitted here.** It is `dS`, which this kernel also has, but
    # it walks Q tiles for a fixed KV block -- so its eight elements are eight
    # q *rows* at one kv column and the store would go down a dB column. The
    # dQ kernel writes it along a row instead; see `return_dbias` there.
    bias: bool = False


@dataclass(frozen=True)
class BwdDkDvKnobs:
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

    def merge(self, other: "BwdDkDvKnobs | None") -> "BwdDkDvKnobs":
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)


_KNOBS_FALLBACK = BwdDkDvKnobs(
    lpt_tile_order=_DEFAULT_LPT_TILE_ORDER,
    # "noninf" and not "fast": `ninf` lets the compiler assume no operand is
    # infinite, and this kernel reads a logsumexp that is deliberately +inf for
    # a row with no live keys. The forward records the same hazard deleting its
    # KV tail mask.
    fp_mode="noninf",
    denormals_are_zero=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    # The loop body holds four WMMA chains and two LDS staging passes; the
    # default GCN scheduler sinks each LDS load next to its consumer. Same
    # reasoning as the forward's causal builds, untested here.
    sched_strategy="max-memory-clause",
    waves_per_eu=None,
    addr_hoist=False,
    padded_head=False,
)


def resolve_knobs(meta: BwdDkDvMetadata, overrides: "BwdDkDvKnobs | None" = None) -> BwdDkDvKnobs:
    """The complete configuration for `meta`.

    `overrides` is applied first, so a pinned knob participates in deriving the
    ones downstream of it -- pinning `num_waves` changes `BLOCK_N` and hence
    nothing else here, but pinning `block_dmodel` changes `block_m`.
    """
    s = _KNOBS_FALLBACK.merge(overrides)
    if s.block_dmodel is None:
        s = replace(s, block_dmodel=meta.head_dim)
    if s.block_dmodel_v is None:
        s = replace(s, block_dmodel_v=meta.head_dim_v if meta.head_dim_v is not None else s.block_dmodel)
    hd, hdv = s.block_dmodel, s.block_dmodel_v
    if s.contraction_shards is None:
        s = replace(s, contraction_shards=default_contraction_shards(hd, hdv))
    if s.transposed_source is None:
        s = replace(s, transposed_source=default_transposed_source(hd, hdv))
    if s.split_head_dim is None:
        s = replace(s, split_head_dim=default_split_head_dim(hd, hdv))
    if s.num_waves is None:
        s = replace(s, num_waves=default_num_waves(hd, hdv))
    if s.block_m is None:
        s = replace(s, block_m=default_block_m(hd, hdv, s.num_waves, s.split_head_dim, s.transposed_source))
    if not _fits_lds(s.block_m, hd, hdv, s.num_waves, s.split_head_dim, s.transposed_source):
        raise ValueError(
            f"block_m={s.block_m} with head_dim=({hd}, {hdv}) needs "
            f"{lds_bytes(s.block_m, hd, hdv, 2, s.num_waves, s.split_head_dim, s.transposed_source)} B of LDS, "
            f"over the {_LDS_LIMIT} B cap"
        )
    return s


@dataclass(frozen=True)
class BwdDkDvPlan:
    """Everything a host needs for one build, from one call."""

    meta: BwdDkDvMetadata
    knobs: BwdDkDvKnobs


def plan(request: BwdDkDvMetadata, overrides: BwdDkDvKnobs | None = None) -> BwdDkDvPlan:
    """The one entry point into tuning: caller's inputs in, a full plan out.

    `request` comes back as `BwdDkDvPlan.meta` unchanged. The rounding up to a
    compiled tile lands in `knobs.block_dmodel`, and `knobs.padded_head`
    records whether the two differ -- exactly as `fmha_tuning_gfx1201.plan`
    does, so the two passes round a given head_dim the same way.
    """
    if request.bias and (request.causal or request.causal_type):
        raise ValueError(
            "bias and causal masking are mutually exclusive, as in the forward: a bias "
            "already is an additive mask, so the pair has no defined meaning. Fold the "
            "causal pattern into the bias tensor, or drop the bias"
        )
    head_dim = request.head_dim
    if head_dim < 1 or head_dim > MAX_HEAD_DIM:
        raise ValueError(f"kernel requires 1 <= head_dim <= {MAX_HEAD_DIM}, got {head_dim}")
    head_dim_v = request.head_dim_v if request.head_dim_v is not None else head_dim
    block_dmodel = _round_to_ladder(head_dim)
    block_dmodel_v = _round_to_ladder(head_dim_v)
    knobs = replace(
        resolve_knobs(
            request,
            (overrides or BwdDkDvKnobs()).merge(BwdDkDvKnobs(block_dmodel=block_dmodel, block_dmodel_v=block_dmodel_v)),
        ),
        padded_head=(block_dmodel != head_dim) or (block_dmodel_v != head_dim_v),
    )
    return BwdDkDvPlan(request, knobs)
