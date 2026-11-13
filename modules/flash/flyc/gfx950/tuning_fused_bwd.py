# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx950 single-pass fused backward: which knobs, for which shape.

The fused kernel (`fused_bwd.py`) is the dK/dV kernel with one change of
scope: **one workgroup owns every KV row of a (batch, kv head)**, so the dQ of
each streamed Q tile is complete the moment the tile has been seen, and is
stored straight away -- no dQ accumulator survives a tile, no atomics, no
second kernel. That makes `BLOCK_KV` a function of the problem rather than a
tuning choice: it is the smallest rung of `KV_LADDER` that holds `seqlen_k`, and
a sequence longer than the top rung is not this kernel's to serve.

So the policy has two inputs where the dK/dV one has one -- the head-dim rung
and the KV rung -- and `_GEOMETRY` is keyed on both. The KV rung is the
`block_kv` knob: `FmhaInputMetadata` has no KV extent, so `entry_plan` takes
the call's `max_seqlen_k` and pins the rung. Everything below the geometry (the
staging granule, the stream buffers, the D-axis ladder, the padded-head floor)
is the dK/dV kernel's, reached by building its knob object and asking it for
traits, so the two cannot drift.

--- What the knobs are --------------------------------------------------------

Plain build options, every value a scalar (`knobs.BwdFusedKnobFields`), so
`asdict(knobs)` renders as PON (see `sdpa-gfx950-knobs-pon-handoff.md`).
`build_traits(meta)` derives the traits on request; there is no `traits` field.

--- Where the LDS goes --------------------------------------------------------

Four regions, all in one allocation, in this order:

    stream   Q and dO tiles, `NUM_STREAM_BUFFERS` deep (the dK/dV kernel's)
    K        the whole KV block's K, staged once from the resident registers,
             read by the dQ GEMM through the transpose path
    dS^T     one `[kv][q]` tile, rewritten every Q tile
    rows     `-log2(e) * LSE` and `delta`, f32, one entry per q row

The rows region is what bounds `seqlen_q`: `q_rows_cap` rows of each. It is
the only `seqlen_q`-dependent resource the kernel has -- K/V are resident, Q
and dO stream -- so the cap is a knob with a default rather than a ladder.
"""

import math
from dataclasses import dataclass, fields, replace

from fmha_knobs import knobs_of, pinned
from fmha_meta import FmhaInputMetadata
from .knobs import BwdFusedKnobFields
from .tuning_dkdv import LDS_CAP_BYTES, BwdDkDvKnobs, BwdDkDvTraits
from .tuning_dkdv import plan as dkdv_plan

__all__ = [
    "KV_LADDER",
    "BwdFusedKnobs",
    "BwdFusedTraits",
    "bwd_fused_knobs",
    "kv_block_for",
    "plan",
    "prefer_fused",
]

# The KV extents one workgroup can own. A rung is `mfma_rows * num_waves`.
KV_LADDER = (64, 128, 256)

# Extra LDS alias scopes for the three regions the dK/dV kernel does not have.
# Every LDS access in the fused kernel is scoped, for the reason
# `dualwave_helpers._tag_lds_alias` gives: an unscoped DS access may alias
# an in-flight stream DMA, and the backend answers that with a full `vmcnt(0)`.
SCOPE_K = "lds_kt"
SCOPE_DS = "lds_ds"
SCOPE_ROWS = "lds_row"


def lanes_per_row(block_dmodel_v):
    """Lanes the delta prologue spends on one row: eight columns each, a power of two.

    A power of two because the row sum is a butterfly of `shuffle_xor`s; a
    rung like 96 rounds up to 16 lanes and masks the columns past the rung.
    """
    return 1 << max(0, (block_dmodel_v // 8 - 1).bit_length())


def rows_per_pass(num_waves, block_dmodel_v):
    """Q rows the whole workgroup covers in one pass of the delta prologue."""
    return num_waves * 64 // lanes_per_row(block_dmodel_v)


def register_demand(mfma_rows, block_dmodel, dropout=False):
    """Registers per lane a wave of this family needs without spilling. Measured, not derived.

    The occupancy hint must not ask for more waves than the register file can
    hold, or the backend meets it by spilling -- and spilling here is not just
    slow: the 32-row body at head_dim 96 with a two-wave budget spilled 142
    VGPRs and 136 SGPRs, and that build returned NaN and faulted (f16, padded,
    causal), the shape of the known lost-SGPR-copy backend defect
    (`HANDOFF-gfx950-lost-sgpr-copy.md`). Measured allocations (VGPR + AGPR):

        16 rows:  head_dim 32 -> 92, 64 -> 124..128, 96 -> 174, 128 -> 216
        32 rows:  head_dim 96 -> over 256

    The fit is `max(1.25 d + 45, 1.6 d + 20)` for 16 rows and `1.5 d + 120` for 32 (the dK/dV
    kernel's `1.5 d` loop invariant plus its transients). The 16-row fit is
    deliberately on the low side at head_dim 64: the build *can* be held to
    128 there (measured 112-128 when asked), and asking for 3 waves instead of
    4 let a causal build grow to 152 and lose the second workgroup -- 1459 us
    against 1139.
    """
    if mfma_rows == 16:
        demand = max(int(1.25 * block_dmodel) + 45, int(1.6 * block_dmodel) + 20)
    else:
        demand = int(1.5 * block_dmodel) + 120
    # Dropout's philox state and per-element keep masks. Measured at head_dim
    # 64, 16 rows: 171-175 registers when unconstrained, against 124 without.
    return demand + (50 if dropout else 0)


def kv_block_for(seqlen_k):
    """The KV rung that holds `seqlen_k`, or raise saying why not."""
    if seqlen_k <= 0:
        raise ValueError(f"seqlen_k must be positive, got {seqlen_k}")
    for rung in KV_LADDER:
        if seqlen_k <= rung:
            return rung
    raise ValueError(
        f"seqlen_k {seqlen_k} exceeds the fused kernel's widest KV block ({max(KV_LADDER)}); "
        "one workgroup must own every KV row, so this shape belongs to the split path"
    )


# `(mfma_rows, num_waves, waves_per_eu, block_q, num_stream_buffers)` by
# `(block_dmodel, block_kv)`. `block_kv == mfma_rows * num_waves` always.
#
# Three structural limits decide most of this before any measurement:
#
# - **The stream DMA spreads a staged tile's lines across waves**, so the wave
#   count must divide `SMEM_N_RPT = block_q / (512 / granule)`: 8 at granule 64
#   and `block_q` 64, 4 at granule 32. Past that, `block_q` 128 doubles it --
#   which is how the 16-row family reaches `block_kv` 256 (16 waves) at granule
#   64 and `block_kv` 128 (8 waves) at granule 32.
# - **A workgroup's waves must co-reside on one CU**, so 8 waves cap a wave at
#   256 registers and 16 waves at 128. The 32-row family's `1.5 d` loop
#   invariant does not fit 8 waves at head_dim 64 causal without spilling, and
#   those spilling builds (head_dim 64 at `block_kv` 256, head_dim 96 at 128,
#   f16) fault or return NaN with ~100+ SGPR spills -- the shape of the
#   lost-SGPR-copy backend defect, so they are not used. The split dK/dV
#   kernel at the same 32-row geometry is clean, so the trigger is register
#   pressure, not the body. The 32-row family survives only at head_dim 32,
#   `block_kv` 128.
# - **LDS.** The K region and the dS^T tile grow with `block_kv`. head_dim 128
#   at `block_kv` 256 needs 209 KB in the 16-row family, and the 32-row family
#   cannot hold `1.5 * 128` registers per lane at 8 waves, so that rung is the
#   split path's. head_dim 32 at 256 has no legal stream geometry at granule 32
#   and is served by widening to the 64 rung (`_with_widened_rung`).
#
# `num_stream_buffers` is 1 everywhere: a second buffer only overlaps tile
# `j+1`'s DMA with tile `j`, and at these lengths a workgroup sees one to four
# tiles -- while the second buffer's LDS costs a co-resident workgroup, which
# hides the same latency better. `waves_per_eu` follows from LDS and registers
# (`_with_occupancy`).
_GEOMETRY = {
    # head_dim, block_kv: rows, waves, wpe (None: derived), block_q, nbuf
    (32, 64): (16, 4, None, 64, 1),
    (32, 128): (32, 4, None, 64, 1),
    (64, 64): (16, 4, None, 64, 1),
    (64, 128): (16, 8, None, 64, 1),
    (64, 256): (16, 16, None, 128, 1),
    (96, 64): (16, 4, None, 64, 1),
    (96, 128): (16, 8, None, 128, 1),
    (128, 64): (16, 4, None, 64, 1),
    (128, 128): (16, 8, None, 64, 1),
}


def plan(request: FmhaInputMetadata, max_seqlen_k):
    """`(meta, widened)`: `request` with the tile the fused build at `max_seqlen_k` uses.

    The dK/dV ladder's tile (`tuning_dkdv.plan`), except where that (head_dim
    rung, KV rung) pair has no geometry and the next rung up does: then that
    rung, with `padded_head` and `widened` set. Only head_dim 32 at `block_kv`
    256 needs it today. The wider tile masks the surplus columns with no floor,
    the same as a caller pinning `block_dmodel`, so the d-contracted GEMMs do
    twice the work they need; the alternative is the split path.
    """
    meta = dkdv_plan(request)
    if request.block_dmodel is not None:
        return meta, False
    block_kv = kv_block_for(max_seqlen_k)
    if (meta.block_dmodel, block_kv) in _GEOMETRY:
        return meta, False
    for rung in sorted({d for d, kv in _GEOMETRY if kv == block_kv}):
        if rung > meta.block_dmodel:
            return replace(meta, block_dmodel=rung, block_dmodel_v=rung, padded_head=True), True
    return meta, False


@dataclass(frozen=True)
class BwdFusedTraits(BwdDkDvTraits):
    """The dK/dV traits plus the fused kernel's three extra LDS regions.

    Every field of the parent keeps its meaning (see `BwdDkDvTraits` for the
    role swap: `BLOCK_KV` is the KV rows a workgroup owns, `BLOCK_Q` the Q rows
    of a streamed tile). The additions are layout only: where the K tile, the
    dS^T tile and the two row arrays live, in bf16 elements from the start of
    the one allocation.

    **The K and dS^T regions reuse the stream tile's LDS formula**
    (`mfma16.lds_elem`) in tiles of `BLOCK_Q` tokens, so the
    transpose read that serves them is the published, probed one rather than
    a new lane map. The formula depends on the granule and `BLOCK_Q` only, not
    on the tile's width, which is what lets the dS^T tile -- `BLOCK_Q` wide
    whatever the head dim -- use it too.
    """

    Q_ROWS_CAP: int = 0

    @property
    def KV_TILES(self):
        """`BLOCK_Q`-token tiles the K and dS^T regions are each cut into."""
        return self.BLOCK_KV // self.BLOCK_Q

    @property
    def DS_TILE_ELEMS(self):
        """LDS elements of one `BLOCK_Q`-token, `BLOCK_Q`-wide dS^T tile."""
        return self.SMEM_N_RPT * (self.BLOCK_Q // self.D_128B_SIZE) * self.SMEM_V_LINE_STRIDE

    @property
    def K_REGION_BASE(self):
        return self.LDS_STREAM_TOTAL_SIZE

    @property
    def DS_REGION_BASE(self):
        return self.K_REGION_BASE + self.KV_TILES * self.STREAM_TILE_ELEMS

    @property
    def ROWS_REGION_BASE(self):
        """First element of the f32 row arrays: `-log2e*LSE` then `delta`."""
        return self.DS_REGION_BASE + self.KV_TILES * self.DS_TILE_ELEMS

    @property
    def LDS_TOTAL_ELEMS(self):
        # Two f32 arrays of `Q_ROWS_CAP`, at two bf16 elements per f32.
        return self.ROWS_REGION_BASE + 2 * 2 * self.Q_ROWS_CAP

    @property
    def LDS_TOTAL_BYTES(self):
        return self.LDS_TOTAL_ELEMS * self.BF16_BYTES


@dataclass(frozen=True)
class BwdFusedKnobs(BwdFusedKnobFields):
    """How to compute it. `None` means "policy decides"."""

    def resolve(self, meta: FmhaInputMetadata) -> "BwdFusedKnobs":
        """The complete build configuration for `meta` at this `block_kv`.

        `block_kv` must be set: it is the KV rung, which `FmhaInputMetadata`
        cannot say. `entry_plan` derives it from `max_seqlen_k`.
        """
        floor = BwdDkDvKnobs(hdim_qk_floor=self.hdim_qk_floor)._checked_scope(meta)._with_widths(meta).hdim_qk_floor
        out = replace(_FALLBACK.merge(self), hdim_qk_floor=floor)
        return out._with_geometry(meta)._with_rows_cap(meta)._with_occupancy(meta)._checked_against_traits(meta)

    def _with_geometry(self, meta):
        block_kv = self.block_kv
        if block_kv is None:
            raise ValueError("block_kv is the KV rung and must be pinned: entry_plan derives it from max_seqlen_k")
        block_dmodel = meta.block_dmodel
        key = (block_dmodel, block_kv)
        if key not in _GEOMETRY and None in (self.mfma_rows, self.num_waves, self.block_q):
            raise ValueError(
                f"no fused geometry for head_dim rung {block_dmodel} at block_kv {block_kv}; "
                "the K tile, the dS^T tile and the dK/dV accumulators do not all fit, so this shape "
                "belongs to the split path"
            )
        rows, waves, wpe, bq, nbuf = _GEOMETRY.get(key, (None,) * 5)
        rows = self.mfma_rows if self.mfma_rows is not None else rows
        waves = self.num_waves if self.num_waves is not None else block_kv // rows
        if rows * waves != block_kv:
            raise ValueError(f"mfma_rows {rows} x num_waves {waves} must equal block_kv {block_kv}")
        return replace(
            self,
            block_kv=block_kv,
            mfma_rows=rows,
            num_waves=waves,
            waves_per_eu=self.waves_per_eu if self.waves_per_eu is not None else wpe,
            block_q=self.block_q if self.block_q is not None else bq,
            num_stream_buffers=self.num_stream_buffers if self.num_stream_buffers is not None else nbuf,
            # The dK/dV policy's choice for the rung (`_TIGHT_REGISTERS`): only
            # head_dim 32 wants the tight arm.
            tight_registers=self.tight_registers if self.tight_registers is not None else block_dmodel == 32,
            head_dim_granule=(
                self.head_dim_granule
                if self.head_dim_granule is not None
                else (64 if block_dmodel % 64 == 0 else 32)
            ),
        )

    def _with_rows_cap(self, meta):
        """Round `q_rows_cap` up so the prologue's passes and the tile walk both fit.

        The walk can run one tile past the last real one (the body consumes
        two tiles per iteration) and a causal walk can start on an odd tile, so
        the cap carries two tiles of slack over the ceiling it was asked for.
        """
        quantum = math.lcm(self.block_q, rows_per_pass(self.num_waves, meta.block_dmodel_v))
        cap = -(-(self.q_rows_cap + 2 * self.block_q) // quantum) * quantum
        return replace(self, q_rows_cap=cap)

    def _with_occupancy(self, meta):
        """`waves_per_eu` from the LDS footprint, unless the table or the caller pinned it.

        **The register budget has to match the occupancy LDS already allows**,
        or the backend spends the difference. With `waves_per_eu=1` LLVM
        budgets the whole 512-entry file and has no reason to stop at the
        number that would let the LDS-permitted workgroups co-reside --
        measured at head_dim 64, causal, `block_kv` 128: 160 registers (one
        workgroup per CU) and 1466 us, against 128 registers (two) and 1187 us
        when asked for four waves per SIMD. So the hint is the LDS occupancy
        expressed in waves per SIMD, and where registers cannot meet it (head
        dim 128) the backend says so and the build is no worse than before.
        """
        if self.waves_per_eu is not None:
            return self
        lds = self.build_traits(meta, check_lds=False).LDS_TOTAL_BYTES
        per_cu = max(1, LDS_CAP_BYTES // lds)
        by_lds = per_cu * self.num_waves // 4
        by_regs = 512 // register_demand(self.mfma_rows, meta.block_dmodel, meta.dropout)
        return replace(self, waves_per_eu=max(1, min(8, by_lds, by_regs)))

    def dkdv_knobs(self):
        """The dK/dV knob object this build's stream and resident geometry is."""
        return BwdDkDvKnobs(
            hdim_qk_floor=self.hdim_qk_floor,
            num_waves=self.num_waves,
            block_kv=self.block_kv,
            block_q=self.block_q,
            head_dim_granule=self.head_dim_granule,
            dkv_shards=1,
            mfma_rows=self.mfma_rows,
            num_stream_buffers=self.num_stream_buffers,
            waves_per_eu=self.waves_per_eu,
            tight_registers=self.tight_registers,
            strides_constexpr=self.strides_constexpr,
        )

    def _checked_against_traits(self, meta):
        self.build_traits(meta)
        return self.with_flyc_fields()

    def build_traits(self, meta, check_lds=True):
        """The traits this configuration implies. The one knob->traits map."""
        knobs = self if self.waves_per_eu is not None else replace(self, waves_per_eu=1)
        base = knobs.dkdv_knobs().build_traits(meta)
        if self.block_kv % self.block_q:
            raise ValueError(f"block_kv {self.block_kv} must be a multiple of block_q {self.block_q}")
        if self.block_q % 32:
            raise ValueError(f"block_q {self.block_q} must be a multiple of 32")
        names = tuple(base.LDS_SCOPE_NAMES) + (SCOPE_K, SCOPE_DS, SCOPE_ROWS)
        traits = BwdFusedTraits(
            **{f.name: getattr(base, f.name) for f in fields(base) if f.name != "LDS_SCOPE_NAMES"},
            LDS_SCOPE_NAMES=names,
            Q_ROWS_CAP=self.q_rows_cap,
        )
        if check_lds and traits.LDS_TOTAL_BYTES > LDS_CAP_BYTES:
            raise ValueError(
                f"the fused kernel needs {traits.LDS_TOTAL_BYTES} B of LDS at head_dim rung "
                f"{meta.block_dmodel}, block_kv {self.block_kv}, block_q {self.block_q}, "
                f"{self.num_stream_buffers} stream buffer(s) and q_rows_cap {self.q_rows_cap}; the cap is "
                f"{LDS_CAP_BYTES}. Drop a stream buffer or lower q_rows_cap."
            )
        return traits


# Defaults the policy has no shape-dependent opinion about.
_FALLBACK = BwdFusedKnobs(
    strides_constexpr=False,
    q_rows_cap=512,
)


def bwd_fused_knobs(arch="gfx950", **overrides):
    """Knobs for `arch`, with `overrides` pinned."""
    if arch != "gfx950":
        raise ValueError(f"this tuning module is gfx950-only, got {arch!r}")
    return BwdFusedKnobs(**overrides)


# --- dispatch: fused or split ------------------------------------------------
#
# Measured on one MI355X (gfx950), bf16, H=16, against the split flyc path
# (Triton `bwd_preprocess` + flyc dK/dV + flyc dQ, each by its own policy),
# same process, arms alternating. Fixed-FLOPs batch (B*H*Sq*Sk = 4*8*4096^2),
# speedup = split / fused:
#
#   Sq=Sk         16    32    64    96   128   192   256
#   hd 32  nc   3.83  3.99  3.13  1.82  1.81  0.32  0.37   (192/256: widened to rung 64)
#   hd 32  c    3.36  3.43  2.96  1.71  1.71  0.30  0.34
#   hd 64  nc   3.99  3.25  2.51  1.58  1.52  1.02  1.11
#   hd 64  c    2.50  2.46  2.11  1.72  1.59  0.59  0.78
#   hd 128 nc   1.64  1.70  1.70  1.29  1.33     -     -   (no fused geometry)
#   hd 128 c    1.73  1.66  1.71  1.35  1.38     -     -
#
# Sq != Sk follows Sk: 512x64 2.05-3.10x, 256x128 1.47-2.19x, 64x128
# 1.38-2.35x; 128x256 0.29-0.98x. GQA widens the margin (Hk=4: 1.55-2.99x,
# Hk=1: 1.60-3.15x at 64 and 128).
#
# **Under-fill does not flip it at Sk <= 128.** B*H from 16 to 4096 at S=64
# and 128, timed as kernel durations (rocprofv3; the split path's three
# durations summed, so its two extra launch gaps are *not* charged to it):
# fused is 1.12x-2.33x at every point, smallest at head_dim 128, S=128, B*H
# <= 128. At these lengths the split kernels launch one KV block and one Q
# block per head as well, so they under-fill exactly as much as the fused
# one does; the fused kernel's single workgroup per (batch, KV head) only
# costs anything once the split path has several blocks per head -- which is
# the `block_kv` 256 rung, already excluded on its own numbers.
#
# What is excluded, and why:
# - `seqlen_k > 128`: the 256 rung is at best 1.11x (head_dim 64,
#   non-causal) and loses causal and at head_dim 32; head_dim 128 has no
#   geometry. A whole-sequence workgroup at 16 waves reads the streamed tile
#   once per 16 KV rows and is LDS-bound, and causal walks every Q tile
#   against every KV row.
# - `head_dim > 128`: no fused geometry (the K tile and dK/dV accumulators).
# - `seqlen_q` past the default `q_rows_cap`: a wider cap is a knob, but it
#   costs LDS and therefore co-resident workgroups, and was not measured.
#
# **Dropout is not excluded.** It used to be: its builds sit at the SGPR cliff
# and the lost-SGPR-copy backend defect corrupted the dK/dV epilogue's
# addressing, which every SGPR value lives across the whole walk to reach.
# `FusedKvOutputs` now forms that addressing in the prologue and carries it in
# VGPRs, and every dropout build in the matrix passes -- 112/112 builds both
# under flydsl 0.4.2 + LLVM PR 223465 and under stock 0.4.2, including the two
# that failed before (head_dim 32 `block_kv` 128 causal bf16; head_dim 96
# `block_kv` 64 non-causal f16).
# Re-run the matrix (`FEATS="dropout bias"`) on every FlyDSL re-sync or LLVM
# bump: a new failure is the condition for excluding that build here, keyed on
# the build (head_dim rung, kv tile, causal, dtype, dropout), not on shapes.
_FUSED_MAX_SEQLEN_K = 128
_FUSED_MAX_HEAD_DIM = 128


def prefer_fused(
    *,
    head_dim,
    seqlen_q,
    seqlen_k,
    head_dim_v=None,
    dropout=False,
    arch="gfx950",
):
    """True where the fused backward measured faster than the split path, and is safe to build.

    `seqlen_q`/`seqlen_k` are the *maximum* lengths of the call (the varlen
    `max_seqlen_*`), since the build serves every sequence up to them. Causal,
    windows, bias, GQA, varlen, bf16 and f16 do not change the answer: every
    measured point inside the region won, and every feature there is tested.
    Dropout does not change it either; `dropout` is accepted so a caller can
    pass the whole call, and so an exclusion can be keyed on it again.
    """
    # TODO: dropout is admitted only because `fused_bwd.FusedKvOutputs`
    # works around a gfx950 backend miscompile (see the FIXME there). That workaround
    # covers the dK/dV epilogue only; if a dropout build fails the 112-build matrix
    # after a FlyDSL or LLVM bump, exclude that exact build here (keyed on head_dim
    # rung, kv tile, causal, dtype, dropout) rather than all dropout.
    del dropout
    if arch != "gfx950":
        return False
    if max(head_dim, head_dim if head_dim_v is None else head_dim_v) > _FUSED_MAX_HEAD_DIM:
        return False
    if seqlen_k > _FUSED_MAX_SEQLEN_K:
        return False
    block_q = 64
    rows_needed = (-(-seqlen_q // block_q) + 2) * block_q
    return rows_needed <= _FALLBACK.q_rows_cap + 2 * block_q


# Unified entry (flyc/tuning_fused_bwd.py): the shared metadata, pinned.

def entry_resolve(meta, *, overrides=None) -> BwdFusedKnobs:
    """`overrides` must pin `block_kv`: the metadata has no KV extent."""
    return pinned(BwdFusedKnobs,
                  overrides,
                  # The tile comes from the caller: no floor.
                  hdim_qk_floor=0,
                  # Strides are runtime arguments: one binary serves every layout.
                  strides_constexpr=False).resolve(meta)


def entry_plan(meta, *, max_seqlen_k, overrides=None):
    meta, widened = plan(meta, max_seqlen_k)
    pins = BwdFusedKnobs(block_kv=kv_block_for(max_seqlen_k), hdim_qk_floor=0 if widened else None)
    knobs = BwdFusedKnobs().merge(pins).merge(knobs_of(BwdFusedKnobs, overrides)).resolve(meta)
    if max_seqlen_k > knobs.block_kv:
        raise ValueError(f"max_seqlen_k {max_seqlen_k} does not fit block_kv {knobs.block_kv}")
    return meta, knobs


def entry_build(meta, knobs: BwdFusedKnobs):
    from .fused_bwd import build_fmha_bwd_fused_gfx950_module_primary
    return build_fmha_bwd_fused_gfx950_module_primary(meta, knobs)
