# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""dK / dV backward attention for gfx1201 -- the transpose of the forward loop.

A port of AOTriton's `bwd_kernel_dk_dv` + `bwd_inner_dk_dv`, against **our**
forward kernel's ABI (`flash_attn_func_gfx1201_aiw.py`): BHSD shapes, three
strides per tensor, `fx.Pointer` arguments, VarlenBits addressing, and the
logsumexp layout `fmha.lse_token_pitch` describes. AOTriton's `cu_seqlens_q/k`
+ `batch_size` logic is deliberately *not* ported; the varlen prologue below
is byte-for-byte the forward's.

`sdpa_dkdv_dataflow_gfx1201.svg` draws all of the below as one picture: the
VRAM / LDS / VGPR dataflow for a single workgroup, in the wide configuration,
with the narrower ones as annotated special cases.

--- Why the loop is inverted, and what that costs -------------------------

The forward preloads Q into registers and streams K/V through LDS. dK/dV needs
the opposite: a K/V tile is resident and Q/dO stream past it, because dK and dV
accumulate over *all* query rows.

    forward   Q resident  (per wave: 16 q rows)   K, V   stream through LDS
    dK/dV     K,V resident (per wave: 16 kv rows) Q, dO  stream through LDS

`Aperture`, `stage`, `publish_transposed`, `TransposedTiling`, `MaskedAxis`,
`make_addr_pair` and `reader` all transfer unchanged with the roles swapped --
they were written against "a tensor, its bounds and where it lands", not
against K or V specifically. Three things did not transfer and are local to
this file; they are named in the report.

**One wave owns 16 KV rows**, so `BLOCK_N = 16 * NUM_WAVES` and dK/dV
accumulators stay wave-private. The alternative -- every wave over the same KV
tile, splitting the Q stream -- would need a `BLOCK_N x head_dim` f32 reduction
through LDS per workgroup, which does not fit at any useful width.

--- The four GEMMs and why Q/dO are staged twice --------------------------

WMMA 16x16x16, wave32. An A operand's eight per-lane elements run along the
*contraction* axis `k'` (lane holds `A[m'=lane16][k'=klane*8+j]`); a B operand's
run along it too (`B[k'=klane*8+j][n'=lane16]`); a result element `si` is
`D[m'=klane*8+si][n'=lane16]`.

    S   [q][kv]  = Q  . K^T    A = Q  (contract d)   B = k_pack   (resident)
    dP  [q][kv]  = dO . V^T    A = dO (contract d)   B = v_pack   (resident)
    dV^T[d][kv] += dO^T . P    A = dO^T (contract q) B = P
    dK^T[d][kv] += Q^T  . dS   A = Q^T  (contract q) B = dS

So Q is contracted over `d` once and over `q` once, and so is dO. An A operand
contracted over `d` wants eight contiguous `d` at one `q` -- a row-major tile.
Contracted over `q` it wants eight contiguous `q` at one `d` -- a transposed
tile. Hence **four LDS tiles**, and `fmha_tuning_bwd_dkdv_gfx1201.lds_bytes` is
what bounds `block_m`. Deriving one orientation from the other in LDS is the
forward's `V_LDS_LAYOUT="row"` path: eight strided scalar reads per operand,
which it measured 2.7% slower for a single tensor and would pay here twice --
that is `transposed_source="derived"` below, and above head_dim 256 the trade
goes the other way because the four tiles simply do not fit.

Two layout facts fall out and are worth stating because they are what makes the
port cheap:

- **K and V are read exactly as the forward reads Q** -- `K[kv=lane16][d
  contiguous]` is a B operand for S with no transpose anywhere.
- **P and dS need no repacking.** `S` is produced with `m' = q`, so a lane's
  eight result elements are eight consecutive `q` at one `kv` -- which is
  precisely the B-operand layout the dV/dK GEMMs want. Choosing `m' = kv`
  instead (the forward's orientation for its S) would have forced a transpose
  of P through LDS every iteration.
- **dK^T / dV^T come out with `d` per-lane-contiguous**, so the epilogue is one
  `v8` store per accumulator, the same shape as the forward's O store. That is
  why the GEMMs compute the transposes rather than dK/dV directly.

--- Regions ---------------------------------------------------------------

`fmha.decompose_causal_regions` cuts a Q block's KV range into
`[masked][full][masked]`. Its transpose -- a KV block's Q range -- is *the same
function with the axes swapped*:

    decompose_causal_regions(start_k, k_len, q_len, w_right, w_left,
                             BLOCK_N, BLOCK_M, alive)

Every line of it maps: `q_hi` becomes `k_hi`, `blk_last` becomes the last Q
block, `l_first_full` becomes the first fully-live Q block, and `left_col0`
becomes the first visited Q row. Checked term by term against AOTriton's
`calculate_intervals`, which `bwd_kernel_dk_dv` also calls with the axes
swapped (it passes `mask_on_seq_k` into the `mask_on_seq_q` slot).

**This build walks one always-masked loop over `[v_lo, v_hi]`**, using the
decomposition only for the visited *range*, not for the full/masked split. The
runs are contiguous by construction, so `left_col0` and `n_left + n_full +
n_right` describe the whole range whichever of them is empty. Emitting the
split as two loops -- what the forward does -- doubles a body that already
carries four WMMA chains, and correctness came first. See the report.

Non-causal goes through the same path with `window_left = window_right =
max(seqlen_q, seqlen_k)`, which makes the visited range the whole sequence and
the leading masked run empty. One code path, and the tail masks are the same
ones causal needs anyway.

--- Wide head dims: `contraction_shards` and `transposed_source` ----------

Per lane, wave32, one wave owning 16 KV rows, an f16 operand tile is
`head_dim/4` VGPRs and an f32 accumulator `head_dim/2`. Measured ISA gives
`vgpr = state + 62`, and the model reproduces the spill counts it can be
checked against (head_dim 256 unsplit predicts 190, ISA reports 203). Above
head_dim 256 *both* budgets fail, and BLOCK_M is already at the WMMA tile
floor of 16, so neither is reachable by tuning:

    head_dim 512, split alone   packs 128 + accs 256 = 384  (+62 = 446 / 256)
    head_dim 512, four tiles                            90496 / 65536

**`contraction_shards`** shards the contraction as well as the role. A team is
`2C` waves and wave `(c, r)` holds only `K[:, c*D/C : (c+1)*D/C]` (r=0) or the
V equivalent, so state falls to `0.75 * head_dim / C` -- 384, 192, 96 at C =
1, 2, 4. Each wave then holds a *partial* S or dP, so the two-slot relay
becomes a `2C`-slot reduction. It does not grow the exchange, which is one slot
per wave either way.

**`transposed_source="derived"`** drops the two transposed tiles. They were
never performing a transpose -- `global_load_tr_b128` does that on the way in
from global -- they are a redistribution shuffle, because the cooperative
load's block-to-wave assignment does not match the d-ownership the GEMMs
consume by. So the same eight halves are read strided out of the row-major
tile: eight `ds_read_u16` against one `ds_read_b128`, since gfx1201 has no
`ds_load_tr16_b128` (gfx1250 does). At head_dim 512 that is the 40960 B which
makes the difference between 90496 and 49536.

Both are off below head_dim 320 and both are measured losses there -- C=2
costs 15-30% wherever C=1 does not spill, because a wider team means a
narrower `BLOCK_N` and more Q/dO re-reads. `fmha_tuning_bwd_dkdv_gfx1201`
holds the numbers and the policy.

--- Not implemented -------------------------------------------------------

- No bias. `BIAS_TYPE` has no dB output here and AOTriton computes it in a
  separate kernel.
- `delta = rowsum(dO * O)` is the caller's, computed in torch. AOTriton has a
  fused `bwd_preprocess`; that is a later optimisation.
"""

import math as host_math
from dataclasses import fields

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
from fmha_tuning_bwd_dkdv_gfx1201 import (  # noqa: F401
    LDS_PAD,
    ROWS_PER_WAVE,
    BwdDkDvKnobs,
    BwdDkDvMetadata,
    resolve_knobs,
)
from gfx1201_standalone import buffer_ops
from gfx1201_standalone import utils as common_utils
from philox import Philox

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_dkdv_gfx1201_kernel"
_LOG2E = host_math.log2(host_math.e)

# Causal alignment as a window sentinel, resolved per sequence in the kernel by
# `fmha.resolve_window`. The host only emits them.
_WINDOW_TOPLEFT = fmha.WINDOW_TOPLEFT
_WINDOW_BOTRIGHT = fmha.WINDOW_BOTRIGHT


# ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
#
# **These are transcriptions of the forward's, and the two must not drift.**
# They live at module scope here rather than inside the builder because
# nothing in them depends on a knob -- the forward nests them only because its
# `_varlen_args` rejects the combination with `STRIDES_CONSTEXPR`, which this
# kernel does not have. Folding the two copies into one shared module is the
# right end state and is named in the report; until then, a backward caller
# may equally pass the dict the *forward's* `exe.varlen_compact(...)` returns,
# which is the cheapest way to be sure the two agree.


_COMPILED = abi.new_compiled_cache()


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, fmha.llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(fmha.llvm_value(value), fmha.llvm_value(ptr))


def build_bwd_dkdv_module_primary(meta: BwdDkDvMetadata, knobs: BwdDkDvKnobs):
    """Build the gfx1201 dK/dV kernel for one `(metadata, knobs)` pair.

    Both must be fully resolved -- pass `resolve_knobs(meta)` or, from a
    caller that only knows a head_dim, `plan(meta).knobs`. Nothing here falls
    back to a policy; an unresolved knob is a caller error.
    """
    num_heads = meta.num_heads
    causal = meta.causal
    dtype_str = meta.dtype_str
    sm_scale = meta.sm_scale
    causal_type = meta.causal_type
    dropout = meta.dropout
    philox_width = meta.philox_width

    BLOCK_DMODEL = knobs.block_dmodel
    BLOCK_DMODEL_V = knobs.block_dmodel_v
    BLOCK_M = knobs.block_m
    NUM_WAVES = knobs.num_waves
    LPT_TILE_ORDER = knobs.lpt_tile_order
    SPLIT_HEAD_DIM = knobs.split_head_dim
    PADDED_HEAD = knobs.padded_head
    WAVES_PER_EU = knobs.waves_per_eu
    SCHED_STRATEGY = knobs.sched_strategy
    FP_MODE = knobs.fp_mode
    DENORMALS_ARE_ZERO = knobs.denormals_are_zero
    UNSAFE_FP_MATH = knobs.unsafe_fp_math
    FAST_FP_MATH = knobs.fast_fp_math
    ADDR_HOIST = knobs.addr_hoist

    # ---- WMMA / wave32 constants ----
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    # K elements each lane holds of an A/B operand: `WMMA_K / (WARP_SIZE /
    # WMMA_M)`. Two lanes share each row and split the K extent.
    WMMA_LANE_K = WMMA_K // (WARP_SIZE // WMMA_M)

    CONTRACTION_SHARDS = knobs.contraction_shards
    TRANSPOSED_SOURCE = knobs.transposed_source
    KEEP_TR_TILES = TRANSPOSED_SOURCE == "tile"
    assert TRANSPOSED_SOURCE in (
        "tile",
        "derived",
    ), f"transposed_source must be 'tile' or 'derived', got {TRANSPOSED_SOURCE!r}"

    # A *team* of waves cooperates on one set of 16 KV rows. Three regimes, one
    # expression, which is why `split_head_dim` and `contraction_shards`
    # compose rather than branching against each other:
    #
    #   split off      TEAM_WAVES 1   one wave owns its KV rows outright
    #   split, C=1     TEAM_WAVES 2   two roles: K/S and V/dP
    #   split, C>1     TEAM_WAVES 2C  those two roles times C contraction shards
    #
    # `NUM_TEAMS` is what used to be `NUM_PAIRS`, and at C=1 it is exactly that.
    if SPLIT_HEAD_DIM:
        assert BLOCK_DMODEL == BLOCK_DMODEL_V, (
            f"split_head_dim gives both waves one operand array and one LDS stride, "
            f"so it needs head_dim == head_dim_v; got {BLOCK_DMODEL} and {BLOCK_DMODEL_V}"
        )
        assert (
            CONTRACTION_SHARDS >= 1 and (CONTRACTION_SHARDS & (CONTRACTION_SHARDS - 1)) == 0
        ), f"contraction_shards must be a power of two, got {CONTRACTION_SHARDS}"
    else:
        assert (
            CONTRACTION_SHARDS == 1
        ), f"contraction_shards {CONTRACTION_SHARDS} needs split_head_dim: it shards the roles' contraction"
    TEAM_WAVES = 2 * CONTRACTION_SHARDS if SPLIT_HEAD_DIM else 1
    assert NUM_WAVES % TEAM_WAVES == 0, f"num_waves {NUM_WAVES} must be a multiple of the {TEAM_WAVES}-wave team"
    NUM_TEAMS = NUM_WAVES // TEAM_WAVES
    BLOCK_N = ROWS_PER_WAVE * NUM_TEAMS
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE

    D_STEPS = BLOCK_DMODEL // WMMA_K  # k_packs, dk accumulators
    DV_STEPS = BLOCK_DMODEL_V // WMMA_K  # v_packs, dv accumulators
    Q_SUBTILES = BLOCK_M // WMMA_M

    # ---- Validity predicate over the knob space ----
    #
    # Assertions and not ValueError, as in the forward: every name here was
    # produced by `fmha_tuning_bwd_dkdv_gfx1201`, so a violation is that module
    # contradicting itself. Caller input is checked in the interface.
    assert (
        BLOCK_DMODEL % 16 == 0 and BLOCK_DMODEL > 0
    ), f"BLOCK_DMODEL must be a positive multiple of 16, got {BLOCK_DMODEL}"
    assert (
        BLOCK_DMODEL_V % 16 == 0 and BLOCK_DMODEL_V > 0
    ), f"BLOCK_DMODEL_V must be a positive multiple of 16, got {BLOCK_DMODEL_V}"
    assert BLOCK_M % WMMA_M == 0 and BLOCK_M > 0, f"BLOCK_M must be a positive multiple of {WMMA_M}, got {BLOCK_M}"
    assert NUM_WAVES >= 1, f"NUM_WAVES must be at least 1, got {NUM_WAVES}"
    assert dtype_str in ("f16", "bf16"), f"dK/dV supports f16/bf16, got {dtype_str!r}"

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(BLOCK_DMODEL)

    # Causal vocabulary, identical to the forward's: 0 none, 1 top-left, 2
    # bottom-right, 3 explicit window. The kernel only ever sees 0 or 3; 1 and
    # 2 resolve to a sentinel window on the host so there is one way to express
    # a diagonal rather than two free to drift.
    if causal_type is None:
        HOST_CAUSAL_TYPE = 1 if causal else 0
    else:
        HOST_CAUSAL_TYPE = causal_type
    if HOST_CAUSAL_TYPE not in (0, 1, 2, 3):
        raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {HOST_CAUSAL_TYPE}")
    if bool(HOST_CAUSAL_TYPE) != bool(causal):
        raise ValueError(f"causal={causal} disagrees with causal_type={HOST_CAUSAL_TYPE}")
    CAUSAL_TYPE = 0 if HOST_CAUSAL_TYPE == 0 else 3
    CAUSAL = bool(CAUSAL_TYPE)

    ENABLE_DROPOUT = bool(dropout)
    PHILOX = Philox.for_arch() if philox_width is None else Philox(width=philox_width)

    # Attention bias. An input only -- dB is the dQ kernel's, for the layout
    # reason `BwdDkDvMetadata.bias` records.
    BIAS_TYPE = 1 if meta.bias else 0
    assert not (BIAS_TYPE and CAUSAL), "bias and causal are mutually exclusive, as in the forward"

    NUM_HEADS = num_heads  # noqa: F841  (kept for parity with the forward's metadata)

    fastmath = fmha.FastMath(FP_MODE)

    # ---- LDS layout ----
    #
    # Four tiles; see the module docstring for why both orientations exist.
    # Padding is the forward's `_LDS_PAD`, for the same bank-spreading reason.
    QRM_STRIDE = BLOCK_DMODEL + LDS_PAD
    QTR_STRIDE = BLOCK_M + LDS_PAD
    DORM_STRIDE = BLOCK_DMODEL_V + LDS_PAD
    DOTR_STRIDE = BLOCK_M + LDS_PAD

    # Under `transposed_source="derived"` the two transposed tiles are not
    # allocated and their operands are read strided out of the row-major pair.
    # Sizes go to zero rather than the bases being skipped, so every downstream
    # offset expression is the one the four-tile layout has.
    QRM_BASE = 0
    QRM_SIZE = BLOCK_M * QRM_STRIDE
    QTR_BASE = QRM_BASE + QRM_SIZE
    QTR_SIZE = (BLOCK_DMODEL * QTR_STRIDE) if KEEP_TR_TILES else 0
    DORM_BASE = QTR_BASE + QTR_SIZE
    DORM_SIZE = BLOCK_M * DORM_STRIDE
    DOTR_BASE = DORM_BASE + DORM_SIZE
    DOTR_SIZE = (BLOCK_DMODEL_V * DOTR_STRIDE) if KEEP_TR_TILES else 0
    # Per-row LSE and delta for the Q tile in flight, staged once per
    # iteration and read by every wave. Two BLOCK_M-long f32 runs, expressed in
    # 16-bit elements because the whole LDS block is typed to the kernel's
    # element type. 16-byte aligned so the reads can widen later.
    ROWQ_BASE = ((DOTR_BASE + DOTR_SIZE) + 7) // 8 * 8
    ROWQ_ELEMS = 2 * BLOCK_M * 2  # f32 -> two 16-bit slots each
    ROWQ_BYTE0 = ROWQ_BASE * 2
    # The team exchange, only under `split_head_dim`. LDS is typed in 16-bit
    # elements but `lds_f32_*` index in f32; the _F32 names are the indexing
    # unit and PDS_ELEMS the block size. Mixing them is a silent 2x overrun.
    #
    # One slot per wave whatever the team width -- two waves relaying S and dP,
    # or `2C` waves reducing partials, is `NUM_WAVES` slots either way.
    PDS_BASE = (ROWQ_BASE + ROWQ_ELEMS + 7) // 8 * 8
    PDS_SLOT_F32 = WARP_SIZE * 8
    PDS_TEAM_F32 = TEAM_WAVES * PDS_SLOT_F32
    PDS_ELEMS = (NUM_TEAMS * PDS_TEAM_F32 * 2) if SPLIT_HEAD_DIM else 0
    PDS_BYTE0 = PDS_BASE * 2
    _ROWQ_BATCHES = (BLOCK_M + BLOCK_SIZE - 1) // BLOCK_SIZE
    LDS_TOTAL = PDS_BASE + PDS_ELEMS

    # A wave accumulates 1/TEAM_WAVES of the d range and, when sharded,
    # contracts over 1/CONTRACTION_SHARDS of it. The two partitions are
    # independent: each only has to *be* a partition.
    assert (
        D_STEPS % TEAM_WAVES == 0 and DV_STEPS % TEAM_WAVES == 0
    ), f"a {TEAM_WAVES}-wave team needs D_STEPS/DV_STEPS divisible by {TEAM_WAVES}"
    D_STEPS_OWN = D_STEPS // TEAM_WAVES
    DV_STEPS_OWN = DV_STEPS // TEAM_WAVES
    D_STEPS_SHARD = D_STEPS // CONTRACTION_SHARDS

    # Cooperative-load vector width, in elements. Fixed at 8 for the alignment
    # reason the forward records: the D-axis pitch is guaranteed a multiple of
    # 16 bytes and nothing more, so a 16-element load would be over-promising.
    VEC_WIDTH = 8

    def _load_geom(width):
        return fmha.load_geom(width, VEC_WIDTH, BLOCK_SIZE, BLOCK_M)

    QRM_TPR, QRM_RPB, QRM_BATCHES, QRM_GUARD = _load_geom(BLOCK_DMODEL)
    DORM_TPR, DORM_RPB, DORM_BATCHES, DORM_GUARD = _load_geom(BLOCK_DMODEL_V)

    # `global_load_tr_b128` transposes an 8x8 block of 16-bit elements across
    # each group of 8 lanes, so one wave-wide load produces a 16(d) x 16(q)
    # block already in WMMA-operand order. Spread those blocks over the waves;
    # the tiling need not divide evenly, tail tiles are guarded at the store.
    QTR_D_BLOCKS = BLOCK_DMODEL // WMMA_N
    _QTR_TILES = QTR_D_BLOCKS * (BLOCK_M // WMMA_K)
    QTR_LOADS = (_QTR_TILES + NUM_WAVES - 1) // NUM_WAVES
    QTR_GUARD = QTR_LOADS * NUM_WAVES != _QTR_TILES

    DOTR_D_BLOCKS = BLOCK_DMODEL_V // WMMA_N
    _DOTR_TILES = DOTR_D_BLOCKS * (BLOCK_M // WMMA_K)
    DOTR_LOADS = (_DOTR_TILES + NUM_WAVES - 1) // NUM_WAVES
    DOTR_GUARD = DOTR_LOADS * NUM_WAVES != _DOTR_TILES

    elem_numeric_cls = abi.dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        qdo: fx.Array[elem_numeric_cls, LDS_TOTAL, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def bwd_dkdv_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls
        f32_ty = ir.F32Type.get()

        def _to_global_ptr_i64(ptr):
            return fx.as_ir_value(fx.Int64(fx.ptrtoint(ptr)))

        q_ptr = fmha.pointer_to_llvm_ptr(Q)
        q_ptr_i64 = _to_global_ptr_i64(Q)
        k_ptr = fmha.pointer_to_llvm_ptr(K)
        v_ptr = fmha.pointer_to_llvm_ptr(V)
        do_ptr = fmha.pointer_to_llvm_ptr(DO)
        do_ptr_i64 = _to_global_ptr_i64(DO)
        if const_expr(BIAS_TYPE):
            bias_ptr = fmha.pointer_to_llvm_ptr(B)
        dk_ptr = fmha.pointer_to_llvm_ptr(DK)
        dv_ptr = fmha.pointer_to_llvm_ptr(DV)
        l_ptr = fmha.pointer_to_llvm_ptr(LSE)
        delta_ptr = fmha.pointer_to_llvm_ptr(Delta)

        v8f16_type = Vec.make_type(8, elem_dtype)
        vxf16_type = Vec.make_type(VEC_WIDTH, elem_dtype)

        # ---- Varlen prologue: VarlenBits -> six scalars ----
        #
        # Verbatim the forward's, which is the point: the two kernels must
        # agree about where a sequence lives, and the only way to guarantee
        # that is to call the same decoder with the same arguments.
        z_i32 = fx.Int32(gpu.block_idx.z)
        seqlen_q_i32, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32
        )
        seqlen_k_i32, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32
        )
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        seqlen_q_v = fx.Index(seqlen_q_i32)
        seqlen_k_v = fx.Index(seqlen_k_i32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_qdo = lds.qdo.ptr
        # `ptrtoint` on a shared pointer yields the 32-bit LDS byte offset,
        # which is what the f32 scratch helpers index from.
        _lds_byte_base = fx.as_ir_value(fx.ptrtoint(lds_qdo))

        tid, wave_id, lane, lane16, klane = fmha.wave_lanes(WARP_SIZE)

        # 3D grid: (kv_tile, head_k, batch). The KV tile goes on x, which
        # dispatches fastest, for the reason the forward gives about its q_tile
        # axis: under causal masking a workgroup's cost varies with this index
        # (a low KV block is visited by nearly every Q block, a high one by
        # few), so spreading it across scheduling groups gives each group a
        # mixed rather than a uniform duration.
        if const_expr(LPT_TILE_ORDER):
            # Max_seqlen_k, not this sequence's: the reversal must permute the
            # *grid*, whose x extent the host sized from Max_seqlen_k.
            _nkvt = (fx.Index(max_seqlen_k) + (BLOCK_N - 1)) // BLOCK_N
            kv_tile_idx = _nkvt - fx.Index(1) - fx.Index(gpu.block_idx.x)
        else:
            kv_tile_idx = fx.Index(gpu.block_idx.x)
        head_k = fx.Index(gpu.block_idx.y)
        start_k_i32 = fx.Int32(kv_tile_idx) * fx.Int32(BLOCK_N)
        start_k = fx.Index(start_k_i32)

        # Does this workgroup own any real key? Under varlen the grid's KV
        # extent is sized from Max_seqlen_k, so whole workgroups land past the
        # end of a shorter sequence.
        _alive = start_k_i32 < seqlen_k_i32
        # The base must be clamped and not merely the row within the tile: on a
        # packed tensor `row_off + start_k` rows in runs past the end of the
        # whole allocation, not merely past this sequence. The forward records
        # a 1.3 MB overshoot hitting an unmapped page on the Q side.
        _k_start_addr = fx.Index(start_k if _alive else fx.Index(0))

        _q_last_i32 = common_utils.smax(seqlen_q_i32 - fx.Int32(1), fx.Int32(0))
        _k_last_i32 = common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0))
        q_seq_last = fx.Index(_q_last_i32)
        k_seq_last = fx.Index(_k_last_i32)

        # MQA/GQA: this KV head is shared by `group` query heads, and dK/dV sum
        # over all of them. AOTriton loops `off_h_q` in the group; so does this
        # kernel, folded into the Q-block loop below.
        group_i32 = num_head_q // num_head_k

        # ---- Strides ----
        q_st = (fx.Index(stride_q_batch), fx.Index(stride_q_head), fx.Index(stride_q_seq))
        k_st = (fx.Index(stride_k_batch), fx.Index(stride_k_head), fx.Index(stride_k_seq))
        v_st = (fx.Index(stride_v_batch), fx.Index(stride_v_head), fx.Index(stride_v_seq))
        do_st = (fx.Index(stride_do_batch), fx.Index(stride_do_head), fx.Index(stride_do_seq))
        dk_st = (fx.Index(stride_dk_batch), fx.Index(stride_dk_head), fx.Index(stride_dk_seq))
        dv_st = (fx.Index(stride_dv_batch), fx.Index(stride_dv_head), fx.Index(stride_dv_seq))

        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        _k_row_off_v = fx.Index(k_row_off)

        sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))

        # ---- Column masking, PADDED_HEAD ----
        # One rule, exactly as the forward: an element is valid iff its column
        # is below hdim. A chunk starting past hdim is redirected to column 0
        # and masked away wholesale; a straddling chunk is loaded and masked per
        # element.
        qk_cols = fmha.MaskedAxis(fx.Index(hdim_qk), active=PADDED_HEAD, elem_dtype=elem_dtype)
        vo_cols = fmha.MaskedAxis(fx.Index(hdim_vo), active=PADDED_HEAD, elem_dtype=elem_dtype)

        def load_global_f16xN(base_ptr, base64, off32):
            return _pointer_load(vxf16_type, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def load_global_v8f16(base_ptr, base64, off32):
            return _pointer_load(v8f16_type, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def store_global_v8f16(base_ptr, base64, off32, val):
            _pointer_store(val, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def load_global_f16(base_ptr, off64):
            """One bias element.

            Scalar, and unavoidably so: this kernel runs the loop transposed,
            so a lane holds one kv column and eight q *rows*, and those eight
            bias entries are `stride_b_seq_q` apart rather than adjacent. The dQ
            kernel, walking the other axis, gets all eight in one v8. It is the
            same layout cost the dropout path here already pays, and the reason
            dB is emitted from dQ rather than from this kernel.
            """
            return _pointer_load(elem_type, buffer_ops.get_element_ptr(base_ptr, fx.Int64(off64), elem_type=elem_type))

        def load_global_f32(base_ptr, off64):
            """One f32 from a compact rank-2 tensor (logsumexp or delta).

            Scalar and not a vector load even though a lane's eight rows are
            consecutive: the row-group pitch is `lse_tokens`, which no contract
            makes a multiple of 8, so a v8f32 would claim 32-byte alignment the
            tensor does not have. That over-promise is the same failure
            `fmha.lds_load_v8` documents costing 2.2x on LDS, and here it is
            undefined behaviour rather than merely slow.
            """
            return _pointer_load(f32_ty, buffer_ops.get_element_ptr(base_ptr, fx.Int64(off64), elem_type=f32_ty))

        # ---- K and V: register-resident, one 16-row KV tile per wave ----
        #
        # Read exactly as the forward reads Q -- `K[kv = lane16][d contiguous]`
        # is already a WMMA B operand for S, with no transpose anywhere. The row
        # bound is the *real* seqlen_k, and it is never inactive.
        _addr_kw_k = dict(seqlen_k=seqlen_k_v, seq_last=k_seq_last, hoist=ADDR_HOIST, clamp=True)
        k_tbase, k_toff, _ = fmha.make_addr_pair(k_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k)
        v_tbase, v_toff, _ = fmha.make_addr_pair(v_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k)
        dk_tbase, dk_toff, _ = fmha.make_addr_pair(dk_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k)
        dv_tbase, dv_toff, _ = fmha.make_addr_pair(dv_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k)

        kv_rows = fmha.MaskedAxis(seqlen_k_v)
        k_ap = fmha.Aperture(qk_cols, rows=kv_rows)
        v_ap = fmha.Aperture(vo_cols, rows=kv_rows)

        # This wave's 16 KV rows, and this lane's row inside them.
        # Under `split_head_dim` a *pair* owns the 16 KV rows and the two waves
        # differ only in which half of the head dim they accumulate.
        # wave -> (team, slot), then slot -> (shard, role):
        #
        #   role   0 holds K and produces S, 1 holds V and produces dP
        #   shard  which 1/C of the contraction this wave reduces over
        #   slot   which 1/TEAM_WAVES of the d range it accumulates for
        #
        # At `TEAM_WAVES == 1` every index is 0 and this is the unsharded
        # kernel; at 2 it is `wave_id // 2` and `wave_id % 2`, the pair.
        group_of_wave = wave_id // fx.Index(TEAM_WAVES)
        _slot = wave_id % fx.Index(TEAM_WAVES)
        role = fx.Int32(_slot % fx.Index(2)) if const_expr(SPLIT_HEAD_DIM) else fx.Int32(0)
        _own_i32 = fx.Int32(_slot) * fx.Int32(D_STEPS_OWN)
        _vown_i32 = fx.Int32(_slot) * fx.Int32(DV_STEPS_OWN)
        kv_row_in_tile = group_of_wave * fx.Index(ROWS_PER_WAVE) + lane16
        kv_row_abs = start_k + kv_row_in_tile
        kv_row_abs_i32 = start_k_i32 + fx.Int32(group_of_wave) * fx.Int32(ROWS_PER_WAVE) + fx.Int32(lane16)
        if const_expr(SPLIT_HEAD_DIM):
            _pds_team = fx.Int32(group_of_wave) * fx.Int32(PDS_TEAM_F32) + fx.Int32(lane) * fx.Int32(8)
            _pds_mine = _pds_team + fx.Int32(_slot) * fx.Int32(PDS_SLOT_F32)
            _pds_theirs = _pds_team + (fx.Int32(1) - role) * fx.Int32(PDS_SLOT_F32)
            _rm_base_off = fx.Index(role) * fx.Index(DORM_BASE - QRM_BASE)
            # First contraction step this wave's packs cover, in WMMA_K units.
            _shard_k0 = fx.Int32(_slot // fx.Index(2)) * fx.Int32(D_STEPS_SHARD)

        k_tile_base = k_tbase(_k_start_addr)
        v_tile_base = v_tbase(_k_start_addr)

        def fetch_k(row, col):
            return load_global_v8f16(k_ptr, k_tile_base, k_toff(row, col))

        def fetch_v(row, col):
            return load_global_v8f16(v_ptr, v_tile_base, v_toff(row, col))

        # The index-typed row for the bounds test, deliberately -- building the
        # i32 copy early starts its live range early, which the forward
        # measured at 6% on its widest causal build. `gate` emits the signed
        # compare that `fx.Index` being unsigned would otherwise deny.
        _kv_in, _kv_safe = k_ap.rows.gate(kv_row_abs, kv_row_in_tile)

        if const_expr(SPLIT_HEAD_DIM):
            # Role 0 holds K (for S), role 1 holds V (for dP) -- D/4 registers
            # instead of D/2. The role is resolved ONCE, here, on plain values:
            # an aperture method call inside a ternary arm is `name.method(...)`
            # under a dynamic branch, the carried-state trap `fmha_common`'s
            # header documents, and it surfaces as a bad_alloc out of
            # `get_ir_types` rather than the documented UnboundLocalError. An
            # !llvm.ptr cannot cross an scf.if result either, so the pointer is
            # chosen as an i64 address and rebuilt.
            _role0 = role == fx.Int32(0)
            _v4_ty = Vec.make_type(4, elem_dtype)
            _my_addr = fx.Int64(fx.ptrtoint(K)) if _role0 else fx.Int64(fx.ptrtoint(V))
            my_ptr = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), fx.as_ir_value(_my_addr)).result
            _my_st = tuple(_a if _role0 else _b for _a, _b in zip(k_st, v_st))
            my_tbase, my_toff, _ = fmha.make_addr_pair(_my_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw_k)
            my_tile_base = my_tbase(_k_start_addr)

            # The column bound is the role's too. `BLOCK_DMODEL ==
            # BLOCK_DMODEL_V` is asserted above, so the two roles share a tile
            # width -- but the *runtime* extents need not agree, and reading V
            # against `hdim_qk` is wrong in both directions: with a wider V it
            # zeroes the columns in `[hdim_qk, hdim_vo)` whose dO is live, so
            # dP silently drops them; with a narrower one it leaves columns
            # past `hdim_vo` unzeroed, and they survive only because dO is zero
            # there. Selected on plain values for the reason above, before any
            # aperture exists to call a method on.
            if const_expr(PADDED_HEAD):
                _my_hdim = fx.Int32(hdim_qk) if _role0 else fx.Int32(hdim_vo)
                my_ap = fmha.Aperture(fmha.MaskedAxis(fx.Index(_my_hdim), elem_dtype=elem_dtype), rows=kv_rows)
            else:
                my_ap = k_ap

            def fetch_my(row, col):
                return load_global_v8f16(my_ptr, my_tile_base, my_toff(row, col))

            # `contraction_shards` cuts this again to the wave's own shard, so
            # the packs are `D/(4C)` registers. The column is constexpr at C=1
            # and only becomes a dynamic expression when there is a shard
            # offset to add -- the unsharded build must emit what it always did.
            k_packs = []
            for ks in range_constexpr(D_STEPS_SHARD):
                if const_expr(CONTRACTION_SHARDS > 1):
                    _c = fx.Index((_shard_k0 + fx.Int32(ks)) * fx.Int32(WMMA_K)) + klane * WMMA_LANE_K
                else:
                    _c = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                k_packs.append(my_ap.read_v8(fetch_my, _kv_safe, _c, _kv_in))
            v_packs = k_packs
        else:
            k_packs = []
            for ks in range_constexpr(D_STEPS):
                _c = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                k_packs.append(k_ap.read_v8(fetch_k, _kv_safe, _c, _kv_in))
            v_packs = []
            for ks in range_constexpr(DV_STEPS):
                _c = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                v_packs.append(v_ap.read_v8(fetch_v, _kv_safe, _c, _kv_in))

        # ---- Q and dO: the staged apertures ----
        #
        # Each tensor gets two, because the two orientations are two different
        # LDS placements and two different cooperative geometries. Holding each
        # on its own object is the forward's rule about per-tensor geometry
        # taken one step further -- a site that gates on a guard has to name
        # whose guard it is.
        q_rm_ap = fmha.Aperture(
            qk_cols,
            lds_base=QRM_BASE,
            lds_stride=QRM_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=QRM_TPR,
            rows_per_batch=QRM_RPB,
            num_batches=QRM_BATCHES,
            needs_guard=QRM_GUARD,
        )
        q_tr_ap = fmha.Aperture(qk_cols, lds_base=QTR_BASE, lds_stride=QTR_STRIDE, vec_width=VEC_WIDTH)
        do_rm_ap = fmha.Aperture(
            vo_cols,
            lds_base=DORM_BASE,
            lds_stride=DORM_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=DORM_TPR,
            rows_per_batch=DORM_RPB,
            num_batches=DORM_BATCHES,
            needs_guard=DORM_GUARD,
        )
        do_tr_ap = fmha.Aperture(vo_cols, lds_base=DOTR_BASE, lds_stride=DOTR_STRIDE, vec_width=VEC_WIDTH)

        q_tiling = fmha.TransposedTiling(
            d_blocks=QTR_D_BLOCKS,
            tiles=_QTR_TILES,
            loads=QTR_LOADS,
            needs_guard=QTR_GUARD,
            num_waves=NUM_WAVES,
            d_step=WMMA_N,
            kv_step=WMMA_K,
            wave_id=wave_id,
            load_d_off=((lane // 8) % 2) * WMMA_LANE_K,
            load_kv_off=(lane // 16) * WMMA_LANE_K + (lane % 8),
            store_d_off=lane16,
            store_kv_off=klane * WMMA_LANE_K,
        )
        do_tiling = fmha.TransposedTiling(
            d_blocks=DOTR_D_BLOCKS,
            tiles=_DOTR_TILES,
            loads=DOTR_LOADS,
            needs_guard=DOTR_GUARD,
            num_waves=NUM_WAVES,
            d_step=WMMA_N,
            kv_step=WMMA_K,
            wave_id=wave_id,
            load_d_off=((lane // 8) % 2) * WMMA_LANE_K,
            load_kv_off=(lane // 16) * WMMA_LANE_K + (lane % 8),
            store_d_off=lane16,
            store_kv_off=klane * WMMA_LANE_K,
        )

        q_row_in_batch = tid // QRM_TPR
        q_col_base = (tid % QRM_TPR) * VEC_WIDTH
        do_row_in_batch = tid // DORM_TPR
        do_col_base = (tid % DORM_TPR) * VEC_WIDTH

        # ---- The window, and the visited Q-block range ----
        if const_expr(CAUSAL):
            # Sentinels resolve per sequence, not on the host -- under varlen
            # bottom-right needs `seqlen_k[z] - seqlen_q[z]`.
            _wl_i32, _wr_i32 = fmha.resolve_window(window_left, window_right, seqlen_q_i32, seqlen_k_i32)
        else:
            # A window wide enough to admit everything, so the same
            # decomposition serves both arms. `smax` of the two lengths bounds
            # every reachable offset without risking the i32 overflow a literal
            # sentinel would invite in `q_start + window_right`.
            _wide = common_utils.smax(seqlen_q_i32, seqlen_k_i32)
            _wl_i32, _wr_i32 = _wide, _wide

        # The transpose of `decompose_causal_regions`: axes and window bounds
        # swapped, `block_m`/`block_n` swapped. See the module docstring.
        _regions = fmha.decompose_causal_regions(
            start_k_i32,
            seqlen_k_i32,
            seqlen_q_i32,
            _wr_i32,
            _wl_i32,
            BLOCK_N,
            BLOCK_M,
            _alive,
        )
        # The three runs are contiguous slices of one visited range, so their
        # union is `[left_col0, left_col0 + n*BLOCK_M)` whichever of them is
        # empty -- including all three, for a dead workgroup.
        _n_blocks_i32 = _regions.n_left + _regions.n_full + _regions.n_right
        _first_row_i32 = _regions.left_col0
        # Clamped only so the division below cannot divide by zero. The loop it
        # feeds has zero trips when `_n_blocks_i32` is zero, so the value is
        # never consumed.
        _nb_safe_i32 = common_utils.smax(_n_blocks_i32, fx.Int32(1))

        # The GQA group is folded into the same loop rather than nested outside
        # it: the visited Q range does not depend on the query head, so
        # `it -> (head offset, block)` is one division and no second `scf.for`.
        _total_i32 = group_i32 * _n_blocks_i32

        if const_expr(ENABLE_DROPOUT):
            _ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            _ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            _ph_stride_all = PHILOX.grid_plane(_ph_off, fx.Int32(0), max_seqlen_q, max_seqlen_k)[1]

        # ---- Loop-carried state: dK^T then dV^T accumulators ----
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        init_args = []
        for _ in range_constexpr(D_STEPS_OWN):
            init_args.append(fx.as_ir_value(c_zero_v8f32))
        for _ in range_constexpr(DV_STEPS_OWN):
            init_args.append(fx.as_ir_value(c_zero_v8f32))

        def _pack_v8(vals):
            """Eight f32 to one 16-bit WMMA operand.

            bf16 goes through the forward's bitwise truncation rather than a
            rounding convert, for the reason recorded there: RDNA4 WMMA has no
            f32 A/B form, and round-to-nearest costs 2-5% for an error the
            output sums away.
            """
            if const_expr(dtype_str == "bf16"):
                return fmha.bf16_trunc_pack_v8(vals, elem_dtype)
            return Vec.from_elements([fx.Float32(v).to(elem_dtype) for v in vals], elem_dtype).ir_value()

        loop_results = init_args
        for _it, _iter_args in range(fx.Index(0), fx.Index(_total_i32), 1, init=init_args):
            dk_accs = [_iter_args[i] for i in range_constexpr(D_STEPS_OWN)]
            dv_accs = [_iter_args[D_STEPS_OWN + i] for i in range_constexpr(DV_STEPS_OWN)]

            _it_i32 = fx.Int32(_it)
            _g_i32 = _it_i32 // _nb_safe_i32
            _bi_i32 = _it_i32 - _g_i32 * _nb_safe_i32
            head_q = fx.Index(fx.Int32(head_k) * group_i32 + _g_i32)
            _q_row_start_i32 = _first_row_i32 + _bi_i32 * fx.Int32(BLOCK_M)
            _q_row_start = fx.Index(_q_row_start_i32)

            # Q and dO addressing for *this* query head. Rebuilt each iteration
            # because `head_q` moves with the GQA fold; everything in it is a
            # uniform scalar, so this is SGPR arithmetic.
            _addr_kw_q = dict(seqlen_k=seqlen_q_v, seq_last=q_seq_last, hoist=ADDR_HOIST, clamp=True)
            _, _, q_addr = fmha.make_addr_pair(q_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw_q)
            _, _, do_addr = fmha.make_addr_pair(do_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw_q)
            if const_expr(BIAS_TYPE):
                # Rebuilt per iteration because `head_q` moves with the GQA
                # fold, exactly as the Q/dO addressing above is. Uniform
                # scalars, so this is SGPR arithmetic.
                _b_head = _q_batch_v * fx.Index(stride_b_batch) + head_q * fx.Index(stride_b_head)

            _fetch_q = fmha.reader(q_addr, lambda b, o: load_global_f16xN(q_ptr, b, o))(_q_row_start)
            _fetch_do = fmha.reader(do_addr, lambda b, o: load_global_f16xN(do_ptr, b, o))(_q_row_start)

            # VRAM -> LDS, both orientations of both tensors. Rows past
            # seqlen_q are clamped by the address closure, so what lands in LDS
            # is a duplicate of a real row; the P mask below -- not this -- is
            # what keeps it out of the answer.
            fmha.stage(q_rm_ap, lds_qdo, _fetch_q, q_row_in_batch, q_col_base, fx.Index(BLOCK_M))
            fmha.stage(do_rm_ap, lds_qdo, _fetch_do, do_row_in_batch, do_col_base, fx.Index(BLOCK_M))
            if const_expr(KEEP_TR_TILES):
                # Under "derived" there is no transposed tile to fill, and the
                # `global_load_tr_b128` fetches that fed it go with it -- the
                # same eight halves are read out of the row-major tile at GEMM
                # time instead.
                _fetch_q_tr = fmha.reader(q_addr, lambda b, o: fmha.global_load_tr_v8(q_ptr_i64, b, o, v8f16_type))(
                    _q_row_start
                )
                _fetch_do_tr = fmha.reader(do_addr, lambda b, o: fmha.global_load_tr_v8(do_ptr_i64, b, o, v8f16_type))(
                    _q_row_start
                )
                fmha.publish_transposed(
                    q_tr_ap, q_tiling, lds_qdo, fmha.read_transposed(q_tr_ap, q_tiling, _fetch_q_tr)
                )
                fmha.publish_transposed(
                    do_tr_ap, do_tiling, lds_qdo, fmha.read_transposed(do_tr_ap, do_tiling, _fetch_do_tr)
                )
            # LSE and delta for this Q tile. Q-indexed, so the same BLOCK_M
            # values serve every wave -- staging them once replaces 16 scalar
            # global loads *per lane per iteration*, and with them the 64-bit
            # address chain that dominated this loop.
            _rowq_base, _rowq_pitch = fmha.lse_row_addressing(
                varlen_bits, _q_batch_v, head_q, num_head_q, lse_tokens, _q_row_off_v
            )
            for _rb in range_constexpr(_ROWQ_BATCHES):
                _r = fx.Int32(tid) + fx.Int32(_rb * BLOCK_SIZE)
                if _r < fx.Int32(BLOCK_M):
                    _rr = _q_row_start_i32 + _r
                    _ok = _rr < seqlen_q_i32
                    _o = _rowq_base + fx.Index(fx.Index(_rr) if _ok else fx.Index(0)) * _rowq_pitch
                    fmha.lds_f32_store(_lds_byte_base, ROWQ_BYTE0, _r, load_global_f32(l_ptr, _o))
                    fmha.lds_f32_store(
                        _lds_byte_base, ROWQ_BYTE0, _r + fx.Int32(BLOCK_M), load_global_f32(delta_ptr, _o)
                    )
            gpu.barrier()

            for qsub in range_constexpr(Q_SUBTILES):
                _qs = qsub * WMMA_M

                # ==== GEMM1: S[q][kv] = Q . K^T ====
                # `m' = q`, so a lane's eight results are eight consecutive q
                # rows at one kv -- which is exactly the B-operand layout the
                # dV and dK GEMMs want, so P needs no repacking.
                if const_expr(SPLIT_HEAD_DIM):
                    # One GEMM per wave: role 0 makes S, role 1 makes dP. Same
                    # shape and accumulator, different operands -- and those
                    # were selected in the prologue, so this loop is branch-free.
                    _mine = fx.as_ir_value(c_zero_v8f32)
                    for ks in range_constexpr(D_STEPS_SHARD):
                        if const_expr(CONTRACTION_SHARDS > 1):
                            _c = fx.Index((_shard_k0 + fx.Int32(ks)) * fx.Int32(WMMA_K)) + klane * WMMA_LANE_K
                        else:
                            _c = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                        _a_op = fmha.lds_load_v8(
                            lds_qdo, q_rm_ap.lds_index(fx.Index(_qs) + lane16, _c) + _rm_base_off, _v4_ty
                        )
                        _mine = fmha.wmma_acc(_a_op, k_packs[ks], _mine)

                    for _si in range_constexpr(8):
                        fmha.lds_f32_store(
                            _lds_byte_base, PDS_BYTE0, _pds_mine + fx.Int32(_si), fx.Float32(Vec(_mine)[_si])
                        )
                    gpu.barrier()
                    if const_expr(CONTRACTION_SHARDS == 1):
                        # Relay: write mine, read the partner's. After the
                        # barrier both waves hold S and dP, so the P/dS
                        # arithmetic below is identical in both and needs no
                        # branch of its own. `_mine` stays in registers, so this
                        # is one LDS read where the reduction below is two.
                        _other = [
                            fmha.lds_f32_load(_lds_byte_base, PDS_BYTE0, _pds_theirs + fx.Int32(_si))
                            for _si in range_constexpr(8)
                        ]
                        _s_vals = None
                    else:
                        # Reduce: `slot = 2*shard + role`, so the even slots sum
                        # to S and the odd ones to dP. Own slot is read back
                        # rather than kept, which costs one LDS read and makes
                        # every wave's arithmetic below identical.
                        _s_vals = []
                        _dp_vals = []
                        for _si in range_constexpr(8):
                            _s_sum = None
                            _dp_sum = None
                            for _c_sh in range_constexpr(CONTRACTION_SHARDS):
                                _sv_c = fmha.lds_f32_load(
                                    _lds_byte_base, PDS_BYTE0, _pds_team + fx.Int32((2 * _c_sh) * PDS_SLOT_F32 + _si)
                                )
                                _dv_c = fmha.lds_f32_load(
                                    _lds_byte_base,
                                    PDS_BYTE0,
                                    _pds_team + fx.Int32((2 * _c_sh + 1) * PDS_SLOT_F32 + _si),
                                )
                                _s_sum = _sv_c if _c_sh == 0 else fastmath.add(_s_sum, _sv_c)
                                _dp_sum = _dv_c if _c_sh == 0 else fastmath.add(_dp_sum, _dv_c)
                            _s_vals.append(_s_sum)
                            _dp_vals.append(_dp_sum)
                    s_acc = _mine
                    dp_acc = _mine
                else:
                    s_acc = fx.as_ir_value(c_zero_v8f32)
                    for ks in range_constexpr(D_STEPS):
                        _qa = q_rm_ap.from_lds(
                            lds_qdo, fx.Index(_qs) + lane16, fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                        )
                        s_acc = fmha.wmma_acc(_qa, k_packs[ks], s_acc)

                    dp_acc = fx.as_ir_value(c_zero_v8f32)
                    for ks in range_constexpr(DV_STEPS):
                        _da = do_rm_ap.from_lds(
                            lds_qdo, fx.Index(_qs) + lane16, fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                        )
                        dp_acc = fmha.wmma_acc(_da, v_packs[ks], dp_acc)

                # ==== P and dS ====
                #
                # Element `si` of the accumulators is q row
                # `q_row_start + qsub*16 + klane*8 + si` at kv column
                # `start_k + wave_id*16 + lane16`. The eight q rows are
                # consecutive; the kv column is the same for all eight.
                _q_base_i32 = _q_row_start_i32 + fx.Int32(_qs) + fx.Int32(klane) * fx.Int32(WMMA_LANE_K)
                _kv_dead = kv_row_abs_i32 >= seqlen_k_i32

                _p_vals = []
                _ds_vals = []
                for si in range_constexpr(8):
                    _q_row_i32 = _q_base_i32 + fx.Int32(si)
                    _q_ok = _q_row_i32 < seqlen_q_i32
                    # Local row within the staged tile: 32-bit LDS indexing,
                    # no 64-bit address arithmetic and no global load.
                    _lrow = fx.Int32(_qs + si) + fx.Int32(klane) * fx.Int32(WMMA_LANE_K)
                    _lse = fmha.lds_f32_load(_lds_byte_base, ROWQ_BYTE0, _lrow)
                    _di = fmha.lds_f32_load(_lds_byte_base, ROWQ_BYTE0, _lrow + fx.Int32(BLOCK_M))

                    # p = exp2(sm_scale*log2e*qk - lse*log2e), AOTriton's
                    # `exp2(qk_scale*qk - l_i*RCP_LN2)` with the logsumexp in
                    # natural units. A row the forward found no live keys for
                    # carries +inf here, which makes p exactly 0 -- which is
                    # why the forward writes +inf and not -inf.
                    # Under the split, S is mine when role 0 and the partner's
                    # otherwise; dP the other way round. When the contraction is
                    # sharded the reduction already gave every wave both, so
                    # there is no role selection left to make.
                    if const_expr(SPLIT_HEAD_DIM and CONTRACTION_SHARDS > 1):
                        _sv = fx.Float32(_s_vals[si])
                    elif const_expr(SPLIT_HEAD_DIM):
                        _mv = fx.Float32(Vec(_mine)[si])
                        _ov = fx.Float32(_other[si])
                        _sv = _mv if _role0 else _ov
                    else:
                        _sv = fx.Float32(Vec(s_acc)[si])
                    _s = fastmath.mul(_sv, sm_log2e)
                    if const_expr(BIAS_TYPE):
                        # After the scale and before the mask, as in the
                        # forward, so the log2(e) factor applies here too.
                        # Both indices are clamped: a dead row or column still
                        # issues the load, and `_dead` below is what removes
                        # its contribution -- the same contract the Q/dO
                        # staging has with its clamped rows.
                        _bq = fx.Index(_q_row_i32) if _q_ok else fx.Index(0)
                        _bk = kv_row_abs if (kv_row_abs_i32 < seqlen_k_i32) else fx.Index(0)
                        _boff = _b_head + (_q_row_off_v + _bq) * fx.Index(stride_b_seq_q) + _bk
                        _bv = fx.Float32(fx.as_dsl_value(load_global_f16(bias_ptr, _boff)).to(fx.Float32))
                        _s = fastmath.add(_s, fastmath.mul(_bv, fx.Float32(_LOG2E)))
                    _e = fastmath.sub(_s, fastmath.mul(fx.Float32(_lse), fx.Float32(_LOG2E)))
                    _p = rocdl.exp2(f32_ty, fx.as_ir_value(_e))

                    # One select for every reason this element is not a real
                    # (query, key) pair. Q rows past seqlen_q and kv columns
                    # past seqlen_k both read clamped duplicates of real rows,
                    # so their scores are finite garbage rather than something
                    # the arithmetic would reject.
                    _dead = _kv_dead | (_q_ok == fx.Boolean(False))
                    if const_expr(CAUSAL):
                        # Both edges of the band, signed throughout: for plain
                        # causal `window_left` resolves to seqlen_q, which
                        # makes the left term inert rather than absent.
                        _dead = _dead | (kv_row_abs_i32 > _q_row_i32 + _wr_i32)
                        _dead = _dead | (kv_row_abs_i32 < _q_row_i32 - _wl_i32)
                    _p = fx.as_ir_value(fx.Float32(0.0) if _dead else fx.Float32(_p))

                    if const_expr(SPLIT_HEAD_DIM and CONTRACTION_SHARDS > 1):
                        _dpv = fx.Float32(_dp_vals[si])
                    elif const_expr(SPLIT_HEAD_DIM):
                        _dpv = _ov if _role0 else _mv
                    else:
                        _dpv = fx.Float32(Vec(dp_acc)[si])
                    if const_expr(ENABLE_DROPOUT):
                        # The stream is packed `randoms_per_offset` to an
                        # offset along the *kv* axis, but a lane here holds one
                        # kv column and eight q rows, so each element is its
                        # own Philox call and the slot within it is
                        # `kv % randoms_per_offset` -- a runtime index, hence
                        # the select chain. This is the layout cost of running
                        # the loop transposed and it is the reason dropout is
                        # off by default; see the module docstring.
                        _ph_base = _ph_off + fx.Int64(
                            fx.Int32(z_i32) * fx.Int32(num_head_q) + fx.Int32(head_q)
                        ) * fx.Int64(max_seqlen_q) * fx.Int64(_ph_stride_all)
                        _rn = PHILOX.randoms_per_offset
                        _poff = PHILOX.grid_offset(_ph_base, _ph_stride_all, _q_row_i32, kv_row_abs_i32)
                        _keeps = PHILOX.keep_span(_ph_seed, _poff, _rn, idropout_p)
                        _slot = fx.Int32(kv_row_abs_i32) % fx.Int32(_rn)
                        _keep = _keeps[0]
                        for _r in range_constexpr(_rn - 1):
                            _keep = _keeps[_r + 1] if _slot == fx.Int32(_r + 1) else _keep
                        # dV takes the dropped-and-rescaled P; dS takes the
                        # original. Reversing that is AOTriton's explicit
                        # "CAVEAT: do NOT update p".
                        _p_drop = fx.as_ir_value(
                            fastmath.mul(fx.Float32(_p), dropout_scale) if _keep else fx.Float32(0.0)
                        )
                        _dpv = fx.Float32(fastmath.mul(_dpv, dropout_scale) if _keep else fx.Float32(0.0))
                    else:
                        _p_drop = _p

                    _p_vals.append(_p_drop)
                    _ds_vals.append(fx.as_ir_value(fastmath.mul(fx.Float32(_p), fastmath.sub(_dpv, fx.Float32(_di)))))

                _p_pack = _pack_v8(_p_vals)
                _ds_pack = _pack_v8(_ds_vals)

                # One transposed A operand, from whichever source the knob
                # selects. "tile" reads the eight halves contiguously out of the
                # transposed tile; "derived" reads the same eight out of the
                # row-major tile, strided by its pitch -- which is why that tile
                # need not exist. gfx1201 has no `ds_load_tr16_b128`, so the
                # second form is eight reads rather than one.
                def _tr_operand(rm_ap, tr_ap, d_pos, q0):
                    if const_expr(KEEP_TR_TILES):
                        return tr_ap.from_lds(lds_qdo, d_pos, q0)
                    return Vec.from_elements(
                        [
                            fx.ptr_load(lds_qdo + fx.Int32(rm_ap.lds_index(q0 + fx.Index(j), d_pos)))
                            for j in range_constexpr(8)
                        ],
                        elem_dtype,
                    ).ir_value()

                _q0 = fx.Index(_qs) + klane * WMMA_LANE_K

                # ==== GEMM3: dV^T[d][kv] += dO^T . P ====
                for dc in range_constexpr(DV_STEPS_OWN):
                    _gd = fx.Index((fx.Int32(dc) + _vown_i32) * fx.Int32(WMMA_N))
                    _a = _tr_operand(do_rm_ap, do_tr_ap, _gd + lane16, _q0)
                    dv_accs[dc] = fmha.wmma_acc(_a, _p_pack, dv_accs[dc])

                # ==== GEMM4: dK^T[d][kv] += Q^T . dS ====
                for dc in range_constexpr(D_STEPS_OWN):
                    _gd = fx.Index((fx.Int32(dc) + _own_i32) * fx.Int32(WMMA_N))
                    _a = _tr_operand(q_rm_ap, q_tr_ap, _gd + lane16, _q0)
                    dk_accs[dc] = fmha.wmma_acc(_a, _ds_pack, dk_accs[dc])

            # Every wave must finish reading this Q/dO window before the next
            # iteration overwrites it.
            gpu.barrier()
            loop_results = yield dk_accs + dv_accs

        # ---- Epilogue ----
        #
        # dK^T and dV^T come out of the GEMMs with `m' = d`, so element `si` of
        # a lane is d column `dc*16 + klane*8 + si` -- eight *contiguous* d at
        # one kv row. That is one v8 store per accumulator, the same shape as
        # the forward's O store, and it is the reason the two GEMMs above
        # compute the transposes rather than dK and dV directly.
        dk_base = dk_tbase(_k_start_addr)
        dv_base = dv_tbase(_k_start_addr)

        def write_dk(row, col, val):
            store_global_v8f16(dk_ptr, dk_base, dk_toff(row, col), val)

        def write_dv(row, col, val):
            store_global_v8f16(dv_ptr, dv_base, dv_toff(row, col), val)

        dk_out_ap = fmha.Aperture(qk_cols)
        dv_out_ap = fmha.Aperture(vo_cols)

        _row_live = kv_row_abs_i32 < seqlen_k_i32
        if _row_live:
            # dK carries the sm_scale AOTriton applies once at the end: the
            # accumulator is dS^T Q with dS taken against the *unscaled* score,
            # so the chain rule's factor lands here rather than per element.
            _scale_vec = Vec.from_elements([fx.Float32(sm_scale)], fx.Float32).broadcast_to(8).ir_value()
            # Under the split each wave stores only the head-dim half it
            # accumulated; the pair writes disjoint columns, hence no reduction.
            for dc in range_constexpr(D_STEPS_OWN):
                _v = fastmath.mul(loop_results[dc], _scale_vec)
                _t = Vec(_v).to(elem_dtype).ir_value()
                _gc = fx.Index((fx.Int32(dc) + _own_i32) * fx.Int32(WMMA_N)) + klane * WMMA_LANE_K
                fmha.write_v8(dk_out_ap, write_dk, kv_row_in_tile, _gc, _t)
            for dc in range_constexpr(DV_STEPS_OWN):
                _t = Vec(loop_results[D_STEPS_OWN + dc]).to(elem_dtype).ir_value()
                _gc = fx.Index((fx.Int32(dc) + _vown_i32) * fx.Int32(WMMA_N)) + klane * WMMA_LANE_K
                fmha.write_v8(dv_out_ap, write_dv, kv_row_in_tile, _gc, _t)

    @flyc.jit
    def launch_bwd_dkdv(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        batch_size: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        nseq_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        # The grid's KV extent keys on Max_seqlen_k: under varlen there is no
        # single seqlen_k, so every sequence gets the longest one's worth of
        # workgroups and the short ones exit having walked zero Q blocks.
        num_kv_tiles = (fx.Index(max_seqlen_k) + (BLOCK_N - 1)) // BLOCK_N

        launcher = bwd_dkdv_kernel(
            Q,
            K,
            V,
            B,
            DO,
            DK,
            DV,
            LSE,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
        )

        if const_expr(WAVES_PER_EU is not None):
            _wpe = int(WAVES_PER_EU)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.WAVES_PER_EU"] = ir.IntegerAttr.get(T.i32, _wpe)
        _fwgs = int(BLOCK_SIZE)
        flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["rocdl.FLAT_WORK_GROUP_SIZE"] = flat_wg_attr

        passthrough_entries = []
        if const_expr(SCHED_STRATEGY):
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("amdgpu-sched-strategy"), ir.StringAttr.get(SCHED_STRATEGY)])
            )
        if const_expr(DENORMALS_ARE_ZERO):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(num_kv_tiles, fx.Index(num_head_k), nseq_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_bwd_dkdv.compile_hints = {
        "FAST_FP_MATH": FAST_FP_MATH,
        "UNSAFE_FP_MATH": UNSAFE_FP_MATH,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        L,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        num_seqlens=0,
        scale=None,
        stream=None,
        window=None,
        varlen=None,
        dropout_p=None,
        philox_seed=0,
        philox_offset1=None,
        philox_offset2=0,
        bias=None,
    ):
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta_t, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV)],
            q_heads=("DO",),
            k_heads=("DK", "DV"),
        )
        _lp = abi.row_tensor_arg(L, "logsumexp", meta_t[0], seqlen_q, varlen)
        _dp = abi.row_tensor_arg(Delta, "delta", meta_t[0], seqlen_q, varlen)
        _wl, _wr = abi.resolve_window(CAUSAL_TYPE, HOST_CAUSAL_TYPE, window, seqlen_q, seqlen_k)
        _vb, _sq0, _sq1, _sk0, _sk1, _mq, _mk = abi.varlen_args(
            False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens
        )
        _ps, _po1, _po2, _ip, _dsc, _hold = abi.dropout_args(
            ENABLE_DROPOUT, dropout_p, philox_seed, philox_offset1, philox_offset2, Q.device, stream
        )
        # `with_db=False`: dB is the dQ kernel's, so the DB slot is discarded
        # rather than being a kernarg this kernel does not have.
        _bp, _, _bs, _ = abi.bias_args(BIAS_TYPE, False, bias, None, Q)
        abi.run_compiled(
            _COMPILED,
            launch_bwd_dkdv,
            # The shared backward pointer block is
            #     Q, K, V, B, DO, <outputs>, LSE, Delta
            # with `<outputs>` = (DK, DV) here; the dQ kernel's is (DQ, DB),
            # dB being one of its outputs rather than an extra slot.
            *ptrs[:3],
            _bp,
            *ptrs[3:],
            _lp,
            _dp,
            _sq0,
            _sq1,
            _sk0,
            _sk1,
            _vb,
            batch_size,
            num_seqlens,
            _mq,
            _mk,
            _wl,
            _wr,
            _ps,
            _po1,
            _po2,
            _ip,
            _dsc,
            *meta_t,
            abi.resolve_scale(Q, scale, PADDED_HEAD, sm_scale),
            *st,
            *_bs,
            stream if stream is not None else fx.Stream(None),
        )

    _launch.varlen_bits = abi.varlen_bits
    _launch.varlen_compact = abi.varlen_compact
    _launch.varlen_padded = abi.varlen_padded
    _launch.varlen_strided = abi.varlen_strided
    return _launch


def build_bwd_dkdv_module(**kwargs):
    """Keyword front end: name a problem, get the policy's schedule."""
    meta_fields = {f.name for f in fields(BwdDkDvMetadata)}
    knob_fields = {f.name for f in fields(BwdDkDvKnobs)}
    unknown = set(kwargs) - meta_fields - knob_fields
    if unknown:
        raise TypeError(f"unknown build parameter(s): {sorted(unknown)}")
    meta = BwdDkDvMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    overrides = BwdDkDvKnobs(**{k: v for k, v in kwargs.items() if k in knob_fields})
    return build_bwd_dkdv_module_primary(meta, resolve_knobs(meta, overrides))
