# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Backward pass, dQ only, for gfx1201 (RDNA4).

A port of AOTriton's ``bwd_kernel_dq`` / ``bwd_inner_dq``, written against the
same helpers as the forward kernel in ``flash_attn_func_gfx1201_aiw.py`` and
sharing its ABI. It computes

    dQ[i, :] = sm_scale * sum_j  dS[i, j] * K[j, :]
    dS[i, j] = P[i, j] * (dP[i, j] - delta[i])
    dP[i, j] = dO[i, :] . V[j, :]
    P[i, j]  = exp2(qk_scale * S[i, j] - lse2[i])

with ``S = Q K^T``, ``qk_scale = sm_scale * log2(e)`` and
``lse2 = LSE * log2(e)``. ``LSE`` is the forward kernel's own output, read
through the same ``lse_token_pitch`` layout it was written with.

``delta = rowsum(dO * O)`` arrives as a **tensor**, computed by the caller.
AOTriton fuses that into a ``bwd_preprocess`` kernel; here it is
``(dO.float() * O.float()).sum(-1)`` on the host side. Fusing it is a later
optimization and changes nothing about this file except one argument.

--- Why this is shaped like the forward kernel ----------------------------

Both preload a Q tile and walk KV tiles, so the whole structure transfers:
``Aperture``, ``stage``, ``reader``, ``decompose_causal_regions`` and
``decode_addressing`` are used exactly as the forward uses them. The
difference is that dQ needs *three* GEMMs per KV tile instead of two, and the
third one reads K along the other axis.

The WMMA operand algebra, which is what fixes every LDS layout below. In
wave32 the 16x16x16 instruction ``D = A @ B + C`` places

    A operand   lane holds A[lane16][klane*8 + j]      j = 0..7
    B operand   lane holds B[klane*8 + j][lane16]
    C/D         lane holds D[klane*8 + si][lane16]     si = 0..7

so an A and a B operand are *the same eight registers* read as a matrix and as
its transpose, and a C/D accumulator can be handed straight back as either.
That identity is what makes the three GEMMs line up:

    GEMM1   S^T[kv][q] = K[kv][:] . Q[q][:]
            A = K   read from a row-major K tile   (lane16 = kv, klane*8+j = d)
            B = Q   the register preload           (lane16 = q,  klane*8+j = d)

    GEMM2   dP^T[kv][q] = V[kv][:] . dO[q][:]
            A = V   read from a row-major V tile   -- the *identical* access
            B = dO  the register preload           -- pattern as GEMM1

    GEMM3   dQ[d][q] += K^T[d][kv] * dS^T[kv][q]
            A = K^T  (lane16 = d, klane*8+j = kv)  -- K along the other axis
            B = dS   the accumulator from GEMM1/2, handed back unchanged

GEMM1 and GEMM2 therefore cost nothing new: they are the forward's GEMM1 twice
over. GEMM3 is the forward's GEMM2 with V^T replaced by K^T and P by dS, and
its accumulator lands in exactly the forward's O layout -- eight contiguous d
per lane at one q row -- so the epilogue is one v8 store per 16 columns.

**K is read both ways, and that is inherent.** AOTriton says the same thing in
one line ("we cannot avoid transpose here as this loop uses k both normal and
transpose"). Two ways to serve GEMM3, selected by ``kt_lds_layout``:

    "scalar"      read K^T out of the row-major tile with 8 strided 16-bit LDS
                  loads per operand. No extra LDS, no extra global traffic.
    "transposed"  stage a second copy K^T[d][kv] with ``global_load_tr_b128``,
                  exactly as the forward stages V^T, and read one vector per
                  operand. Costs a second K global load and
                  ``head_dim * (BLOCK_N + 4)`` more elements of LDS.

The two are **bitwise identical** -- same values, same order -- which is what
the test asserts, and is the reason it is safe to switch on the evidence of a
benchmark alone.

--- Head-dimension sharding -----------------------------------------------

``QK_SHARDS`` waves cooperate on one Q row sub-tile, each holding only
``head_dim / QK_SHARDS`` columns of Q, dO and the dQ accumulator. It is the
forward's mechanism, and it fits dQ more neatly than it fits the forward:
**GEMM1's reduction axis and GEMM3's output axis are the same axis**, so one
column offset drives Q, dO, K, V, K^T and the output store, and the shards'
output stores are disjoint -- there is nothing to combine at the end. What has
to be combined is in the middle: the partial S from GEMM1 and the partial dP
from GEMM2, both summed across the shards through LDS before the softmax.

It is the only way head_dim 384 and 512 exist at all. Unsharded, a wave holds
``head_dim`` VGPRs of operands and accumulators before addressing and staging;
512 would need 544 against a 256-VGPR file. ``QK_SHARDS == 1`` is the
unsharded kernel *exactly*, down to the emitted ISA.

--- What this kernel does not do ------------------------------------------

No V/O column window, no row sub-tiles, no bias, and the distance-1 K/V
prefetch is off by default. Those are the forward's speed knobs and each is a
separate correctness surface; this pass is scored on matching torch autograd.
See ``fmha_tuning_bwd_dq_gfx1201.py`` for what is worth measuring first.

Shape:  Q/K/V/dO/dQ are BHSD (batch, num_heads, seq_len, head_dim); the memory
        layout is free so long as D is innermost. Strides arrive as
        (batch, head, seq) -- axis 3 is never passed.
Grid:   (num_head_q, num_q_tiles, batch)
"""

import math as host_math
from dataclasses import fields

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
from fmha_tuning_bwd_dq_gfx1201 import (  # noqa: F401
    BwdDqInputMetadata,
    BwdDqKnobs,
    resolve_knobs,
)
from gfx1201_standalone import buffer_ops
from gfx1201_standalone import kernels_common as common_kernels
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

KERNEL_NAME = "fmha_bwd_dq_gfx1201_kernel"
_LOG2E = host_math.log2(host_math.e)

# Causal alignment as a *window* sentinel; the kernel resolves it per sequence.
# Same two values the forward emits, taken from the same place rather than
# re-spelled, so the two kernels cannot drift.
_WINDOW_TOPLEFT = fmha.WINDOW_TOPLEFT
_WINDOW_BOTRIGHT = fmha.WINDOW_BOTRIGHT


# `_run_compiled`, `_llvm_value`, `_pointer_load` and `_pointer_store` are
# duplicated from `flash_attn_func_gfx1201_aiw.py`, which duplicated the first
# from `kernels/common/tensor_shim.py`. That file is read-only for this work
# and importing it would make the backward kernel depend on the forward *kernel
# module* rather than only on the shared helpers -- a dependency the eventual
# consolidation would have to unpick. Four small functions is the cheaper debt.


_COMPILED = abi.new_compiled_cache()


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, fmha.llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(fmha.llvm_value(value), fmha.llvm_value(ptr))


def build_bwd_dq_module_primary(meta, knobs):
    """Build the gfx1201 backward-dQ kernel.

    Takes the two objects rather than a long parameter list, split on *who
    decides*: `meta` is what the caller asked for, `knobs` is what the tuning
    policy answered. Build `knobs` with
    `fmha_tuning_bwd_dq_gfx1201.resolve_knobs(meta)` -- every field must be
    resolved, since nothing here falls back to a policy.
    """
    num_heads = meta.num_heads
    causal = meta.causal
    dtype_str = meta.dtype_str
    sm_scale = meta.sm_scale
    causal_type = meta.causal_type
    dropout = meta.dropout
    philox_width = meta.philox_width

    BLOCK_DMODEL = knobs.block_dmodel
    BLOCK_M = knobs.block_m
    BLOCK_N = knobs.block_n
    NUM_WAVES = knobs.num_waves
    QK_SHARDS = knobs.shards
    KT_LDS_LAYOUT = knobs.kt_lds_layout
    KV_ADDR_HOIST = knobs.kv_addr_hoist
    KV_PREFETCH_DIST = knobs.kv_prefetch_dist
    WAVES_PER_EU = knobs.waves_per_eu
    FLAT_WORK_GROUP_SIZE = knobs.flat_work_group_size
    SCHED_STRATEGY_KNOB = knobs.sched_strategy
    PADDED_HEAD = knobs.padded_head
    LPT_TILE_ORDER_KNOB = knobs.lpt_tile_order
    FP_MODE = knobs.fp_mode
    DENORMALS_ARE_ZERO = knobs.denormals_are_zero
    UNSAFE_FP_MATH = knobs.unsafe_fp_math
    FAST_FP_MATH = knobs.fast_fp_math

    # ---- WMMA / wave32 constants ----
    # RDNA4 is wave32 and its WMMA is 16x16x16. The last two are derived, not
    # free parameters; see the module docstring for the operand algebra.
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    # K elements each lane holds of a WMMA A/B operand: two lanes share a row
    # and split the K extent, so WMMA_K / (WARP_SIZE / WMMA_M) == 8.
    WMMA_LANE_K = WMMA_K // (WARP_SIZE // WMMA_M)

    ROWS_PER_WAVE = WMMA_M

    # ---- Head-dimension sharding ----
    #
    # `QK_SHARDS` waves cooperate on one Q row sub-tile. Wave *s* holds only
    # columns `[s*QK_SLICE, (s+1)*QK_SLICE)` of Q and dO, reduces GEMM1 and
    # GEMM2 over that slice alone, and owns exactly those columns of the dQ
    # output in GEMM3 -- so **one offset serves all three GEMMs**, because
    # GEMM1's reduction axis and GEMM3's output axis are both the QK head dim.
    # The partial S and dP are summed across the shards through LDS between
    # GEMM2 and the softmax.
    #
    # This is the *only* way head_dim 384/512 is expressible. A wave carries
    # `BLOCK_DMODEL/4` VGPRs of Q packs, the same of dO, and `BLOCK_DMODEL/2`
    # of dQ accumulators -- `BLOCK_DMODEL` registers per lane before operands,
    # addressing and staging. Measured from the ISA: head_dim 128 is 221 VGPRs
    # with no spill, head_dim 256 is pinned at 256 with 132 bytes of scratch.
    # 512 would need 544. Sharding divides all three sets by `QK_SHARDS` and
    # nothing else does: `BLOCK_M` only sets how many waves are in the
    # workgroup, and `BLOCK_N` sizes only the S/dP accumulators.
    #
    # `QK_SHARDS == 1` is the unsharded kernel exactly; every sharded construct
    # below is behind `const_expr(QK_SHARDS > 1)`.
    QK_SLICE = BLOCK_DMODEL // QK_SHARDS
    Q_TILES_PER_BLOCK = NUM_WAVES // QK_SHARDS

    # ---- Validity predicate over the knob space ----
    #
    # Assertions, not ValueError: every name here was resolved by
    # `resolve_knobs`, so a violation is that module contradicting itself
    # rather than a caller mistake. Caller input is checked in `plan` and in
    # the launcher.
    assert (
        BLOCK_DMODEL % 16 == 0 and 16 <= BLOCK_DMODEL <= 512
    ), f"bwd_dq needs 16 <= BLOCK_DMODEL <= 512 and BLOCK_DMODEL % 16 == 0, got {BLOCK_DMODEL}"
    assert dtype_str in ("f16", "bf16"), f"bwd_dq supports f16/bf16, got {dtype_str!r}"
    assert BLOCK_N in (16, 32), f"bwd_dq supports BLOCK_N 16 or 32 (see the tuning module), got {BLOCK_N}"
    # A slice that is not a multiple of WMMA_K would silently drop part of the
    # reduction rather than fail -- the forward measured 0.97 relative error
    # from exactly this at head_dim 224 with 4 shards. Rejected at build time.
    assert NUM_WAVES % QK_SHARDS == 0, f"num_waves ({NUM_WAVES}) must be a multiple of shards ({QK_SHARDS})"
    assert BLOCK_DMODEL % QK_SHARDS == 0 and QK_SLICE % WMMA_K == 0, (
        f"BLOCK_DMODEL {BLOCK_DMODEL} with {QK_SHARDS} shards gives a {QK_SLICE}-wide "
        f"slice, which must be a multiple of WMMA_K={WMMA_K}"
    )
    assert (
        BLOCK_M == ROWS_PER_WAVE * Q_TILES_PER_BLOCK
    ), f"BLOCK_M ({BLOCK_M}) must be {ROWS_PER_WAVE} * {NUM_WAVES} / {QK_SHARDS}"
    assert KT_LDS_LAYOUT in (
        "scalar",
        "transposed",
    ), f"kt_lds_layout must be 'scalar' or 'transposed', got {KT_LDS_LAYOUT!r}"

    BLOCK_SIZE = FLAT_WORK_GROUP_SIZE if FLAT_WORK_GROUP_SIZE else NUM_WAVES * WARP_SIZE
    assert BLOCK_SIZE == NUM_WAVES * WARP_SIZE, f"flat_work_group_size {BLOCK_SIZE} contradicts {NUM_WAVES} waves"

    # One v8f32 S accumulator per 16 KV columns; accumulator `a` holds columns
    # `[a*16, (a+1)*16)`. This used to be spelled as pairs of accumulators
    # inside 32-column "sub-tiles", which was the same indexing written twice
    # over and made BLOCK_N 16 inexpressible -- and BLOCK_N 16 is what fits
    # head_dim 512's K and V tiles inside 64 KiB.
    NUM_S_ACCS = BLOCK_N // WMMA_N
    NUM_S_VALS = NUM_S_ACCS * 8

    # GEMM1 reduces over the QK head dim, GEMM2 over the V/O one. They share a
    # tile width here -- there is no `head_dim_v` knob -- but the *runtime*
    # extents `hdim_qk` and `hdim_vo` may still differ, which is what the two
    # separate column axes below are for. Under sharding each wave covers only
    # its own slice of that width, in all three GEMMs.
    K_STEPS_QK = QK_SLICE // WMMA_K
    K_STEPS_VO = QK_SLICE // WMMA_K
    # dQ accumulators: one v8f32 per 16 output columns of this wave's slice.
    D_CHUNKS = QK_SLICE // WMMA_N

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(BLOCK_DMODEL)

    NUM_HEADS = num_heads  # noqa: F841  (kept for symmetry with the forward)
    CAUSAL = causal

    # Longest-processing-time-first dispatch of the Q tiles. Under causal
    # masking a dQ block's cost grows with its index exactly as the forward's
    # does -- tile 0 walks one KV block, tile N-1 walks N -- so issuing the
    # expensive ones first leaves only cheap tiles to fill the tail. With
    # uniform cost the reversal is a permutation with no load-balancing
    # content, so `and CAUSAL` is part of the knob's definition.
    _LPT_TILE_ORDER = CAUSAL and LPT_TILE_ORDER_KNOB

    # See the forward kernel's note. "noninf" is `fast` minus `ninf`; `ninf`
    # licenses the compiler to fold away the -inf this kernel writes into
    # masked scores *and* the +inf logsumexp the forward emits for a row with
    # no live keys. Both are load-bearing here.
    fastmath = fmha.FastMath(FP_MODE)

    # LLVM's amdgpu-sched-strategy function attribute; "" leaves the default
    # GCN scheduler in place, which is what every width but 512 takes. See
    # `_SCHED_STRATEGY_BY_HEAD_DIM` in the tuning module for the sweep.
    SCHED_STRATEGY = "" if SCHED_STRATEGY_KNOB is None else SCHED_STRATEGY_KNOB

    # Causal masking. `causal_type` is AOTriton's: 0 none, 1 top-left, 2
    # bottom-right, 3 explicit window. As in the forward, **the kernel only
    # ever sees 0 or 3** -- 1 and 2 are host-side conveniences that resolve to
    # a sentinel window before dispatch, and the sentinel is resolved against
    # *this sequence's* lengths inside the kernel.
    if causal_type is None:
        HOST_CAUSAL_TYPE = 1 if causal else 0
    else:
        HOST_CAUSAL_TYPE = causal_type
    if HOST_CAUSAL_TYPE not in (0, 1, 2, 3):
        raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {HOST_CAUSAL_TYPE}")
    if bool(HOST_CAUSAL_TYPE) != bool(causal):
        raise ValueError(f"causal={causal} disagrees with causal_type={HOST_CAUSAL_TYPE}")
    CAUSAL_TYPE = 0 if HOST_CAUSAL_TYPE == 0 else 3  # noqa: F841

    # Dropout, AOTriton's ENABLE_DROPOUT. A build axis, so a build without it
    # emits no PRNG at all. The stream is `philox.py`'s, shared with the
    # forward kernel and the debug mask kernel -- all three must agree bit for
    # bit or the gradient is silently wrong, which is why the offset scheme
    # below is `Philox.grid_plane` / `grid_offset` and not a transcription.
    ENABLE_DROPOUT = bool(dropout)
    PHILOX = Philox.for_arch() if philox_width is None else Philox(width=philox_width)

    # Attention bias. One build axis, not two: dB is `dS`, which this kernel
    # already forms for GEMM3, so a bias build emits it unconditionally. A
    # second knob would buy a store nobody has asked to skip.
    BIAS_TYPE = 1 if meta.bias else 0
    assert not (BIAS_TYPE and CAUSAL), "bias and causal are mutually exclusive, as in the forward"

    # ---- LDS layout ----
    # K and V are padded rather than XOR-swizzled, following the forward (a
    # swizzle was implemented there and measured a net loss).
    _LDS_PAD = 4
    K_STRIDE = BLOCK_DMODEL + _LDS_PAD  # K[kv][d] and V[kv][d]
    V_STRIDE = BLOCK_DMODEL + _LDS_PAD
    # Transposed K^T[d][kv]. +4 makes the row stride 36 elements = 18 dwords,
    # so lane16*18 mod 32 hits 16 distinct banks while staying 8-byte aligned;
    # this is the forward's V^T geometry with the roles of the tensors swapped.
    KT_STRIDE = BLOCK_N + _LDS_PAD

    # Cooperative-load vector width, in elements. 8 == 16 bytes, which is
    # exactly what the D-axis alignment contract guarantees; see the forward's
    # note for why 16 is unsound rather than merely unmeasured.
    VEC_WIDTH = 8

    def _load_geom(width):
        return fmha.load_geom(width, VEC_WIDTH, BLOCK_SIZE, BLOCK_N)

    # ceil() batches, not floor(): flooring silently drops rows whenever
    # rows-per-batch neither reaches BLOCK_N nor divides it, which surfaces as
    # stale LDS rather than as an error.
    K_TPR_LOAD, K_ROWS_PER_BATCH, NUM_BATCHES_K, K_NEEDS_GUARD = _load_geom(BLOCK_DMODEL)
    V_TPR_LOAD, V_ROWS_PER_BATCH, NUM_BATCHES_V, V_NEEDS_GUARD = _load_geom(BLOCK_DMODEL)

    KT_TRANSPOSED = KT_LDS_LAYOUT == "transposed"
    # global_load_tr_b128 transposes an 8x8 tile of 16-bit elements across each
    # group of 8 lanes, so one wave-wide load produces a 16(d) x 16(kv) block
    # already in WMMA-operand order. Split those blocks over the waves; the
    # tiling need not divide evenly, tail tiles are guarded at the LDS store.
    KT_D_BLOCKS = BLOCK_DMODEL // WMMA_N
    _KT_TILES = KT_D_BLOCKS * (BLOCK_N // WMMA_K)
    KT_LOADS = (_KT_TILES + NUM_WAVES - 1) // NUM_WAVES
    KT_NEEDS_GUARD = KT_LOADS * NUM_WAVES != _KT_TILES

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_BASE = LDS_K_TILE_SIZE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_KT_BASE = LDS_V_BASE + LDS_V_TILE_SIZE
    LDS_KT_TILE_SIZE = BLOCK_DMODEL * KT_STRIDE if KT_TRANSPOSED else 0

    # The cross-shard S/dP reduction buffer: one private slot per wave, holding
    # both partials (`2 * NUM_S_VALS` f32 per lane).
    #
    # It **aliases the V tile** when it fits there, exactly as the forward's
    # aliases its own V region, and for a reason that is specific to this
    # kernel's GEMM order: V is read by GEMM2 and never again, while K is read
    # again by GEMM3, so V's storage is dead from the moment the reduction
    # starts. That costs one extra barrier -- every wave must be past GEMM2
    # before the first partial lands on top of V -- and it is what keeps
    # head_dim 512 inside 64 KiB.
    RED_F32_PER_WAVE = 2 * NUM_S_VALS * WARP_SIZE
    RED_F32_TOTAL = NUM_WAVES * RED_F32_PER_WAVE
    RED_ALIASES_V = QK_SHARDS == 1 or RED_F32_TOTAL * 4 <= LDS_V_TILE_SIZE * 2
    LDS_RED_SIZE = 0 if RED_ALIASES_V else (RED_F32_TOTAL * 4 + 1) // 2
    LDS_TOTAL_SIZE = LDS_KT_BASE + LDS_KT_TILE_SIZE + LDS_RED_SIZE
    RED_BYTE0 = (LDS_V_BASE if RED_ALIASES_V else LDS_KT_BASE + LDS_KT_TILE_SIZE) * 2
    assert LDS_TOTAL_SIZE * 2 <= 65536, (
        f"LDS tile is {LDS_TOTAL_SIZE * 2} B, over the 64 KiB workgroup cap "
        f"(BLOCK_DMODEL {BLOCK_DMODEL}, BLOCK_N {BLOCK_N}, num_waves {NUM_WAVES}, "
        f"shards {QK_SHARDS}, kt_lds_layout {KT_LDS_LAYOUT!r})"
    )

    elem_numeric_cls = common_kernels.dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def bwd_dq_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls

        q_ptr = fmha.pointer_to_llvm_ptr(Q)
        k_ptr = fmha.pointer_to_llvm_ptr(K)
        v_ptr = fmha.pointer_to_llvm_ptr(V)
        do_ptr = fmha.pointer_to_llvm_ptr(DO)
        dq_ptr = fmha.pointer_to_llvm_ptr(DQ)
        k_ptr_i64 = fx.as_ir_value(fx.Int64(fx.ptrtoint(K)))
        v8f16_type = Vec.make_type(8, elem_dtype)
        vxf16_type = Vec.make_type(VEC_WIDTH, elem_dtype)
        f32_ty = ir.F32Type.get()

        # ---- Varlen prologue: VarlenBits -> six scalars ----
        #
        # The only place the layout is examined; everything downstream reads
        # the scalars and cannot tell which mode it is in. `z` is uniform, so
        # every load here is scalar and none touches the VGPR budget. This is
        # **our** ABI, not AOTriton's cu_seqlens/num_seqlens: the backward pass
        # has to interoperate with our forward kernel.
        z_i32 = fx.Int32(gpu.block_idx.z)

        seqlen_q_i32, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32
        )
        seqlen_k_i32, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32
        )
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        seqlen_k_v = fx.Index(seqlen_k_i32)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv = lds.kv.ptr

        # f32 view of the reduction region. The kv array is elem_dtype
        # (16-bit), so go through an addrspace(3) LLVM pointer: ptrtoint on a
        # shared pointer yields the 32-bit LDS offset.
        _lds_byte_base = fx.as_ir_value(fx.ptrtoint(lds_kv))

        tid, wave_id, lane, lane16, klane = fmha.wave_lanes(WARP_SIZE)

        # (q_tile, shard) decomposition of the wave index. At QK_SHARDS == 1
        # this is q_tile == wave_id and shard == 0, i.e. the unsharded mapping.
        q_tile_in_block = wave_id // QK_SHARDS
        shard_id = wave_id % QK_SHARDS
        wave_q_offset = q_tile_in_block * ROWS_PER_WAVE
        # Column origin of this wave's head-dim slice, 0 at QK_SHARDS == 1. It
        # indexes Q, dO, K and V for the two reductions *and* dQ and K^T for
        # the output, because both axes are the QK head dim.
        shard_qk_off = shard_id * fx.Index(QK_SLICE)

        # 3D grid: (head_q, q_tile, batch). Same axis order as the forward, and
        # load-bearing for the same reason: the x axis dispatches fastest, so
        # putting q_tile on y and head on x gives each scheduling group a
        # uniform duration under causal masking.
        head_q = fx.Index(gpu.block_idx.x)
        if const_expr(_LPT_TILE_ORDER):
            # Max_seqlen_q, not this sequence's length: the reversal has to be
            # a permutation of the *grid*, whose y extent the host sized from
            # Max_seqlen_q.
            _ntiles = (fx.Index(max_seqlen_q) + (BLOCK_M - 1)) // BLOCK_M
            q_tile_idx = _ntiles - fx.Index(1) - fx.Index(gpu.block_idx.y)
        else:
            q_tile_idx = fx.Index(gpu.block_idx.y)
        start_q = q_tile_idx * BLOCK_M

        # Does this workgroup own any real Q row? Under varlen the grid's Q
        # extent is sized from Max_seqlen_q, so whole workgroups land past the
        # end of a shorter sequence. Compared in i32 because `fx.Index` is
        # unsigned and both operands are bounded by an i32 ABI argument.
        _alive = fx.Int32(start_q) < seqlen_q_i32
        # The Q base must be clamped, not just the row within the tile: a dead
        # workgroup would otherwise address `row_off + start_q` rows into a
        # packed tensor, which runs past the whole allocation rather than only
        # past this sequence.
        _q_start_addr = fx.Index(start_q if _alive else fx.Index(0))

        # MQA/GQA: num_head_q / num_head_k query heads share each KV head.
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        load_row_in_batch = tid // K_TPR_LOAD
        load_col_base = (tid % K_TPR_LOAD) * VEC_WIDTH
        v_row_in_batch = tid // V_TPR_LOAD
        v_col_base = (tid % V_TPR_LOAD) * VEC_WIDTH

        # `max(seqlen_k - 1, 0)`. `fx.Index` is unsigned, so a bare
        # `seqlen_k - 1` wraps at seqlen_k == 0 and the KV clamp would then pin
        # every address to a plausible-looking garbage row.
        _slast_i32 = common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0))
        seq_last = fx.Index(_slast_i32)

        # ---- Address split: 64-bit uniform base + 32-bit divergent offset ----
        # Strides are runtime arguments, never folded: an AOT kernel cannot
        # bake them in, and each tensor carries its own triple because K and V
        # need not share Q's layout and under GQA carry a different head count.
        q_st = (fx.Index(stride_q_batch), fx.Index(stride_q_head), fx.Index(stride_q_seq))
        k_st = (fx.Index(stride_k_batch), fx.Index(stride_k_head), fx.Index(stride_k_seq))
        v_st = (fx.Index(stride_v_batch), fx.Index(stride_v_head), fx.Index(stride_v_seq))
        do_st = (fx.Index(stride_do_batch), fx.Index(stride_do_head), fx.Index(stride_do_seq))
        dq_st = (fx.Index(stride_dq_batch), fx.Index(stride_dq_head), fx.Index(stride_dq_seq))
        sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))

        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        _k_row_off_v = fx.Index(k_row_off)

        # The KV row clamp is unconditional here, unlike the forward, which
        # switches it off for the one schedule that provably cannot over-read.
        # seqlen_k need not be a multiple of BLOCK_N, so the final tile of a
        # ragged sequence addresses rows past the end; the values are discarded
        # by the S mask and the clamp exists only to keep the address inside
        # the allocation.
        _addr_kw = dict(
            seqlen_k=seqlen_k_v,
            seq_last=seq_last,
            hoist=KV_ADDR_HOIST,
            clamp=True,
        )
        q_tbase, q_toff, _ = fmha.make_addr_pair(q_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw)
        do_tbase, do_toff, _ = fmha.make_addr_pair(do_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw)
        dq_tbase, dq_toff, _ = fmha.make_addr_pair(dq_st, head_q, _q_batch_v, _q_row_off_v, **_addr_kw)
        _, _, k_addr = fmha.make_addr_pair(k_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw)
        _, _, v_addr = fmha.make_addr_pair(v_st, head_k, _k_batch_v, _k_row_off_v, **_addr_kw)

        # ---- PADDED_HEAD column handling ----
        # One rule: an element is valid iff its column < hdim. A chunk whose
        # start is past hdim is redirected to column 0 and masked away
        # wholesale; a chunk that straddles is loaded and masked per element.
        qk_cols = fmha.MaskedAxis(fx.Index(hdim_qk), active=PADDED_HEAD, elem_dtype=elem_dtype)
        vo_cols = fmha.MaskedAxis(fx.Index(hdim_vo), active=PADDED_HEAD, elem_dtype=elem_dtype)

        # ---- The staged apertures ----
        # K and V both transit LDS as row-major tiles read with the identical
        # access pattern (GEMM1 and GEMM2 differ only in which register operand
        # they pair with). Each still carries its own geometry, because their
        # column bounds are different runtime extents.
        # Loop-carried values: the dQ accumulators, plus the prefetched K/V
        # batches when the distance-1 schedule is on. `scf_yield_` returns a
        # bare value rather than a list when this is 1.
        _CARRY_N = D_CHUNKS + (NUM_BATCHES_K + NUM_BATCHES_V if KV_PREFETCH_DIST else 0)
        k_ap = fmha.Aperture(
            qk_cols,
            lds_base=0,
            lds_stride=K_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=K_TPR_LOAD,
            rows_per_batch=K_ROWS_PER_BATCH,
            num_batches=NUM_BATCHES_K,
            needs_guard=K_NEEDS_GUARD,
        )
        v_ap = fmha.Aperture(
            vo_cols,
            lds_base=LDS_V_BASE,
            lds_stride=V_STRIDE,
            vec_width=VEC_WIDTH,
            threads_per_row=V_TPR_LOAD,
            rows_per_batch=V_ROWS_PER_BATCH,
            num_batches=NUM_BATCHES_V,
            needs_guard=V_NEEDS_GUARD,
        )
        # A third aperture over the *same* tensor: K^T is a different placement
        # of K, so it needs its own LDS base and stride while sharing the
        # column bound. Only live when the transposed layout is selected.
        kt_ap = fmha.Aperture(
            qk_cols,
            lds_base=LDS_KT_BASE,
            lds_stride=KT_STRIDE,
            vec_width=VEC_WIDTH,
        )
        kt_tiling = fmha.TransposedTiling(
            d_blocks=KT_D_BLOCKS,
            tiles=_KT_TILES,
            loads=KT_LOADS,
            needs_guard=KT_NEEDS_GUARD,
            num_waves=NUM_WAVES,
            d_step=WMMA_N,
            kv_step=WMMA_K,
            wave_id=wave_id,
            # Two distinct mappings. The *load* pair says which address this
            # lane supplies so the hardware transpose lands the right block;
            # the *store* pair says where the lane's result belongs in LDS.
            load_d_off=((lane // 8) % 2) * WMMA_LANE_K,
            load_kv_off=(lane // 16) * WMMA_LANE_K + (lane % 8),
            store_d_off=lane16,
            store_kv_off=klane * WMMA_LANE_K,
        )

        def load_global_f16xN(base_ptr, base64, off32):
            return _pointer_load(vxf16_type, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def load_global_v8f16(base_ptr, base64, off32):
            return _pointer_load(v8f16_type, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        def store_global_f16(base_ptr, off64, val):
            """One dB element. Scalar because the guard is per column."""
            _pointer_store(val, buffer_ops.get_element_ptr(base_ptr, fx.Int64(off64), elem_type=elem_type))

        def load_global_f16(base_ptr, off64):
            """One bias element, for the tail tile. Scalar for the same reason
            the dB store is: past `seqlen_k` the guard is per column, and a
            v8 would issue the whole line regardless of it."""
            return _pointer_load(elem_type, buffer_ops.get_element_ptr(base_ptr, fx.Int64(off64), elem_type=elem_type))

        def store_global_v8(base_ptr, base64, off32, val):
            _pointer_store(val, fmha.split_ptr(base_ptr, base64, off32, elem_type))

        # How each tensor is read: its address split, paired with the load
        # instruction that consumes it. `start -> (row, col) -> value`, which
        # is what every movement helper in `fmha_common` takes.
        fetch_k = fmha.reader(k_addr, lambda b, o: load_global_f16xN(k_ptr, b, o))
        fetch_v = fmha.reader(v_addr, lambda b, o: load_global_f16xN(v_ptr, b, o))
        fetch_k_tr = fmha.reader(k_addr, lambda b, o: fmha.global_load_tr_v8(k_ptr_i64, b, o, v8f16_type))

        # ---- Q and dO preload ----
        # One 16-row sub-tile per wave, held in registers for the whole KV
        # loop. Both are indexed by the query head and share the Q row bound.
        q_row = start_q + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        # Intra-tile row, bounded by BLOCK_M so the divergent offset stays small.
        q_row_in_tile = wave_q_offset + lane16

        # Bias is (B, H, Sq, Sk): its last axis is the KV column, so its "row"
        # stride is stride_db_seq_q and the contiguous axis is the one the KV loop
        # walks. Indexed with the same (batch, q_row_off) the varlen decode
        # produced, so it inherits every layout for free -- the forward's
        # `sdpa-bias-plan.md` 3 argument, unchanged. One row per lane, since a
        # lane owns one Q row for the whole KV loop.

        q_rows_axis = fmha.MaskedAxis(fx.Index(seqlen_q_i32))
        q_ap = fmha.Aperture(qk_cols, rows=q_rows_axis)
        do_ap = fmha.Aperture(vo_cols, rows=q_rows_axis)
        q_tile_base = q_tbase(_q_start_addr)
        do_tile_base = do_tbase(_q_start_addr)

        def fetch_q(row, col):
            return load_global_v8f16(q_ptr, q_tile_base, q_toff(row, col))

        def fetch_do(row, col):
            return load_global_v8f16(do_ptr, do_tile_base, do_toff(row, col))

        # Once per row, not once per column: starting the compare early has
        # measured 6% on the forward's widest causal build, which is why
        # `read_v8` takes the gate rather than recomputing it.
        _q_in, _q_safe = q_ap.rows.gate(q_row, q_row_in_tile)

        # Bias is (B, H, Sq, Sk): its last axis is the KV column, so its "row"
        # stride is the seq_q one and the contiguous axis is the one the KV
        # loop walks. Built here rather than earlier because it needs the row
        # gate: a lane whose Q row is past `seqlen_q` still executes every
        # iteration, and an unclamped row walks off the tensor.
        #
        # `_q_in` with the **absolute** `q_row`, not `_q_safe` -- `MaskedAxis.
        # safe` returns the offset *within the tile*, because that is what the
        # Q/dO address closures want on top of a per-tile base. B has no such
        # base and is indexed absolutely, so `_q_safe` here silently addresses
        # row `q - start_q`: correct only for the first Q block, wrong for
        # every other one.
        if const_expr(BIAS_TYPE):
            _bq = fx.Index(q_row) if _q_in else fx.Index(0)
            _b_ptr = fmha.pointer_to_llvm_ptr(B)
            _b_row = (
                _q_batch_v * fx.Index(stride_b_batch)
                + head_q * fx.Index(stride_b_head)
                + (_q_row_off_v + _bq) * fx.Index(stride_b_seq_q)
            )
            # DB is a different tensor and may be laid out differently, so it
            # gets its own row rather than reusing B's.
            _db_row = (
                _q_batch_v * fx.Index(stride_db_batch)
                + head_q * fx.Index(stride_db_head)
                + (_q_row_off_v + _bq) * fx.Index(stride_db_seq_q)
            )
            _db_ptr = fmha.pointer_to_llvm_ptr(DB)
        q_packs = []
        do_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            _c = shard_qk_off + fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
            q_packs.append(q_ap.read_v8(fetch_q, _q_safe, _c, _q_in))
        for ks in range_constexpr(K_STEPS_VO):
            _c = shard_qk_off + fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
            do_packs.append(do_ap.read_v8(fetch_do, _q_safe, _c, _q_in))

        # ---- logsumexp and delta, one scalar per Q row ----
        #
        # Both are (rows,) f32 tensors in the *logsumexp* layout, so one offset
        # serves both -- which is also why the interface requires delta to be
        # allocated like the LSE it accompanies rather than in some third
        # shape. See `lse_token_pitch`; the layout bit is VarlenBits 17:16.
        #
        # The token index is clamped rather than branched on. A row past
        # `seqlen_q` still has to produce an address, and under a stacked
        # layout a trailing zero-length sequence puts `row_off` exactly at the
        # end of the tensor -- so the clamp is to `tokens - 1`, which is inside
        # the allocation for both layouts. The value read there is then
        # discarded by the select below.
        _tok_i32 = lse_tokens
        _lse_tok = common_utils.smin(q_row_off + q_row_i32, _tok_i32 - fx.Int32(1))
        _nhq_v = fx.Index(num_head_q)
        _tok_v = fx.Index(_tok_i32)
        _tok_row = fx.Index(common_utils.smax(_lse_tok, fx.Int32(0)))
        #   _HT  (H, T)   AOTriton's, and the default: T contiguous
        #   _TH  (T, H)   Transformer Engine's:         H contiguous
        _lse_base, _lse_pitch = fmha.lse_row_addressing(varlen_bits, _q_batch_v, head_q, num_head_q, _tok_i32, _tok_row)
        # `_tok_row` is already the clamped absolute row, so the offset is the
        # base itself -- this caller wants one row, not a range.
        _lse_off = _lse_base

        def _load_row_f32(ptr):
            return _pointer_load(
                f32_ty,
                buffer_ops.get_element_ptr(
                    fmha.pointer_to_llvm_ptr(ptr),
                    fx.Int64(_lse_off),
                    elem_type=f32_ty,
                ),
            )

        # A row past seqlen_q reads a clamped, meaningless slot; zero it. With
        # l_i = 0 and delta = 0 the row's Q and dO packs are already zero, so
        # dS comes out exactly 0 and the row contributes nothing -- and its
        # store is guarded anyway.
        _lse_nat = fx.Float32(fx.Float32(_load_row_f32(LSE)) if _q_in else fx.Float32(0.0))
        delta_i = fx.Float32(fx.Float32(_load_row_f32(Delta)) if _q_in else fx.Float32(0.0))
        # The forward writes LSE in natural units; the exponent here is base 2,
        # so this is AOTriton's `l_i = tl.load(...) * RCP_LN2` exactly.
        #
        # A row the forward found no keys for carries +inf, deliberately (see
        # its epilogue): exp2(anything - inf) is 0, which is what a row that
        # must contribute nothing needs.
        lse2_i = fastmath.mul(_lse_nat, fx.Float32(_LOG2E))

        if const_expr(ENABLE_DROPOUT):
            # The offset scheme, and the only dropout-specific arithmetic in
            # this file. It must be *identical* to the forward's or the
            # regenerated mask differs and the gradient is silently wrong,
            # which is why both call `Philox.grid_plane` / `grid_offset`
            # rather than transcribing the formula.
            _off_zh = fx.Int32(z_i32) * fx.Int32(num_head_q) + fx.Int32(head_q)
            _ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            _ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            _ph_base, _ph_stride = PHILOX.grid_plane(_ph_off, _off_zh, max_seqlen_q, max_seqlen_k)

        # ---- Constants ----
        # Genuinely -inf, so exp2(-inf - lse) is exactly 0. There is no `m_i`
        # floor to worry about here, unlike the forward: this kernel never
        # takes a row max, it reads the forward's logsumexp instead.
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        sm_scale_vec = Vec.from_elements([sm_scale], fx.Float32).broadcast_to(8).ir_value()

        # ---- The visible band ----
        if const_expr(CAUSAL):
            # Sentinels resolve per sequence, not on the host. Everything
            # derived from a window stays i32 from here down: bounds go
            # negative and `fx.Index` is unsigned.
            _wl_i32, _wr_i32 = fmha.resolve_window(window_left, window_right, seqlen_q_i32, seqlen_k_i32)

        # ---- Split the KV range into full and masked regions ----
        # Emitting the regions as separate loops means the masks exist only in
        # the masked one, with the split structural rather than a per-tile
        # branch. Causal has three regions, not two: a left window kills
        # columns at the start of the range as well as the end.
        if const_expr(CAUSAL):
            _regions = fmha.decompose_causal_regions(
                start_q,
                seqlen_q_i32,
                seqlen_k_i32,
                _wl_i32,
                _wr_i32,
                BLOCK_M,
                BLOCK_N,
                _alive,
            )
            _BN_I32 = fx.Int32(BLOCK_N)
            _n_l, _n_f, _n_r = _regions.n_left, _regions.n_full, _regions.n_right
            _l_col0, _f_col0 = _regions.left_col0, _regions.full_col0
            _r_col0, _m_col0 = _regions.right_col0, _regions.masked_col0
            _n_masked = fx.Index(_n_l + _n_r)
        else:
            # No mask beyond the KV tail: one full region, then one partial
            # tile. `kv_upper` is rounded *up* to a whole BLOCK_N because the
            # loop steps by BLOCK_N, and a bound that is not a multiple of it
            # would drop the final partial tile entirely.
            _full_end = (seqlen_k_v // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N)
            kv_upper = fx.Index(((seqlen_k_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N))
            # A dead workgroup walks nothing.
            _full_end = fx.Index(_full_end if _alive else fx.Index(0))
            kv_upper = fx.Index(kv_upper if _alive else fx.Index(0))

        def _read_k(col0):
            return fmha.read_batches(k_ap, fetch_k(col0), load_row_in_batch, load_col_base)

        def _read_v(col0):
            return fmha.read_batches_unmasked(v_ap, fetch_v(col0), v_row_in_batch, v_col_base)

        init_args = [fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(D_CHUNKS)]
        if const_expr(KV_PREFETCH_DIST):
            if const_expr(CAUSAL):
                _kv_tiles_i32 = _n_f + _n_l + _n_r
            else:
                _kv_tiles_i32 = fx.Int32(kv_upper)
            _pf_n = fx.Index(fx.Int32(1) if _kv_tiles_i32 > fx.Int32(0) else fx.Int32(0))
            # Seeded through a 0-or-1-trip `range(..., init=)`, not a dynamic
            # `if`: at seqlen_k == 0 there is no harmless address to clamp to,
            # and FlyDSL's dynamic `if` merges named scalars only while a loop
            # carries exactly the list of vectors a prefetch produces. The trip
            # count is uniform across the workgroup, so no divergence.
            # The first tile actually walked, which is not always the full
            # run's origin: with no full tiles the masked run goes first. The
            # seed feeds iteration 0's publish, so getting this wrong corrupts
            # the very first tile -- and it is invisible on any shape whose
            # full run is non-empty.
            if const_expr(CAUSAL):
                _first_col = fx.Index(_f_col0 if _n_f > fx.Int32(0) else _m_col0)
            else:
                _first_col = fx.Index(0)
            _pf_init = [
                Vec.filled(VEC_WIDTH, 0.0, elem_dtype).ir_value()
                for _ in range_constexpr(NUM_BATCHES_K + NUM_BATCHES_V)
            ]
            _pf = _pf_init
            for _pfi, _pf_args in range(fx.Index(0), _pf_n, 1, init=_pf_init):
                _pf = yield _read_k(_first_col) + _read_v(_first_col)
            # `scf_yield_` returns a bare value, not a list, when the loop
            # carries exactly one.
            if const_expr(len(_pf_init) == 1):
                _pf = [_pf]
            init_args = init_args + [_pf[_i] for _i in range_constexpr(NUM_BATCHES_K + NUM_BATCHES_V)]
        loop_results = init_args

        def kv_loop_body(kv_block_start, inner_iter_args, _MASK_STEPS, next_kv_start=None):
            """One KV tile. `_MASK_STEPS` is a Python bool resolved at trace
            time, so the masked and unmasked regions emit different code.

            `next_kv_start` is the tile a distance-1 prefetch should fetch. It
            defaults to the following tile, which is right whenever the region
            being walked is contiguous; the causal masked loop walks two
            disjoint runs and passes the piecewise successor explicitly.
            Getting that wrong fetches the wrong tile and is **invisible to a
            correctness test**, because the value is overwritten before use --
            the forward's copy of this carries the same warning."""
            # `inner_iter_args` is always a list, even at BLOCK_DMODEL 16 where
            # the loop carries exactly one value. The *loop result* is not --
            # see the unwrap after each loop below.
            dq_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]
            if const_expr(KV_PREFETCH_DIST):
                _kv = [inner_iter_args[D_CHUNKS + i] for i in range_constexpr(NUM_BATCHES_K + NUM_BATCHES_V)]
                _k_cur, _v_cur = _kv[:NUM_BATCHES_K], _kv[NUM_BATCHES_K:]
            if const_expr(next_kv_start is None):
                next_kv_start = kv_block_start + fx.Index(BLOCK_N)

            # Everyone must be done reading the previous tile before it is
            # overwritten. Two barriers per iteration, as in the forward.
            gpu.barrier()
            if const_expr(KV_PREFETCH_DIST):
                fmha.publish(k_ap, lds_kv, _k_cur, load_row_in_batch, load_col_base, fx.Index(BLOCK_N))
                fmha.publish(v_ap, lds_kv, _v_cur, v_row_in_batch, v_col_base, fx.Index(BLOCK_N))
                # Issued after the store and before the barrier, so the load
                # flies over all three GEMMs rather than only the back-edge.
                _k_next, _v_next = _read_k(next_kv_start), _read_v(next_kv_start)
            else:
                fmha.stage(
                    k_ap,
                    lds_kv,
                    fetch_k(kv_block_start),
                    load_row_in_batch,
                    load_col_base,
                    fx.Index(BLOCK_N),
                )
                fmha.stage(
                    v_ap,
                    lds_kv,
                    fetch_v(kv_block_start),
                    v_row_in_batch,
                    v_col_base,
                    fx.Index(BLOCK_N),
                )
            if const_expr(KT_TRANSPOSED):
                fmha.publish_transposed(
                    kt_ap,
                    kt_tiling,
                    lds_kv,
                    fmha.read_transposed(kt_ap, kt_tiling, fetch_k_tr(kv_block_start)),
                )
            gpu.barrier()

            # ==== GEMM1: S^T = K @ Q^T ====
            # A = K row-major (lane16 = kv row), B = the Q register preload.
            # Accumulator element si is KV column klane*8 + si at Q row lane16.
            s_accs = [fx.as_ir_value(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
            for ks in range_constexpr(K_STEPS_QK):
                k_col = shard_qk_off + fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                for a_idx in range_constexpr(NUM_S_ACCS):
                    k_pack = k_ap.from_lds(lds_kv, lane16 + fx.Index(a_idx * WMMA_N), k_col)
                    s_accs[a_idx] = fmha.wmma_acc(k_pack, q_packs[ks], s_accs[a_idx])

            # ==== GEMM2: dP^T = V @ dO^T ====
            # The identical access pattern with V in place of K and dO in place
            # of Q, so the accumulators share GEMM1's element mapping.
            dp_accs = [fx.as_ir_value(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
            for ks in range_constexpr(K_STEPS_VO):
                v_col = shard_qk_off + fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                for a_idx in range_constexpr(NUM_S_ACCS):
                    v_pack = v_ap.from_lds(lds_kv, lane16 + fx.Index(a_idx * WMMA_N), v_col)
                    dp_accs[a_idx] = fmha.wmma_acc(v_pack, do_packs[ks], dp_accs[a_idx])

            # ==== Cross-shard reduction of S and dP ====
            # Each shard-wave holds a partial sum over its own slice of the
            # head dim; the full S and dP are their sums. Both go through one
            # call so they share the two barriers -- they are needed at the
            # same point and neither can start before GEMM2 ends.
            if const_expr(QK_SHARDS > 1):
                if const_expr(RED_ALIASES_V):
                    # The partials land on top of the V tile, so every wave
                    # must be past its GEMM2 reads first.
                    gpu.barrier()
                _red = fmha.reduce_s_across_shards(
                    s_accs + dp_accs,
                    lds_byte_base=_lds_byte_base,
                    byte0=RED_BYTE0,
                    wave_id=wave_id,
                    lane=lane,
                    shard_id=shard_id,
                    q_tile_in_block=q_tile_in_block,
                    num_shards=QK_SHARDS,
                    f32_per_wave=RED_F32_PER_WAVE,
                    warp_size=WARP_SIZE,
                    fastmath=fastmath,
                )
                s_accs = _red[:NUM_S_ACCS]
                dp_accs = _red[NUM_S_ACCS:]

            s_raw = []
            dp_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_raw.append(Vec(s_accs[st])[r])
                    dp_raw.append(Vec(dp_accs[st])[r])

            # Scaled before the mask, exactly as in the forward: the mask must
            # run *after* the multiply, or a -inf written first can be folded
            # away by a fast-math multiply.
            s_raw = [fastmath.mul(v, sm_log2e) for v in s_raw]

            if const_expr(BIAS_TYPE):
                # After the scale because the exponent lives in the base-2
                # scaled domain, so a bias in natural units is multiplied by
                # log2(e) first -- AOTriton's `qk += bias * 1.44269504089`.
                # Before the mask so a column past seqlen_k stays -inf rather
                # than becoming -inf + bias.
                #
                # Not a gather: element `_st*8 + _r` is KV column
                # `_st*16 + klane*8 + _r`, so within a group only `_r` varies
                # and one v8 load covers all eight.
                # **Both the value and the address are guarded, and the tail
                # tile pays for it one element at a time.** Zeroing the value
                # alone was not enough: the v8 access is still issued, so a
                # group whose columns are all past `seqlen_k` reads a whole
                # 16-byte line the bias tensor does not own. The same omission
                # in the forward faulted for real (AOTriton `test_irregulars`
                # seqlen_k 1063, hdim 24).
                #
                # Sliding the eight-wide base back into the row is what cannot
                # be done -- it shifts which column each element carries, so
                # the live ones would take a neighbour's bias. Clamping *per
                # element* has no such problem, because a dead element's value
                # is discarded anyway. Same contract dK/dV's bias load has.
                #
                # Only the tail tile is affected: full tiles end at
                # `blk_last_whole`, so every column of one exists, and
                # `_MASK_STEPS` is `const_expr` so they keep the v8.
                if const_expr(_MASK_STEPS):
                    _bkv = fx.Int32(kv_block_start)
                    _bklane = fx.Int32(klane) * fx.Int32(8)
                    for _i in range_constexpr(NUM_S_VALS):
                        _bcol = _bkv + fx.Int32(fmha.acc_elem_column(_i)) + _bklane
                        _bin = _bcol < seqlen_k_i32
                        _bsafe = fx.Index(fx.Int32(_bcol) if _bin else fx.Int32(0))
                        _braw = fx.Float32(fx.as_dsl_value(load_global_f16(_b_ptr, _b_row + _bsafe)).to(fx.Float32))
                        _bs = fx.Float32(_braw if _bin else fx.Float32(0.0))
                        s_raw[_i] = fastmath.add(s_raw[_i], fastmath.mul(_bs, fx.Float32(_LOG2E)))
                else:
                    for _st in range_constexpr(NUM_S_ACCS):
                        _c0 = fx.Int32(kv_block_start) + fx.Int32(_st * WMMA_N) + fx.Int32(klane) * fx.Int32(8)
                        _bv = load_global_v8f16(_b_ptr, _b_row + fx.Index(_c0), fx.Index(0))
                        for _r in range_constexpr(8):
                            _bs = fx.Float32(Vec(_bv)[_r].to(fx.Float32))
                            s_raw[_st * 8 + _r] = fastmath.add(
                                s_raw[_st * 8 + _r], fastmath.mul(_bs, fx.Float32(_LOG2E))
                            )

            if const_expr(_MASK_STEPS):
                # Element i of the flattened accumulators is KV column
                #   (i//16)*32 + ((i//8)%2)*16 + klane*8 + i%8
                # -- the GEMM unroll walks (sub-tile, half) pairs, and within a
                # 16-row WMMA block a lane holds rows klane*8 + si.
                #
                # Two conditions, one select: a column is dead if it is past
                # seqlen_k, or (causal) outside this row's band. No `scf.if`
                # guard around it -- the region split already made this code
                # reachable only from tiles that need it, and a runtime branch
                # inside a latency-bound loop measured far worse than the
                # selects on the forward.
                _kv_i32 = fx.Int32(kv_block_start)
                _klane_off = fx.Int32(klane) * fx.Int32(8)
                for _i in range_constexpr(NUM_S_VALS):
                    _col = _kv_i32 + fx.Int32(fmha.acc_elem_column(_i)) + _klane_off
                    _dead = _col >= seqlen_k_i32
                    if const_expr(CAUSAL):
                        # Both edges of the band, signed throughout:
                        # `q_row - w_left` is negative for every row when
                        # w_left is "unbounded" (== seqlen_q), which is how
                        # plain causal maps onto this path.
                        _dead = _dead | (_col > q_row_i32 + _wr_i32)
                        _dead = _dead | (_col < q_row_i32 - _wl_i32)
                    s_raw[_i] = c_neg_inf if _dead else s_raw[_i]

            # P, from the forward's logsumexp rather than from a running max:
            # the backward pass has the denominator already. A dead column is
            # -inf here and exp2 takes it to exactly 0, which is what makes dS
            # zero there whatever dP holds.
            p_vals = []
            for _i in range_constexpr(NUM_S_VALS):
                p_vals.append(rocdl.exp2(f32_ty, fx.as_ir_value(fastmath.sub(fx.Float32(s_raw[_i]), lse2_i))))

            if const_expr(ENABLE_DROPOUT):
                # AOTriton masks **dP**, not P: P is the undropped probability
                # (the logsumexp it came from is the undropped sum), and the
                # dropped entries contribute nothing to dO. A group of eight
                # consecutive elements is eight contiguous KV columns, so each
                # group is one span of the stream.
                for _st in range_constexpr(NUM_S_ACCS):
                    _bcol = fx.Int64(kv_block_start) + _st * WMMA_N + fx.Int64(fx.Int32(klane) * 8)
                    _first = PHILOX.grid_offset(_ph_base, _ph_stride, q_row, _bcol)
                    _keep = PHILOX.keep_span(_ph_seed, _first, 8, idropout_p)
                    for _r in range_constexpr(8):
                        _i = _st * 8 + _r
                        dp_raw[_i] = fx.as_ir_value(
                            fastmath.mul(fx.Float32(dp_raw[_i]), dropout_scale) if _keep[_r] else fx.Float32(0.0)
                        )

            # dS = P * (dP - delta). One flat unroll; both operands share the
            # accumulator element mapping, and delta is a per-lane scalar
            # because it belongs to the Q row, which is lane16.
            ds_vals = []
            for _i in range_constexpr(NUM_S_VALS):
                ds_vals.append(
                    fastmath.mul(
                        fx.Float32(p_vals[_i]),
                        fastmath.sub(fx.Float32(dp_raw[_i]), delta_i),
                    )
                )

            if const_expr(BIAS_TYPE):
                # dB = dS. The bias is added to the score, so d(score)/d(bias)
                # is 1 and the gradient the kernel already has for GEMM3 *is*
                # the bias gradient -- this is a store, not a computation.
                #
                # Written from the dQ kernel rather than dK/dV because this one
                # walks KV tiles for a fixed Q block, so the eight elements of
                # a group are eight contiguous dB columns and the store is one
                # v8; dK/dV would walk down a column. And because bias excludes
                # causal, the visited region is the full rectangle, so between
                # them the Q blocks cover every (q, kv) exactly once.
                # **Bounded, and per element.** A v8 here would be wrong at
                # any ragged extent: dB's KV pitch is exactly `seqlen_k` with
                # no padding, so a chunk straddling the end lands in the *next
                # row*. That is unlike `fmha.write_v8`, whose whole-chunk skip
                # is exact only because the head-dim pitch is a 16-byte
                # multiple. `dS` is already 0 in those columns -- masked to
                # -inf, so `p` is 0 -- which is precisely why the address, not
                # the value, is what has to be guarded.
                #
                # The row guard is the same point: a lane past `seqlen_q` has
                # a clamped `_q_safe`, so an unguarded store would write real
                # row 0's gradient.
                if _q_in:
                    for _st in range_constexpr(NUM_S_ACCS):
                        _c0 = fx.Int32(kv_block_start) + fx.Int32(_st * WMMA_N) + fx.Int32(klane) * fx.Int32(8)
                        for _j in range_constexpr(8):
                            if _c0 + fx.Int32(_j) < seqlen_k_i32:
                                store_global_f16(
                                    _db_ptr,
                                    _db_row + fx.Index(_c0 + fx.Int32(_j)),
                                    fx.Float32(ds_vals[_st * 8 + _j]).to(elem_dtype),
                                )

            # ==== Build the dS packs ====
            # Truncated to the input's 16-bit type because RDNA4 WMMA has no
            # F32xF32 form: A/B operands are f16/bf16/iu8/iu4/fp8 only (ISA
            # manual Table 41). See `fmha.bf16_trunc_pack_v8` for the accuracy
            # measurement behind the bf16 spelling.
            ds_packs = []
            for _acc in range_constexpr(NUM_S_ACCS):
                _slice = [ds_vals[_acc * 8 + j] for j in range(8)]
                if const_expr(dtype_str == "bf16"):
                    ds_packs.append(fmha.bf16_trunc_pack_v8(_slice, elem_dtype))
                else:
                    ds_packs.append(
                        Vec.from_elements(
                            [fx.Float32(_slice[j]).to(elem_dtype) for j in range_constexpr(8)],
                            elem_dtype,
                        ).ir_value()
                    )

            def _load_kt(dc_val, acc_idx):
                """One K^T WMMA A-operand: A[d = lane16][kv = klane*8 + j]."""
                d_pos = shard_qk_off + fx.Index(dc_val * WMMA_N) + lane16
                kv0 = fx.Index(acc_idx * WMMA_N) + klane * WMMA_LANE_K
                if const_expr(KT_TRANSPOSED):
                    # K^T[d][kv]: this lane's 8 kv values are contiguous, so
                    # one vector read replaces the eight scalar ones below.
                    return kt_ap.from_lds(lds_kv, d_pos, kv0)
                elems = []
                for j in range_constexpr(8):
                    elems.append(fx.ptr_load(lds_kv + fx.Int32(k_ap.lds_index(kv0 + fx.Index(j), d_pos))))
                return Vec.from_elements(elems, elem_dtype).ir_value()

            # ==== GEMM3: dQ[d][q] += K^T[d][kv] * dS^T[kv][q] ====
            # The accumulator lands in the forward's O layout -- eight
            # contiguous d per lane at one Q row -- which is what makes the
            # epilogue a single v8 store per 16 columns.
            for dc in range_constexpr(D_CHUNKS):
                for a_idx in range_constexpr(NUM_S_ACCS):
                    dq_accs[dc] = fmha.wmma_acc(_load_kt(dc, a_idx), ds_packs[a_idx], dq_accs[dc])

            _out = [dq_accs[i] for i in range_constexpr(D_CHUNKS)]
            if const_expr(KV_PREFETCH_DIST):
                _out = _out + _k_next + _v_next
            return _out

        if const_expr(CAUSAL):
            # Full region first, then the two masked runs walked as one loop
            # over a piecewise index. Two emitted bodies, not three: the body
            # is large and a third copy costs registers for nothing.
            for kv_block_start, inner_iter_args in range(
                fx.Index(_f_col0),
                fx.Index(_f_col0 + _n_f * _BN_I32),
                BLOCK_N,
                init=init_args,
            ):
                _nxt = fx.Index(
                    fx.Int32(kv_block_start) + _BN_I32
                    if fx.Int32(kv_block_start) + _BN_I32 < _f_col0 + _n_f * _BN_I32
                    else _m_col0
                )
                loop_results = yield kv_loop_body(kv_block_start, inner_iter_args, False, next_kv_start=_nxt)
            # `yield` hands back a bare value rather than a one-element list
            # when the loop carries exactly one, which happens at
            # BLOCK_DMODEL 16 (one dQ accumulator). Indexing that would extract
            # a vector *element*, and the failure surfaces far away, in the
            # next loop's WMMA operand type.
            if const_expr(_CARRY_N == 1):
                loop_results = [loop_results]

            def _masked_col(i_idx):
                """Tile column for masked iteration i: the left run, then the
                right one. Discontinuous at the seam."""
                _i = fx.Int32(i_idx)
                return _l_col0 + _i * _BN_I32 if _i < _n_l else _r_col0 + (_i - _n_l) * _BN_I32

            for _mi, inner_iter_args in range(fx.Index(0), _n_masked, 1, init=loop_results):
                loop_results = yield kv_loop_body(
                    fx.Index(_masked_col(_mi)),
                    inner_iter_args,
                    True,
                    next_kv_start=fx.Index(_masked_col(fx.Int32(_mi) + fx.Int32(1))),
                )
        else:
            # Region 1: tiles wholly inside seqlen_k -- no masking emitted.
            for kv_block_start, inner_iter_args in range(fx.Index(0), _full_end, BLOCK_N, init=init_args):
                loop_results = yield kv_loop_body(kv_block_start, inner_iter_args, False)
            # See the causal arm: one carried value comes back bare.
            if const_expr(_CARRY_N == 1):
                loop_results = [loop_results]

            # Region 2: the ragged tail.
            for kv_block_start, inner_iter_args in range(_full_end, kv_upper, BLOCK_N, init=loop_results):
                loop_results = yield kv_loop_body(kv_block_start, inner_iter_args, True)

        if const_expr(_CARRY_N == 1):
            loop_results = [loop_results]

        # ---- Scale and store dQ ----
        # dQ shares Q's rows and columns and is never staged.
        dq_ap = fmha.Aperture(qk_cols, rows=q_rows_axis)
        dq_tile_base = dq_tbase(start_q)

        def write_dq(row, col, val):
            store_global_v8(dq_ptr, dq_tile_base, dq_toff(row, col), val)

        # sm_scale is applied once at the end rather than folded into dS: the
        # exponent already carries `sm_scale * log2(e)`, and dS feeds a 16-bit
        # WMMA operand, so scaling it there would round the scale into f16.
        if _q_in:
            for dc in range_constexpr(D_CHUNKS):
                _scaled = fastmath.mul(loop_results[dc], sm_scale_vec)
                _trunc = Vec(_scaled).to(elem_dtype).ir_value()
                # Each shard owns a disjoint column range of dQ, so there is
                # nothing to combine here -- the store is the reduction.
                _col = shard_qk_off + fx.Index(dc * WMMA_N) + klane * 8
                fmha.write_v8(dq_ap, write_dq, q_row_in_tile, _col, _trunc)

    @flyc.jit
    def launch_bwd_dq(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        DO: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        nseq_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        # The grid's Q extent keys on Max_seqlen_q: under varlen there is no
        # single seqlen_q, so every sequence gets the longest one's worth of
        # workgroups and the short ones exit empty.
        num_q_tiles = (fx.Index(max_seqlen_q) + BLOCK_M - 1) // BLOCK_M

        launcher = bwd_dq_kernel(
            Q,
            K,
            V,
            B,
            DO,
            DQ,
            DB,
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
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
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
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("amdgpu-sched-strategy"),
                        ir.StringAttr.get(SCHED_STRATEGY),
                    ]
                )
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
            if const_expr(FP_MODE == "fast"):
                passthrough_entries.append(
                    ir.ArrayAttr.get([ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")])
                )
                passthrough_entries.append(
                    ir.ArrayAttr.get([ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")])
                )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(fx.Index(num_head_q), num_q_tiles, nseq_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_bwd_dq.compile_hints = {
        "FAST_FP_MATH": FAST_FP_MATH,
        "UNSAFE_FP_MATH": UNSAFE_FP_MATH,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    # Causal alignment is expressed as a *sentinel* window, resolved in the
    # kernel against each sequence's own lengths. The host does not resolve it:
    # under varlen bottom-right needs `seqlen_k[z] - seqlen_q[z]`, which
    # differs per sequence, and a host-side resolution silently gives every
    # sequence the batch-wide difference.

    # ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
    # One byte per side, decoded by the same kernel-side function twice, plus
    # the LSE layout in byte 2. `0` is the dense case and the default. Spelled
    # out here rather than imported from the forward kernel module so that this
    # file states the wire encoding it implements.

    def _args(
        Q,
        K,
        V,
        DO,
        DQ,
        lse,
        delta,
        batch_size,
        num_seqlens,
        seqlen_q,
        seqlen_k,
        scale,
        window,
        varlen,
        dropout_p,
        seed,
        off1,
        off2,
        bias,
        dbias,
    ):
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta_t, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)], q_heads=("DO", "DQ")
        )
        _lp = abi.row_tensor_arg(lse, "logsumexp", meta_t[0], seqlen_q, varlen)
        _dp = abi.row_tensor_arg(delta, "delta", meta_t[0], seqlen_q, varlen)
        _wl, _wr = abi.resolve_window(CAUSAL_TYPE, HOST_CAUSAL_TYPE, window, seqlen_q, seqlen_k)
        _vb, _sq0, _sq1, _sk0, _sk1, _mq, _mk = abi.varlen_args(
            False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens
        )
        _ps, _po1, _po2, _ip, _dsc, _hold = abi.dropout_args(ENABLE_DROPOUT, dropout_p, seed, off1, off2, Q.device)
        _bp, _dbp, _bs, _dbs = abi.bias_args(BIAS_TYPE, True, bias, dbias, Q)
        return (
            # The shared backward pointer block is
            #     Q, K, V, B, DO, <outputs>, LSE, Delta
            # and `<outputs>` is what differs between the split kernels: here
            # it is (DQ, DB), in dK/dV (DK, DV). `ptrs` arrives as
            # (Q, K, V, DO, DQ), so Bias splices in ahead of DO and DB joins
            # DQ as the second output.
            *ptrs[:3],
            _bp,
            ptrs[3],
            ptrs[4],
            _dbp,
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
            *_dbs,
        )

    def _launch(
        Q,
        K,
        V,
        DO,
        DQ,
        lse,
        delta,
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
        dbias=None,
    ):
        args = _args(
            Q,
            K,
            V,
            DO,
            DQ,
            lse,
            delta,
            batch_size,
            num_seqlens,
            seqlen_q,
            seqlen_k,
            scale,
            window,
            varlen,
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            bias,
            dbias,
        )
        abi.run_compiled(_COMPILED, launch_bwd_dq, *args, stream if stream is not None else fx.Stream(None))

    def _compile(
        Q,
        K,
        V,
        DO,
        DQ,
        lse,
        delta,
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
        dbias=None,
    ):
        args = _args(
            Q,
            K,
            V,
            DO,
            DQ,
            lse,
            delta,
            batch_size,
            num_seqlens,
            seqlen_q,
            seqlen_k,
            scale,
            window,
            varlen,
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            bias,
            dbias,
        )
        return flyc.compile(launch_bwd_dq, *args, fx.Stream(stream))

    _launch.compile = _compile
    _launch.varlen_bits = abi.varlen_bits
    _launch.varlen_compact = abi.varlen_compact
    _launch.varlen_padded = abi.varlen_padded
    _launch.varlen_strided = abi.varlen_strided
    _launch.varlen_seqused_k = abi.varlen_seqused_k
    _launch.block_m = BLOCK_M
    return _launch


def build_bwd_dq_module(**kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    Any `BwdDqKnobs` field may be passed as a keyword to pin it; the rest are
    resolved by `resolve_knobs`. This is what keeps "the tuning module is the
    only producer of a schedule" true even for callers who never mention one.
    """
    meta_fields = {f.name for f in fields(BwdDqInputMetadata)}
    knob_fields = {f.name for f in fields(BwdDqKnobs)}
    unknown = set(kwargs) - meta_fields - knob_fields
    if unknown:
        raise TypeError(f"unknown build parameter(s): {sorted(unknown)}")
    meta = BwdDqInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    overrides = BwdDqKnobs(**{k: v for k, v in kwargs.items() if k in knob_fields})
    return build_bwd_dq_module_primary(meta, resolve_knobs(meta, overrides))
