# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Single-pass fused flash-attention backward for gfx950: dQ, dK and dV in one kernel.

**The dK/dV kernel, with the workgroup widened to own every KV row.** Read
`dkdv.py` first: everything that file says about the resident
K/V, the streamed Q/dO, the four d- and q-contracted GEMMs, causal tile cuts,
GQA folding, dropout and bias applies here unchanged, because this kernel *is*
that body. What one workgroup owning a whole `(batch, kv head)` buys is that a
streamed Q tile has met every key by the time the body is done with it -- so its
dS row block is complete, and

    dQ_tile = sm_scale * dS_tile . K

is final the moment it is computed. It is computed right there and stored: no
dQ accumulator survives a tile, no second kernel re-reads Q/dO/K/V to rebuild S
and dP, no atomics. The split path's three kernels (preprocess, dK/dV, dQ) make
13 full-tensor passes; this makes 8 (Q, K, V, O, dO read once each; dQ, dK, dV
written once each), and 5 GEMMs where the split path runs 7.

--- What is added to the dK/dV body ------------------------------------------

Four things, in the order a Q head meets them:

1. **K is copied to LDS once**, from the registers the dK/dV body already holds
   it in (a 16-row wave's resident K pack is eight contiguous d of one KV row,
   which is one `ds_write_b128`). No extra global read.
2. **A per-head prologue fills two f32 row arrays in LDS**: `-log2(e) * LSE`
   and `delta = rowsum(O * dO)`. That is AOTriton's `bwd_preprocess` fused in:
   O is read exactly once, here, and never kept. Both arrays are then read by
   the tile body with `ds_read_b128` instead of `buffer_load`, so the body
   issues no global loads of its own besides the stream DMA -- which is what
   keeps the explicit `vmcnt` budget below exact.
3. **dS goes to LDS**, as a `[kv][q]` tile in the stream tile's own layout. A
   16-row lane holds eight contiguous q of one KV column after
   `pack_group`, so that too is one `ds_write_b128`.
4. **After the body's bottom barrier, the workgroup computes dQ for the tile**:
   `16x16x32` MFMAs whose A operand is K^T and B operand is dS, *both* read
   through `ds_read_b64_tr_b16` with the published 16-row map
   (`mfma16.a16_read_base`) -- the contraction index is the KV row
   on both sides, so the two maps carry the same permutation (none) and
   nothing has to be shuffled. The output lane holds four contiguous d of one
   q row: one 8-byte store.

--- Why one dS buffer is enough ----------------------------------------------

dS(j) is written between the top and bottom barriers of tile j, and read
between the bottom barrier of j and the top barrier of j+1. dS(j+1) is written
after the top barrier of j+1, which no wave passes before every wave has
finished reading dS(j) -- its dQ MFMAs consume those reads, and they precede
the barrier in program order. So the ring the scoping study proposed collapses
to one tile.

--- The `vmcnt` budget --------------------------------------------------------

The dK/dV body waits for its stream tile with `vmcnt(N)`, `N` being the DMA
issues known to be younger. Here dQ stores are VMEM too, and land between the
DMAs: each tile end issues the prefetch DMA (`ND` ops) and then the tile's dQ
stores (`S` ops). The DMA for tile `j+1` was issued `NUM_STREAM_BUFFERS` tile
ends ago, so what is younger than it is `S + (NBUF - 1) * (ND + S)`.
`FusedM16TileBody.run` waits on exactly that. The prologue ends in a full
`s_waitcnt 0`, so the first tiles of each head start from an empty count and
the formula is never asked to count prologue traffic.

--- Scope ----------------------------------------------------------------------

Both dK/dV families: 16 KV rows per wave (`dkdv_m16`, `FusedM16TileBody`)
wherever it fits, 32 (`Fused32TileBody`) only at head_dim 32 with `BLOCK_KV`
128; `tuning_fused_bwd` has the geometry and why. `BLOCK_KV` up to 256 (head_dim <= 64); 128 at head_dim 128.

Phase 1, tested against fp32 and the split path: bf16 and f16; the dK/dV
ladder's head dims with padded heads; ragged `seqlen_q`/`seqlen_k` including
1 and non-multiples of 16; bottom-right causal; GQA/MQA; `sm_scale`. dQ rows a
causal walk never visits are written as zeros by the prologue.

Phase 2, through the dK/dV body's own hooks and checked against the split
path: the bias input, sliding windows and packed varlen agree, and so does
dropout, which is also checked against fp32 autograd with the kernels' own
mask. Dropout builds sit at the SGPR cliff, where the lost-SGPR-copy backend
defect (`HANDOFF-gfx950-lost-sgpr-copy.md`) corrupted the dK/dV epilogue's
addressing; `FusedKvOutputs` keeps that addressing out of SGPRs across the
walk, and says what was measured. `dB` is not produced; the `DB` slot is in
the ABI so adding it does not move the wire format.
"""

import weakref

import fmha_utils as fmha
from . import dkdv_m16 as m16
from .dkdv import (
    _ROW_RUNS,
    BwdDkDvGemmHelper,
    BwdDkDvKernelContext,
    BwdDkDvResidentLoader,
    BwdDkDvSoftmaxHelper,
    BwdDkDvStreamLoader,
    BwdDkDvStoreHelper,
    BwdDkDvStreamReader,
    BwdDkDvTileBody,
    _masked_ks_steps,
    bias_log2e,
    philox_keep,
)
from fmha_meta import FmhaInputMetadata
from fmha_utils import MaskedAxis
from .dualwave_helpers import (
    _ds_read_tr16_b64_imm,
    _slab_span_elems,
    check_score_tensor,
    traits_cache_key,
    wire_ptr,
    wire_view,
)
from .mfma16 import MFMA16_M, a16_read_base, lds_elem, tok_off, tok_off_dyn
from .tuning_fused_bwd import (
    SCOPE_DS,
    SCOPE_K,
    SCOPE_ROWS,
    entry_plan,
    lanes_per_row,
    rows_per_pass,
)
from .standalone import buffer_ops, dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = [
    "KERNEL_NAME",
    "build_fmha_bwd_fused_gfx950_module",
    "build_fmha_bwd_fused_gfx950_module_primary",
]

KERNEL_NAME = "fmha_bwd_fused_gfx950_kernel"

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier
_waitcnt_vm_n = dualwave._waitcnt_vm_n

_COMPILED = weakref.WeakKeyDictionary()

# An element offset past the end of any dQ descriptor this kernel builds (its
# byte bound is far below 2 GiB), so a store sent there is dropped by the
# hardware bound. A constant rather than the dK/dV kernel's per-slab
# `seqlen * stride`, which would be two more live SGPRs.
_OOB_ELEM = 0x3FFFFFF0

# `s_waitcnt 0`: vmcnt, expcnt and lgkmcnt all zero.
_WAIT_ALL = 0

# The 16x16x32 MFMA's contraction extent and the operand elements one lane holds.
_K32 = 32
_LANE_K = 8
_ACC = 4


# --- scoped LDS access ---------------------------------------------------------


def _lds_ptr(ctx, elem_idx):
    """addrspace(3) pointer to bf16 element `elem_idx` of the one allocation."""
    return buffer_ops.get_element_ptr(
        ctx.lds_kv_base_ptr,
        byte_offset=elem_idx * ctx.traits.BF16_BYTES,
        elem_type=T.i8,
    )


def _lds_load(ctx, result_type, elem_idx, scope, alignment):
    return dualwave.llvm.LoadOp(
        result_type,
        _lds_ptr(ctx, elem_idx),
        alignment=alignment,
        alias_scopes=dualwave._dualwave_lds_alias_scopes(scope),
        noalias_scopes=dualwave._dualwave_lds_noalias_scopes(scope, ctx.traits.LDS_SCOPE_NAMES),
    ).result


def _lds_store(ctx, value, elem_idx, scope, alignment):
    dualwave.llvm.StoreOp(
        as_mlir_value(value),
        _lds_ptr(ctx, elem_idx),
        alignment=alignment,
        alias_scopes=dualwave._dualwave_lds_alias_scopes(scope),
        noalias_scopes=dualwave._dualwave_lds_noalias_scopes(scope, ctx.traits.LDS_SCOPE_NAMES),
    )


def _kv_tile_elem(traits, tile_elems, kv):
    """Element offset of runtime KV row `kv` in a region cut into `BLOCK_Q`-token tiles."""
    bq = fx.Index(traits.BLOCK_Q)
    return (kv // bq) * fx.Index(tile_elems) + tok_off_dyn(traits, kv % bq)


def _d_elem_dyn(traits, d):
    """`lds_elem(traits, 0, d)` for a runtime (uniform) column `d`."""
    g = fx.Index(traits.D_128B_SIZE)
    return (d // g) * fx.Index(traits.SMEM_N_RPT * traits.SMEM_V_LINE_STRIDE) + (d % g)


# --- context ---------------------------------------------------------------------


class BwdFusedKernelContext(BwdDkDvKernelContext):
    """The dK/dV context plus O, dQ, and the three extra LDS regions.

    `strides` carries 24 slots: the dK/dV kernel's 18 (Q, K, V, dO, dK, dV)
    then O and dQ. O and dQ are both Q-row-shaped and move with the query head,
    so they join the per-head rebinding in `bind_q_head`; nothing else about
    the GQA walk changes.

    **There is no Delta tensor.** The dK/dV base builds a delta resource from
    one; it is handed LSE in that slot so the dead descriptor folds away, and
    nothing reads `delta_rsrc` -- the tile body reads LDS.
    """

    def __init__(self, traits, *, strides, OUT, DQ, LSE, **kwargs):
        super().__init__(traits, strides=strides[:18], Delta=LSE, LSE=LSE, **kwargs)
        self.OUT = OUT
        self.DQ = DQ
        (
            self.stride_out_batch,
            self.stride_out_head,
            self.stride_out_seq,
            self.stride_dq_batch,
            self.stride_dq_head,
            self.stride_dq_seq,
        ) = strides[18:24]

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.stream.ptr))
        self.lds_kv_base_ptr = lds.stream.ptr.llvm_ptr
        self.lds_bt_base_ptr = None

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_out_seq_v = fx.Index(self.stride_out_seq)
        self.stride_dq_seq_v = fx.Index(self.stride_dq_seq)

    def _init_q_head_invariants(self):
        super()._init_q_head_invariants()
        elem_bytes = fx.Index(self.traits.BF16_BYTES)
        self.out_span_bytes = _slab_span_elems(self.seqlen_q_v, self.stride_out_seq, self.hdim_vo) * elem_bytes
        self.dq_span_bytes = _slab_span_elems(self.seqlen_q_v, self.stride_dq_seq, self.hdim_qk) * elem_bytes
        # **The O and dQ bases are not hoisted here**, unlike the dK/dV base's
        # Q/dO/LSE ones: `bind_q_head` derives them from the head index and the
        # kernarg strides each time. Hoisting keeps a base and a per-head step
        # live across the whole GQA walk -- eight more SGPRs -- and this
        # kernel sits at the SGPR cliff. Measured: head_dim 100 (rung 128,
        # padded), causal, f16, `block_kv` 64 faulted with the hoisted form
        # (128 SGPR spills, the lost-SGPR-copy signature: a kernarg block
        # loaded with `s_load_dwordx16` and only two of its four pairs copied
        # before the staging range is spilled) and passes without it (119);
        # so did four of seven failing dropout builds. Spill pressure is not
        # the defect's control variable (`HANDOFF-gfx950-lost-sgpr-copy.md`),
        # so this lowers the odds rather than closing it.
        # dO again, as a raw resource for the delta prologue's vector loads.
        self.do_span_bytes = self.do_slab_span_elems * elem_bytes

    def bind_q_head(self, g):
        super().bind_q_head(g)
        self.out_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.OUT))),
            self._slab_byte_base(
                self.stride_out_batch,
                self.stride_out_head,
                self.stride_out_seq,
                self.q_row_off,
                self.q_head_idx,
            ),
            self.out_span_bytes,
        )
        self.dq_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.DQ))),
            self._slab_byte_base(
                self.stride_dq_batch,
                self.stride_dq_head,
                self.stride_dq_seq,
                self.q_row_off,
                self.q_head_idx,
            ),
            self.dq_span_bytes,
        )
        self.do_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.DO))),
            self.do_slab_base0_bytes + g * self.do_head_step_bytes,
            self.do_span_bytes,
        )
        # The raw LSE resource the dK/dV base just built, under a name that
        # says it is the global one; the body reads the LDS copy.
        self.lse_g_rsrc = self.lse_rsrc

    def init_tile_bounds(self, **kwargs):
        """The Q tiles this workgroup walks: `[split_t0, split_t_end)`, **not** rounded.

        The dK/dV kernel rounds its walk up to an even count because its body
        consumes two tiles per iteration; at the sequence lengths this kernel
        exists for that rounding is the whole cost -- `seqlen_q` 64 is one tile,
        and a second, all-padding tile doubled the body's work. Here the loop
        takes pairs and a single tile finishes an odd count (see `_fused_body`),
        so the walk is exact. The causal cut is the dK/dV kernel's, verbatim
        but for the rounding.
        """
        traits = self.traits
        bq = fx.Index(traits.BLOCK_Q)
        self.kv_tile_size = traits.BLOCK_Q
        self.causal_end_raw_i32 = None
        self.q_tiles_real = (self.seqlen_q_v + bq - fx.Index(1)) // bq
        if const_expr(not traits.CAUSAL):
            self.window_left_i32 = None
            self.causal_lo_i32 = None
            self.split_t0 = fx.Index(0)
            self.split_t_end = self.q_tiles_real
        else:
            # `varlen_bits` carries `principal_diagonal_choice` (bit 18), which
            # moves a *numeric* bound to this sequence's bottom-right diagonal,
            # as in the dK/dV kernel this cut is taken from.
            left, right = fmha.resolve_window(
                self.window_left_arg,
                self.window_right_arg,
                fx.Int32(self.seqlen_q_v),
                self.seqlen_kv_i32,
                self.varlen_bits_arg,
            )
            self.window_left_i32 = fx.Int32(left)
            self.causal_lo_i32 = fx.Int32(fx.Int32(0) - fx.Int32(right))
            regions = fmha.decompose_causal_regions(
                fx.Int32(self.kv_start),
                self.seqlen_kv_i32,
                fx.Int32(self.seqlen_q_v),
                fx.Int32(right),
                fx.Int32(left),
                traits.BLOCK_KV,
                traits.BLOCK_Q,
                fx.Boolean(True),
            )
            n_tiles = regions.n_left + regions.n_full + regions.n_right
            n_tiles = fx.Index((n_tiles > fx.Int32(0)).select(n_tiles, fx.Int32(0)))
            self.split_t0 = fx.Index(regions.left_col0) // bq
            self.split_t_end = self.split_t0 + n_tiles
        self.num_q_tiles = self.split_t_end - self.split_t0
        self.max_num_tiles = self.num_q_tiles
        # The walk in whole pairs, and whether one tile is left over.
        self.pair_t_end = self.split_t0 + (self.num_q_tiles // fx.Index(2)) * fx.Index(2)
        self.has_tail = self.pair_t_end < self.split_t_end


# --- the row prologue: -log2e*LSE and delta into LDS -----------------------------


class FusedRowPrologue(dualwave.DualwaveKernelContext):
    """Fill the two f32 row arrays for the current query head.

    `lanes_per_row` lanes per q row, eight d each, so a wave covers
    `64 / lanes` rows with one fully coalesced 128-bit load per tensor; the row
    sum is a `shuffle_xor` butterfly over those lanes. Rows past `seqlen_q`
    read zero through the descriptor bound and store `delta = 0`, `LSE = 0`,
    which is the dK/dV kernel's padding-row argument: `P = 1`, times a zero dO
    row, contributes nothing.

    Columns past the real head dim are discarded before the product, both of
    them: the 8xD contract leaves the pad's contents to the caller, and a NaN
    there times a masked zero is still NaN.
    """

    def init_rows(self):
        """Per-lane constants of the prologue. Loop-invariant across heads."""
        traits = self.traits
        ctx = self.ctx_ref
        self.lanes = lanes_per_row(traits.HEAD_DIM)
        self.rpp = rows_per_pass(traits.NUM_WAVES, traits.HEAD_DIM)
        # Rows one pass covers: at least a tile, so a lane issues every load
        # of its rows before waiting on any -- `reps` rows per lane, unrolled.
        self.pass_rows = max(traits.BLOCK_Q, self.rpp)
        self.reps = self.pass_rows // self.rpp
        self.row_in_pass = self.tid // fx.Index(self.lanes)
        self.col = (self.tid % fx.Index(self.lanes)) * fx.Index(8)
        self.need_mask = self.PADDED_HEAD or self.lanes * 8 != traits.HEAD_DIM
        self.rows_base = fx.Index(traits.ROWS_REGION_BASE)
        self.delta_base = self.rows_base + fx.Index(2 * traits.Q_ROWS_CAP)
        self.neg_log2e = fx.Float32(-dualwave._LOG2E)
        self.hdim_vo_v = ctx.hdim_vo

    def pass_bounds(self, row_lo, row_hi):
        """Passes covering rows `[row_lo, row_hi)`, rounded out to whole passes."""
        pr = fx.Index(self.pass_rows)
        return row_lo // pr, (row_hi + pr - fx.Index(1)) // pr

    def one_pass(self, p):
        """Rows `p * pass_rows .. + pass_rows` of the current query head.

        Every load of the pass is issued before the first is consumed: the
        prologue is a latency chain on the critical path of every head, and at
        the sequence lengths this kernel serves it is most of a workgroup's
        life.
        """
        traits = self.traits
        ctx = self.ctx_ref
        fm = self.fm_fast
        col = self.col
        rows = [p * fx.Index(self.pass_rows) + fx.Index(r * self.rpp) + self.row_in_pass for r in range(self.reps)]
        loads = []
        for row in rows:
            o = buffer_ops.buffer_load(
                ctx.out_rsrc,
                as_mlir_value(fx.Int32(row * ctx.stride_out_seq_v + col)),
                vec_width=8,
                dtype=self.elem_dtype,
            )
            do = buffer_ops.buffer_load(
                ctx.do_rsrc,
                as_mlir_value(fx.Int32(row * ctx.stride_do_seq_v + col)),
                vec_width=8,
                dtype=self.elem_dtype,
            )
            lse = buffer_ops.buffer_load(
                ctx.lse_g_rsrc,
                as_mlir_value(fx.Int32(row * ctx.lse_pitch)),
                vec_width=1,
                dtype=fx.Float32,
            )
            loads.append((o, do, lse))
        width = fx.Int32(traits.WARP_SIZE)
        for row, (o, do, lse) in zip(rows, loads):
            if const_expr(self.need_mask):
                cols = MaskedAxis(fx.Index(self.hdim_vo_v), elem_dtype=self.elem_dtype, bitmask=True)
                o = cols.discard(o, col, 8)
                do = cols.discard(do, col, 8)
            prod = Vec(o).to(fx.Float32) * Vec(do).to(fx.Float32)
            part = fx.Float32(prod.reduce("add", init_val=fx.Float32(0.0), fastmath=fm))
            for step in range_constexpr(self.lanes.bit_length() - 1):
                other = part.shuffle_xor(fx.Int32(self.lanes >> (step + 1)), width)
                part = fx.Float32(part + other)
            nlse = fx.Float32(fx.Float32(lse) * self.neg_log2e)
            # Every lane of the row writes the same two values to the same two
            # addresses: no branch, and nothing narrows EXEC.
            _lds_store(self, nlse, self.rows_base + row * fx.Index(2), SCOPE_ROWS, 4)
            _lds_store(self, part, self.delta_base + row * fx.Index(2), SCOPE_ROWS, 4)


# --- dQ for one tile ---------------------------------------------------------------


class FusedDqHelper(dualwave.DualwaveKernelContext):
    """`dQ_tile = sm_scale * dS . K`, from the two LDS regions, stored at once.

    The tile's output is `BLOCK_Q x head_dim`, cut into 16x16 blocks indexed by
    (q block, d chunk). A wave takes whole q blocks while there are enough to
    go round, and otherwise one q block and a strided share of its chunks --
    so the B operand (dS) is read once per wave and reused across the wave's
    chunks, and the A operand (K^T) once per (chunk, k-step).
    """

    def init_dq(self):
        traits = self.traits
        waves = traits.NUM_WAVES
        nqb = traits.BLOCK_Q // MFMA16_M
        nc = traits.HEAD_DIM // MFMA16_M
        wave = self.wave_id_uni
        if waves <= nqb:
            if nqb % waves:
                raise ValueError(f"{nqb} q blocks do not divide across {waves} waves")
            self.nq_w = nqb // waves
            self.nc_w = nc
            self.qb0 = wave * fx.Index(self.nq_w)
            self.chunk_stride = 1
            self.c0 = fx.Index(0)
        else:
            split = waves // nqb
            if waves % nqb or nc % split:
                raise ValueError(f"{waves} waves cannot share {nqb} q blocks x {nc} d chunks evenly")
            self.nq_w = 1
            self.nc_w = nc // split
            self.qb0 = wave % fx.Index(nqb)
            self.chunk_stride = split
            self.c0 = wave // fx.Index(nqb)
        self.stores_per_tile = self.nq_w * self.nc_w
        self.atom16 = fx.make_mma_atom(fx.rocdl.MFMA(MFMA16_M, MFMA16_M, _K32, self.elem_dtype))
        self.acc16_ty = Vec.make_type(_ACC, fx.Float32)
        self.zero16 = Vec.filled(_ACC, 0.0, fx.Float32).ir_value()
        lane_base = a16_read_base(traits, self.lane, _LANE_K)
        k_region = fx.Index(traits.K_REGION_BASE) + lane_base
        ds_region = fx.Index(traits.DS_REGION_BASE) + lane_base
        # Byte addresses, one per (chunk) and per (q block); wave-uniform terms
        # folded in once here so the reads below carry only immediates.
        self.k_addr = [
            fx.Int32((k_region + _d_elem_dyn(traits, self._chunk(k) * fx.Index(MFMA16_M))) * fx.Index(2))
            + fx.Int32(self.lds_kv_base_idx)
            for k in range(self.nc_w)
        ]
        self.ds_addr = [
            fx.Int32((ds_region + _d_elem_dyn(traits, (self.qb0 + fx.Index(i)) * fx.Index(MFMA16_M))) * fx.Index(2))
            + fx.Int32(self.lds_kv_base_idx)
            for i in range(self.nq_w)
        ]
        self.store16 = m16.M16StoreHelper(self)

    def _chunk(self, k):
        return self.c0 + fx.Index(k * self.chunk_stride)

    def _tr_pack(self, addr, imm, scope):
        traits = self.traits
        pair = tok_off(traits, 4) * traits.BF16_BYTES
        lo = _ds_read_tr16_b64_imm(
            self.v_lds_read_vec4_type,
            addr,
            imm,
            scope_name=scope,
            scope_names=traits.LDS_SCOPE_NAMES,
        )
        hi = _ds_read_tr16_b64_imm(
            self.v_lds_read_vec4_type,
            addr,
            imm + pair,
            scope_name=scope,
            scope_names=traits.LDS_SCOPE_NAMES,
        )
        return Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

    def _mfma(self, a, b, c):
        return dualwave.fly.mma_atom_call_ssa([self.acc16_ty], self.atom16, a, b, c)

    def compute(self):
        """Per q block of this wave: `nc_w` accumulators of 4 f32."""
        traits = self.traits
        steps = traits.BLOCK_KV // _K32
        out = []
        for i in range_constexpr(self.nq_w):
            accs = [self.zero16] * self.nc_w
            for s in range_constexpr(steps):
                kt, within = divmod(s * _K32, traits.BLOCK_Q)
                tok = tok_off(traits, within)
                b = self._tr_pack(self.ds_addr[i], (kt * traits.DS_TILE_ELEMS + tok) * traits.BF16_BYTES, SCOPE_DS)
                for k in range_constexpr(self.nc_w):
                    a = self._tr_pack(
                        self.k_addr[k],
                        (kt * traits.STREAM_TILE_ELEMS + tok) * traits.BF16_BYTES,
                        SCOPE_K,
                    )
                    accs[k] = self._mfma(a, b, accs[k])
            out.append(accs)
        return out

    def store(self, tile_idx, blocks):
        """Store `blocks[i][k]` (or zeros, for `blocks=None`) for tile `tile_idx`."""
        traits = self.traits
        ctx = self.ctx_ref
        lane16 = self.lane % fx.Index(MFMA16_M)
        lane_d = (self.lane // fx.Index(MFMA16_M)) * fx.Index(_ACC)
        cols = MaskedAxis(ctx.hdim_qk) if const_expr(self.PADDED_HEAD) else None
        scale = ctx.c_sm_scale
        zero = fx.Float32(0.0)
        for i in range_constexpr(self.nq_w):
            q_row = tile_idx * fx.Index(traits.BLOCK_Q) + (self.qb0 + fx.Index(i)) * fx.Index(MFMA16_M) + lane16
            base = q_row * ctx.stride_dq_seq_v
            for k in range_constexpr(self.nc_w):
                col = self._chunk(k) * fx.Index(MFMA16_M) + lane_d
                if const_expr(blocks is None):
                    vals = [zero] * _ACC
                else:
                    acc = Vec(blocks[i][k])
                    vals = [acc[r] * scale for r in range_constexpr(_ACC)]
                packed = self.store16._pack_out(vals)
                off = base + col
                if const_expr(self.PADDED_HEAD):
                    off = fx.Index(cols.valid(col).select(fx.Index(off), fx.Index(_OOB_ELEM)))
                buffer_ops.buffer_store(packed.ir_value(), ctx.dq_rsrc, as_mlir_value(fx.Int32(off)))


# --- dK and dV out: the epilogue's addressing, parked in VGPRs ------------------------

# FIXME: compiler workaround. Everything in this section exists only because the
# gfx950 backend re-spills a reloaded SGPR range with stale values: FlyDSL#1344
# (greedy-allocator live-range splitting of a partially defined 512-bit argument
# tuple). Reproduces with FlyDSL 0.4.2 (LLVM 941a04e); not fixed by
# llvm/llvm-project#223465. Remove
# `FusedKvOutputs`, `_park_in_vgprs` and `PARK_OUTPUTS` (falling back to the dK/dV
# helpers' `store_accs`) once the pinned LLVM carries a fix, and re-run the
# 112-build dropout/bias matrix to prove it.


def _park_in_vgprs(values):
    """Each value out through one side-effecting no-op asm, in a VGPR.

    The asm emits nothing (every output is tied to its input), but nothing can
    see through it: a later use reads the asm's result, so the values are
    computed *here* and carried in VGPRs, and no pass can sink their
    computation -- or the kernargs it reads -- down to the use. Side-effecting
    so that the asm itself is not sunk either; it runs once per workgroup,
    outside every loop, so the scheduling boundary costs nothing.
    """
    irs = [as_mlir_value(v) for v in values]
    n = len(irs)
    if const_expr(n == 1):
        return [dualwave.llvm.inline_asm(irs[0].type, irs, "", "=v,0", has_side_effects=True)]
    ret_ty = dualwave.ir.Type.parse(f"!llvm.struct<({', '.join(str(v.type) for v in irs)})>")
    constraints = ",".join(["=v"] * n + [str(i) for i in range(n)])
    ret = dualwave.llvm.inline_asm(ret_ty, irs, "", constraints, has_side_effects=True)
    return [dualwave.llvm.extractvalue(irs[i].type, ret, [i]) for i in range(n)]


def _unpark_uniform(v):
    """A parked wave-uniform i32 back in an SGPR, for a descriptor."""
    return fx.Int32(dualwave.rocdl.readfirstlane(T.i32, as_mlir_value(v)))


class FusedKvOutputs(dualwave.DualwaveKernelContext):
    """dK and dV, stored by the epilogue from addressing fixed in the prologue.

    **This is a workaround for a backend defect, not a change of algorithm.**
    The stores, the packing and the bounds are the dK/dV kernels' own
    (`BwdDkDvStoreHelper.store_accs`, `M16StoreHelper.store_accs`); what moves
    is *when* their inputs are formed and *where* they live in between.

    In the dK/dV helpers every input of the epilogue's addressing -- the DK and
    DV pointers, their batch/head/row strides, the scale factors -- is a
    kernarg that stays live in SGPRs from the prologue to the epilogue, across
    the whole GQA and tile walk. At the SGPR cliff those values are spilled to
    VGPR lanes for the duration, and the gfx950 backend's lost-SGPR-copy
    defect (`HANDOFF-gfx950-lost-sgpr-copy.md`, FlyDSL#1087) corrupts such
    spills. Measured on head_dim 32, `BLOCK_KV` 128, causal, bf16, dropout
    (261 SGPR spills): the 512-bit kernarg block `s_load_dwordx16 0x148`
    (dK head/row, dV batch/head/row, dQ strides) is reloaded from its spill
    lanes, only its three dQ pairs are copied into the register range it is
    re-spilled from, and the other ten lanes are overwritten with stale SGPRs.
    The epilogue then multiplies the KV row by a "row stride" of
    `4 * seqlen_q`: 90% of dK and dV is never written and the rest lands on
    the wrong rows -- with dQ, which reads its strides per head, exactly right.
    LLVM PR 223465 does not cover it; it is a wrong value rather than an
    undefined one, so the lost-copy detector does not see it either.

    So nothing the epilogue needs is left in an SGPR across the walk. The
    prologue forms each output's descriptor fields and per-lane row offset,
    and the two scales, and `_park_in_vgprs` holds them: ten VGPRs, one asm,
    once per workgroup. The epilogue `readfirstlane`s the descriptor fields
    back and uses the rest as they are. A VGPR is never spilled through an
    SGPR copy, so whatever the allocator does with the SGPRs, the outputs
    land where the source says. That is what makes this a structural fix
    rather than a re-roll of register allocation: it does not lower the odds
    of the defect, it removes these values from its reach.

    **Dropout builds only.** The ten VGPRs are free there -- the policy budgets
    dropout 50 more registers per lane, and the builds measured kept 0 VGPR
    spills while their SGPR spills halved (head_dim 32 `block_kv` 64 causal:
    170 -> 80; `block_kv` 128: 261 -> 176). A non-dropout build is held to its
    occupancy's VGPR budget instead: head_dim 32 `block_kv` 64 is capped at 80
    VGPRs, and parking there took its scratch spills from 4 to 19 dwords and
    the kernel from 832 to 1038 us (causal: 883 to 1318). No non-dropout build
    has shown the defect, and the matrix is what would show one. The dropout
    builds that were already at 256 VGPRs pay something: head_dim 32
    `block_kv` 128 f16 went from 2 to 10 VGPR spills (non-causal, 1706 -> 1640
    us, faster on the SGPR side) and from 9 to 23 (causal, 1796 -> 2013 us);
    the rest measured ran 0-14% faster, from the halved SGPR spills.
    """

    def init_outputs(self, pack16=None):
        """Form and park everything the epilogue reads. Call in the prologue.

        `pack16` is the 16-row family's `_pack_out`; the 32-row family packs
        through the forward's O path, as its dK/dV kernel does.
        """
        ctx = self.ctx_ref
        traits = self.traits
        self.M16 = traits.MFMA_ROWS == 16
        self.pack16 = pack16
        ebytes = fx.Index(traits.BF16_BYTES)
        col0 = fx.Index(0) if const_expr(self.M16) else ctx.dkv_col_base
        fields = []
        for tensor, s0, s1, s2, hdim in (
            (ctx.DK, ctx.stride_dk_batch, ctx.stride_dk_head, ctx.stride_dk_seq, ctx.hdim_qk),
            (ctx.DV, ctx.stride_dv_batch, ctx.stride_dv_head, ctx.stride_dv_seq, ctx.hdim_vo),
        ):
            # `BwdDkDvKernelContext._slab_rsrc`'s base and bound, as numbers.
            addr = fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) + fx.Int64(
                ctx._slab_byte_base(s0, s1, s2, ctx.kv_row_off, ctx.kv_head_idx, batch_idx=ctx.kv_batch_idx)
            )
            span = _slab_span_elems(ctx.seqlen_kv_v, s2, hdim) * ebytes
            fields += [
                fx.Int32(addr),
                fx.Int32(addr.shrui(fx.Int64(32))),
                fx.Int32(span),
                fx.Int32(ctx.kv_row * fx.Index(s2) + col0),
            ]
        fields += [ctx.c_sm_scale, ctx.c_dropout_scale]
        if const_expr(self.PADDED_HEAD):
            fields += [fx.Int32(ctx.hdim_qk), fx.Int32(ctx.hdim_vo)]
        parked = _park_in_vgprs(fields)
        self.parked = {"dk": parked[0:4], "dv": parked[4:8]}
        self.sm_scale_v = fx.Float32(parked[8])
        self.dropout_scale_v = fx.Float32(parked[9])
        if const_expr(self.PADDED_HEAD):
            self.hdim_v = {"dk": fx.Int32(parked[10]), "dv": fx.Int32(parked[11])}

    def _rsrc(self, which):
        lo, hi, span, _ = self.parked[which]
        addr = (fx.Int64(_unpark_uniform(hi)) << fx.Int64(32)) | (
            fx.Int64(_unpark_uniform(lo)) & fx.Int64(0xFFFFFFFF)
        )
        nrec = fx.Int64(_unpark_uniform(span)) & fx.Int64(0xFFFFFFFF)
        return buffer_ops.create_buffer_resource_from_addr(as_mlir_value(addr), num_records_bytes=as_mlir_value(nrec))

    def _offset(self, which, row_off, col):
        """Element offset of column `col`; past the descriptor if `col` is pad."""
        off = row_off + fx.Int32(col)
        if const_expr(self.PADDED_HEAD):
            off = (fx.Int32(col) < self.hdim_v[which]).select(off, fx.Int32(_OOB_ELEM))
        return off

    def scale(self, dv, dk, acc_width):
        """`dK *= sm_scale`; `dV *= dropout_scale` in a dropout build."""
        sm = Vec.from_elements([self.sm_scale_v], fx.Float32).broadcast_to(acc_width)
        dk = [Vec(a) * sm for a in dk]
        if const_expr(self.traits.ENABLE_DROPOUT):
            drop = Vec.from_elements([self.dropout_scale_v], fx.Float32).broadcast_to(acc_width)
            dv = [Vec(a) * drop for a in dv]
        return dv, dk

    def store(self, accs, which):
        traits = self.traits
        rsrc = self._rsrc(which)
        row_off = fx.Int32(self.parked[which][3])
        if const_expr(self.M16):
            # `M16StoreHelper.store_accs`: four contiguous d per lane per chunk.
            lane_d = (self.lane // fx.Index(MFMA16_M)) * fx.Index(m16.ACC16)
            for c in range_constexpr(len(accs)):
                vals = [Vec(accs[c])[i] for i in range_constexpr(m16.ACC16)]
                off = self._offset(which, row_off, fx.Index(c * MFMA16_M) + lane_d)
                buffer_ops.buffer_store(self.pack16(vals).ir_value(), rsrc, as_mlir_value(off))
        else:
            # `BwdDkDvStoreHelper.store_accs`: eight contiguous d per lane per
            # half-chunk, after the forward's `permlane32_swap` packing. The
            # shard's column origin is already in `row_off`; `col` is the
            # global column, which is what the pad test needs.
            lane_d = self.lane_div_32 * fx.Index(8)
            for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
                for g in range_constexpr(2):
                    local = fx.Index(dc * traits.D_CHUNK + 2 * g * 8) + lane_d
                    off = row_off + fx.Int32(local)
                    if const_expr(self.PADDED_HEAD):
                        col = self.ctx_ref.dkv_col_base + local
                        off = (fx.Int32(col) < self.hdim_v[which]).select(off, fx.Int32(_OOB_ELEM))
                    words = dualwave._packed_o_128_vec(traits, accs, dc, g, self.lane_div_32, self.elem_dtype)
                    buffer_ops.buffer_store(words.bitcast(self.elem_dtype).ir_value(), rsrc, as_mlir_value(off))


# --- the 16-row tile body, with dS out and dQ after ---------------------------------


class FusedM16TileBody(m16.M16TileBody):
    """`M16TileBody.run`, plus the dS write and the tile's dQ.

    The body proper -- GEMMs 1-4, the softmax, the masks, dropout and bias --
    is the 16-row dK/dV body line for line; the differences are marked. It is
    restated rather than wrapped because three of them land *inside* it: the
    top wait, the row reads and the dS write.
    """

    def __init__(self, ctx, *, dq, **kwargs):
        super().__init__(ctx, **kwargs)
        self.dq = dq

    def _row_values(self, which, tile_base, sub):
        """Four f32 for this sub-block's four contiguous q rows, from LDS.

        `which` is 0 for `-log2e*LSE`, 1 for delta. The 16-row family's rows
        are `(sub//2)*32 + (sub%2)*4 + 8*(lane//16) + 0..3` -- contiguous and
        16-byte aligned -- so this is one `ds_read_b128`.
        """
        traits = self.traits
        row0 = (
            tile_base
            + fx.Index((sub // 2) * _K32 + (sub % 2) * 4)
            + (self.lane // fx.Index(MFMA16_M)) * fx.Index(8)
        )
        elem = fx.Index(traits.ROWS_REGION_BASE + which * 2 * traits.Q_ROWS_CAP) + row0 * fx.Index(2)
        v = _lds_load(self, Vec.make_type(_ACC, fx.Float32), elem, SCOPE_ROWS, 16)
        return [Vec(v)[r] for r in range_constexpr(_ACC)]

    def _write_ds(self, ds_pack, g):
        """dS for this lane's KV column and eight q rows of group `g`, into dS^T."""
        traits = self.traits
        ctx = self.ctx_ref
        kv = ctx.kv_row_in_block
        q0 = lds_elem(traits, 0, g * _K32)
        elem = (
            fx.Index(traits.DS_REGION_BASE)
            + _kv_tile_elem(traits, traits.DS_TILE_ELEMS, kv)
            + fx.Index(q0)
            + (self.lane // fx.Index(MFMA16_M)) * fx.Index(_LANE_K)
        )
        # A KV row past `seqlen_k` has a zero K row, so its dS is multiplied by
        # zero -- unless dS is not finite. It is `P * (0 - delta)` with
        # `P = exp2(-lse*log2e)`, unbounded for a row whose scores are all very
        # negative; zeroing it here costs four selects and removes the case.
        raw = Vec(ds_pack).bitcast(fx.Int32)
        live = fx.Int32(ctx.kv_row) < fx.Int32(ctx.seqlen_kv_i32)
        zero = fx.Int32(0)
        words = [live.select(fx.Int32(raw[i]), zero) for i in range_constexpr(4)]
        masked = Vec.from_elements([as_mlir_value(w) for w in words], fx.Int32)
        _lds_store(self, masked.bitcast(self.elem_dtype), elem, SCOPE_DS, 16)

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = dualwave._dualwave_lds_scope("k", buf_id)
        do_scope = dualwave._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)
        n_grp = m16.q_groups(traits)
        n_chunks = m16.d_chunks16(traits)

        # FUSED: the DMA for this tile is followed by its own tile's dQ stores
        # and by `NBUF - 1` more (DMA, stores) tile ends. See the module
        # docstring's `vmcnt` budget.
        nd = ctx.NUM_DMA_K + ctx.NUM_DMA_V
        st = self.dq.stores_per_tile
        _waitcnt_vm_n(st + (traits.NUM_STREAM_BUFFERS - 1) * (nd + st))
        _sched_barrier(0)
        _s_barrier()

        p_packs = [None] * n_grp
        ds_packs = [None] * n_grp
        for g in range_constexpr(n_grp):
            p_half = []
            ds_half = []
            for h in range_constexpr(2):
                sub = 2 * g + h
                s = self.gemm.contract_d(self._reader_for(q_base, q_scope, sub, self.hdim_qk), v_k)
                dp = self.gemm.contract_d(self._reader_for(do_base, do_scope, sub, self.hdim_vo), v_v)
                # FUSED: from LDS, already scaled by -log2(e).
                neg_lse2 = self._row_values(0, tile_base, sub)
                bias2 = None
                if const_expr(traits.BIAS_TYPE):
                    bias2 = self.bias_fn(ctx, self.softmax.q_rows(tile_idx, sub))
                p_list = self.softmax.probabilities(s, neg_lse2, bias2)
                if const_expr(traits.CAUSAL):
                    p_list = self.softmax.mask_if_clipped(p_list, tile_idx, sub)
                # FUSED: from LDS, computed by the prologue.
                delta = self._row_values(1, tile_base, sub)
                keep = None
                if const_expr(traits.ENABLE_DROPOUT):
                    keep = self.keep_fn(ctx, self.softmax.q_rows(tile_idx, sub))
                ds_half.append(self.softmax.dscores(p_list, dp, delta, keep))
                if const_expr(traits.ENABLE_DROPOUT):
                    p_list = [keep[r].select(fx.Float32(p_list[r]), ctx.c_zero_f) for r in range_constexpr(m16.ACC16)]
                p_half.append(p_list)
            p_packs[g] = self.softmax.pack_group(p_half[0], p_half[1])
            ds_packs[g] = self.softmax.pack_group(ds_half[0], ds_half[1])
            # FUSED: publish this group's dS for the tile's dQ.
            self._write_ds(ds_packs[g], g)

        for c in range_constexpr(n_chunks):
            dv[c] = self.gemm.contract_q(self.reader.read_column_chunk(do_base, do_scope, c), p_packs, dv[c])
        for c in range_constexpr(n_chunks):
            dk[c] = self.gemm.contract_q(self.reader.read_column_chunk(q_base, q_scope, c), ds_packs, dk[c])

        _s_waitcnt(traits.LGKMCNT_0_ONLY)
        _sched_barrier(0)
        _s_barrier()
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)

        # FUSED: every wave's dS for this tile is in LDS; the tile's dQ is final.
        self.dq.store(tile_idx, self.dq.compute())
        return dv, dk


class Fused32TileBody(BwdDkDvTileBody):
    """`BwdDkDvTileBody.run` at 32 rows per wave, plus the dS write and the tile's dQ.

    The same three insertions as `FusedM16TileBody`, against the 32-row map:
    accumulator element `r` of half `h` holds q row
    `32h + 8*(r//4) + 4*(lane//32) + r%4` (`_ROW_RUNS`), so the row reads are
    four `ds_read_b128` per half and a packed dS operand -- `values[8p..8p+7]`
    -- is two runs of four contiguous q, `16p + 4*(lane//32) + {0, 8}` on, i.e.
    two 8-byte LDS writes.
    """

    def __init__(self, ctx, *, dq, **kwargs):
        super().__init__(ctx, **kwargs)
        self.dq = dq

    def _row_values(self, which, tile_base, half):
        traits = self.traits
        row_base = tile_base + fx.Index(half * 32) + self.lane_div_32 * fx.Index(4)
        region = fx.Index(traits.ROWS_REGION_BASE + which * 2 * traits.Q_ROWS_CAP)
        values = [None] * 16
        for elem0, col_off, width in _ROW_RUNS:
            elem = region + (row_base + fx.Index(col_off)) * fx.Index(2)
            v = _lds_load(self, Vec.make_type(width, fx.Float32), elem, SCOPE_ROWS, 16)
            for j in range_constexpr(width):
                values[elem0 + j] = Vec(v)[j]
        return values

    def _write_ds(self, ds_pack, half, p):
        traits = self.traits
        ctx = self.ctx_ref
        kv = ctx.kv_row_in_block
        row_elem = fx.Index(traits.DS_REGION_BASE) + _kv_tile_elem(traits, traits.DS_TILE_ELEMS, kv)
        raw = Vec(ds_pack).bitcast(fx.Int32)
        live = fx.Int32(ctx.kv_row) < fx.Int32(ctx.seqlen_kv_i32)
        zero = fx.Int32(0)
        words = [live.select(fx.Int32(raw[i]), zero) for i in range_constexpr(4)]
        for run in range_constexpr(2):
            q0 = half * 32 + 16 * p + 8 * run
            elem = row_elem + fx.Index(lds_elem(traits, 0, q0)) + self.lane_div_32 * fx.Index(4)
            pair = Vec.from_elements([as_mlir_value(w) for w in words[2 * run : 2 * run + 2]], fx.Int32)
            _lds_store(self, pair.bitcast(self.elem_dtype), elem, SCOPE_DS, 8)

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = dualwave._dualwave_lds_scope("k", buf_id)
        do_scope = dualwave._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)

        # FUSED: the `vmcnt` budget; see the module docstring.
        nd = ctx.NUM_DMA_K + ctx.NUM_DMA_V
        st = self.dq.stores_per_tile
        _waitcnt_vm_n(st + (traits.NUM_STREAM_BUFFERS - 1) * (nd + st))
        _sched_barrier(0)
        _s_barrier()

        p_packs = [None] * 4
        ds_packs = [None] * 4
        for group in self.half_groups:
            s = {h: self.gemm.contract_d(self._row_reader(q_base, q_scope, h, self.hdim_qk), v_k) for h in group}
            dp = {h: self.gemm.contract_d(self._row_reader(do_base, do_scope, h, self.hdim_vo), v_v) for h in group}
            # FUSED: from LDS, already scaled by -log2(e).
            neg_lse2 = {h: self._row_values(0, tile_base, h) for h in group}
            bias2 = {h: None for h in group}
            if const_expr(traits.BIAS_TYPE):
                bias2 = {h: bias_log2e(ctx, self.softmax.q_rows(tile_idx, h)) for h in group}
            p_list = {h: self.softmax.probabilities(s[h], neg_lse2[h], bias2[h]) for h in group}
            if const_expr(traits.CAUSAL):
                p_list = {h: self.softmax.mask_if_clipped(p_list[h], tile_idx, h) for h in group}
            delta = {h: self._row_values(1, tile_base, h) for h in group}
            keep = {h: None for h in group}
            if const_expr(traits.ENABLE_DROPOUT):
                keep = {h: philox_keep(ctx, self.softmax.q_rows(tile_idx, h)) for h in group}
            ds_list = {h: self.softmax.dscores(p_list[h], dp[h], delta[h], keep[h]) for h in group}
            if const_expr(traits.ENABLE_DROPOUT):
                p_list = {
                    h: [keep[h][r].select(fx.Float32(p_list[h][r]), ctx.c_zero_f) for r in range_constexpr(16)]
                    for h in group
                }
            for h in group:
                p_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(p_list[h])
                ds_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(ds_list[h])
                # FUSED: publish this half's dS for the tile's dQ.
                for pp in range_constexpr(2):
                    self._write_ds(ds_packs[2 * h + pp], h, pp)

        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            dv[dc] = self.gemm.contract_q(self.reader.read_column_chunk(do_base, do_scope, dc), p_packs, dv[dc])
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            dk[dc] = self.gemm.contract_q(self.reader.read_column_chunk(q_base, q_scope, dc), ds_packs, dk[dc])

        _s_waitcnt(traits.LGKMCNT_0_ONLY)
        _sched_barrier(0)
        _s_barrier()
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)

        # FUSED: the tile's dQ.
        self.dq.store(tile_idx, self.dq.compute())
        return dv, dk


def build_fmha_bwd_fused_gfx950_module_primary(meta, knobs):
    """Build the fused backward for a resolved (meta, knobs) pair."""
    if meta.block_dmodel is None or knobs.hdim_qk_floor is None:
        raise ValueError("meta and knobs must come from `tuning_fused_bwd.plan` or `resolve_knobs`")
    traits = knobs.build_traits(meta)
    # Which dK/dV family the body is: 16 rows per wave (`dkdv_m16`) or 32
    # (`dkdv`). The dQ GEMM is 16x16x32 either way.
    M16 = traits.MFMA_ROWS == 16
    TIGHT_REGISTERS = knobs.tight_registers
    BLOCK_DMODEL = meta.block_dmodel
    PADDED_HEAD = meta.padded_head
    HDIM_QK_FLOOR = knobs.hdim_qk_floor
    STRIDES_CONSTEXPR = knobs.strides_constexpr
    BUILD_SM_SCALE = meta.sm_scale
    NBUF = traits.NUM_STREAM_BUFFERS
    MASKED_STEPS = tuple(_masked_ks_steps(traits, HDIM_QK_FLOOR)) if PADDED_HEAD else ()
    _cache_tag = (
        traits_cache_key(traits),
        BLOCK_DMODEL,
        PADDED_HEAD,
        HDIM_QK_FLOOR,
        STRIDES_CONSTEXPR,
        BUILD_SM_SCALE,
        TIGHT_REGISTERS,
    )

    _lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        # One allocation: stream, K, dS^T, rows. See `BwdFusedTraits`.
        stream: fx.Array[_lds_elem_dtype, traits.LDS_TOTAL_ELEMS, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_fused_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        Out: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
        LSE: fx.Pointer,
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
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
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
        Q = wire_view(Q)
        K = wire_view(K)
        V = wire_view(V)
        B = wire_view(B)
        Out = wire_view(Out)
        DO = wire_view(DO)
        DK = wire_view(DK)
        DV = wire_view(DV)
        DQ = wire_view(DQ)
        LSE = wire_view(LSE)
        ctx = BwdFusedKernelContext(
            traits,
            strides=(
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
                stride_o_batch,
                stride_o_head,
                stride_o_seq,
                stride_dq_batch,
                stride_dq_head,
                stride_dq_seq,
            ),
            OUT=Out,
            DQ=DQ,
            LSE=LSE,
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            padded_head=PADDED_HEAD,
            hdim_qk_floor=HDIM_QK_FLOOR,
            window_left=window_left,
            window_right=window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Bias=B,
            bias_strides=(stride_b_batch, stride_b_head, stride_b_seq_q),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, None, None),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            Q=Q,
            K=K,
            V=V,
            DO=DO,
            DK=DK,
            DV=DV,
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=0,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_kv_row()
        ctx.init_philox()
        ctx.c_neg_log2e = fx.Float32(-dualwave._LOG2E)
        ctx.c_one_f = fx.Float32(1.0)
        ctx.c_dropout_scale = fx.Float32(dropout_scale)

        stream = BwdDkDvStreamLoader(ctx)
        outputs = FusedKvOutputs(ctx)
        dq = FusedDqHelper(ctx)
        dq.init_dq()
        rows = FusedRowPrologue(ctx)
        rows.init_rows()
        if const_expr(M16):
            reader = m16.M16StreamReader(ctx)
            reader.init16(HDIM_QK_FLOOR)
            resident = m16.M16ResidentLoader(ctx)
            gemm = m16.M16GemmHelper(ctx)
            gemm.init16()
            store = m16.M16StoreHelper(ctx)
            softmax = m16.M16SoftmaxHelper(ctx)
            softmax.LSE_STRIDED = False
            tile = FusedM16TileBody(
                ctx,
                dq=dq,
                stream=stream,
                reader=reader,
                gemm=gemm,
                softmax=softmax,
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
                keep_fn=philox_keep,
                bias_fn=bias_log2e,
            )
            n_acc = m16.d_chunks16(traits)
            zero_acc = Vec.filled(m16.ACC16, 0.0, fx.Float32).ir_value()
            acc_width = m16.ACC16
        else:
            reader = BwdDkDvStreamReader(ctx)
            reader.masked_steps = MASKED_STEPS
            resident = BwdDkDvResidentLoader(ctx)
            gemm = BwdDkDvGemmHelper(ctx)
            store = BwdDkDvStoreHelper(ctx)
            softmax = BwdDkDvSoftmaxHelper(ctx)
            softmax.LSE_STRIDED = False
            tile = Fused32TileBody(
                ctx,
                dq=dq,
                stream=stream,
                reader=reader,
                gemm=gemm,
                softmax=softmax,
                hdim_qk=hdim_qk,
                hdim_vo=hdim_vo,
                tight_registers=TIGHT_REGISTERS,
            )
            n_acc = traits.D_CHUNKS_PER_SHARD
            zero_acc = ctx.c_zero_v16f32
            acc_width = 16

        # The 16-row family's output packing; the 32-row family packs inline.
        pack16 = store._pack_out if M16 else None
        # Dropout builds only; see `FusedKvOutputs` for why, and why not the rest.
        # FIXME: compiler workaround, see the FIXME above `_park_in_vgprs`. It
        # protects only the dK/dV epilogue: scalars that stay live across the walk
        # in SGPRs (dQ strides, philox state, loop bounds) are still exposed to the
        # same defect. Costs +12% on hd32 kv128 causal f16 dropout (VGPR-bound).
        PARK_OUTPUTS = bool(traits.ENABLE_DROPOUT)
        t0 = ctx.split_t0
        t_end = ctx.split_t_end
        bq = fx.Index(traits.BLOCK_Q)
        if const_expr(not PARK_OUTPUTS):
            scale_vec = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(acc_width)
        # Rows the prologue fills: the walk's, rounded to whole passes inside it.
        p_lo, p_hi = rows.pass_bounds(t0 * bq, t_end * bq)

        def _stage_k_tile(v_k):
            """This wave's resident K rows into the K region: one 128-bit write per pack.

            Both families hold eight contiguous d of one KV row per pack; they
            differ in which eight. 16 rows: `32*ks + 8*(lane//16)`; 32 rows:
            `16*ks + 8*(lane//32)`.
            """
            kv = ctx.kv_row_in_block
            row_elem = fx.Index(traits.K_REGION_BASE) + _kv_tile_elem(traits, traits.STREAM_TILE_ELEMS, kv)
            if const_expr(M16):
                lane_d = (ctx.lane // fx.Index(MFMA16_M)) * fx.Index(_LANE_K)
                step = _K32
            else:
                lane_d = ctx.lane_div_32 * fx.Index(_LANE_K)
                step = traits.K_STEP_QK
            for ks in range_constexpr(len(v_k)):
                elem = row_elem + fx.Index(lds_elem(traits, 0, ks * step)) + lane_d
                _lds_store(ctx, v_k[ks], elem, SCOPE_K, 16)

        @flyc.jit
        def _fill_rows(p_lo, p_hi):
            for p, carry in range(p_lo, p_hi, fx.Index(1), init=[fx.Int32(0)]):
                rows.one_pass(p)
                carry = yield [carry[0]]

        @flyc.jit
        def _zero_dq_tiles(t_lo, t_hi):
            """dQ rows of tiles the causal walk does not visit get zeros, not garbage."""
            for t, carry in range(t_lo, t_hi, fx.Index(1), init=[fx.Int32(0)]):
                dq.store(t, None)
                carry = yield [carry[0]]

        @flyc.jit
        def _fused_body():
            # First, so the epilogue's addressing is formed and parked before
            # anything else claims registers; see `FusedKvOutputs`.
            if const_expr(PARK_OUTPUTS):
                outputs.init_outputs(pack16)
            v_k = resident.load_resident("k", hdim_qk)
            v_v = resident.load_resident("v", hdim_vo)

            init_args = []
            for _ in range_constexpr(2 * n_acc):
                init_args.append(zero_acc)
            group_results = init_args

            # The GQA walk of the dK/dV kernel; see there for the runtime trip
            # count and the drain.
            for g, group_args in range(fx.Index(0), ctx.gqa_group, fx.Index(1), init=init_args):
                ctx.retarget_q_head(g)

                @flyc.jit
                def _drain_between_heads(g):
                    if g != fx.Index(0):
                        dualwave._waitcnt_vm_n(0)
                        dualwave._sched_barrier(0)
                        dualwave._s_barrier()

                _drain_between_heads(g)

                for b in range_constexpr(NBUF):
                    stream.stage_q_tile(t0 + fx.Index(b), b)
                    stream.stage_do_tile(t0 + fx.Index(b), b)

                # FUSED: the preprocess, the dQ zero fill, then everything in
                # flight drained so the body's `vmcnt` budget starts from zero.
                _fill_rows(p_lo, p_hi)
                # K into LDS, per head: after the loads above are in flight, so
                # its wait on the resident K is not on the DMA's critical path.
                # Rewriting it for a later head writes identical bytes.
                _stage_k_tile(v_k)
                if const_expr(traits.CAUSAL):
                    _zero_dq_tiles(fx.Index(0), t0)
                    _zero_dq_tiles(t_end, ctx.q_tiles_real)
                _s_waitcnt(_WAIT_ALL)
                _sched_barrier(0)
                _s_barrier()

                loop_results = group_args
                for j, loop_args in range(t0, ctx.pair_t_end, fx.Index(2), init=group_args):
                    dv = [loop_args[i] for i in range_constexpr(n_acc)]
                    dk = [loop_args[n_acc + i] for i in range_constexpr(n_acc)]
                    for slot in range_constexpr(2):
                        dv, dk = tile.run(
                            j + fx.Index(slot),
                            slot % NBUF,
                            j + fx.Index(slot + NBUF),
                            v_k,
                            v_v,
                            dv,
                            dk,
                        )
                    loop_results = yield dv + dk
                # FUSED: an odd walk ends on one tile, in buffer 0 -- the pairs
                # consumed an even number from `t0`, so its parity is slot 0's.
                # A loop of zero or one trips rather than an `if`, because a
                # traced `if` cannot carry the accumulators out of its branch
                # here.
                for j, tail_args in range(ctx.pair_t_end, t_end, fx.Index(1), init=loop_results):
                    dv, dk = tile.run(
                        j,
                        0,
                        j + fx.Index(NBUF),
                        v_k,
                        v_v,
                        [tail_args[i] for i in range_constexpr(n_acc)],
                        [tail_args[n_acc + i] for i in range_constexpr(n_acc)],
                    )
                    loop_results = yield dv + dk

                group_results = yield [loop_results[i] for i in range_constexpr(2 * n_acc)]

            dv = [group_results[i] for i in range_constexpr(n_acc)]
            dk = [group_results[n_acc + i] for i in range_constexpr(n_acc)]
            if const_expr(PARK_OUTPUTS):
                # Not the dK/dV kernels' `store_accs`: see `FusedKvOutputs`.
                dv, dk = outputs.scale(dv, dk, acc_width)
                outputs.store(dv, "dv")
                outputs.store(dk, "dk")
            else:
                for dc in range_constexpr(n_acc):
                    dk[dc] = Vec(dk[dc]) * scale_vec
                store.store_accs(dv, "dv", hdim_vo)
                store.store_accs(dk, "dk", hdim_qk)

        active = ctx.active
        if const_expr(active is None):
            _fused_body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _fused_body()

            _run_body_if_active()

    @flyc.jit
    def launch_fmha_bwd_fused_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        Out: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        DQ: fx.Pointer,
        DB: fx.Pointer,
        LSE: fx.Pointer,
        batch_size: fx.Int32,
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
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
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
        _ = _cache_tag
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        fmha_bwd_fused_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            Out,
            DO,
            DK,
            DV,
            DQ,
            DB,
            LSE,
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
            stride_o_batch,
            stride_o_head,
            stride_o_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
            value_attrs=fmha.kernel_attrs(traits.BLOCK_SIZE, traits.WAVES_PER_EU),
        ).launch(
            # One workgroup per (sequence, KV head), which is the design: the
            # KV block is the whole sequence, so there is no KV-block axis.
            grid=(fx.Index(num_head_k), fx.Index(1), bs_idx),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # On the JitFunction, not around the call, so AOT compiles get them too.
    launch_fmha_bwd_fused_gfx950.compile_hints = dict(fmha.COMPILE_HINTS)

    def _resolve_window_args(window):
        """`(window_left, window_right)` for the wire. The dK/dV kernel's rule."""
        if not traits.CAUSAL:
            if window is not None:
                raise ValueError("window= requires a causal build; this one has causal=False")
            return 0, 0
        if not traits.WINDOW:
            if window is not None:
                raise ValueError("this build is not compiled for windows; pass causal_type=3 in FmhaInputMetadata")
            return fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT
        if window is None:
            raise ValueError(
                "a window build requires window=(left, right); pass "
                "(fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT) for plain bottom-right causal"
            )
        wl, wr = window
        return int(wl), int(wr)

    def _args(
        Q,
        K,
        V,
        O,  # noqa: E741
        DO,
        DQ,
        DK,
        DV,
        LSE,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        window=None,
        varlen=None,
        num_seqlens=0,
        bias=None,
        dropout_p=None,
        philox_seed=None,
        philox_offset1=None,
        philox_offset2=0,
        stream=None,
    ):
        """Every kernel argument but the stream, in launch order.

        The diagonal a *numeric* `window` bound is measured from is VarlenBits
        bit 18 (`principal_diagonal_choice`), and `varlen["bits"]` is its only
        source. A dense call carries it with
        `varlen=fmha.varlen_dense(principal_diagonal=...)`; `varlen=None` is
        word 0, i.e. top-left. Same contract as the forward's and the split
        backward's launchers.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        _ptrs, shape_meta, st = fmha.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("O", O), ("DO", DO), ("DK", DK), ("DV", DV), ("DQ", DQ)],
            q_heads=("O", "DO", "DQ"),
            k_heads=("DK", "DV"),
        )
        del _ptrs
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta
        if num_head_q % num_head_k:
            raise ValueError(f"num_heads_q ({num_head_q}) must be a multiple of num_heads_k ({num_head_k})")
        if int(seqlen_k) > traits.BLOCK_KV:
            raise ValueError(
                f"seqlen_k {int(seqlen_k)} exceeds this build's BLOCK_KV {traits.BLOCK_KV}: one workgroup "
                "owns every KV row, so a longer sequence needs a wider build or the split path"
            )
        rows_needed = ((int(seqlen_q) + traits.BLOCK_Q - 1) // traits.BLOCK_Q + 2) * traits.BLOCK_Q
        if rows_needed > traits.Q_ROWS_CAP:
            raise ValueError(
                f"seqlen_q {int(seqlen_q)} needs {rows_needed} LSE/delta rows in LDS; this build holds "
                f"{traits.Q_ROWS_CAP}. Raise q_rows_cap."
            )
        if hdim_qk > BLOCK_DMODEL or hdim_vo > BLOCK_DMODEL:
            raise ValueError(f"this build serves head dims up to {BLOCK_DMODEL}, got {hdim_qk}/{hdim_vo}")
        if not PADDED_HEAD and (hdim_qk != BLOCK_DMODEL or hdim_vo != BLOCK_DMODEL):
            raise ValueError(f"this build serves head_dim {BLOCK_DMODEL} exactly, got {hdim_qk}/{hdim_vo}")
        if HDIM_QK_FLOOR and min(hdim_qk, hdim_vo) <= HDIM_QK_FLOOR:
            raise ValueError(f"this build serves head dims in ({HDIM_QK_FLOOR}, {BLOCK_DMODEL}]")
        if DQ.shape[3] != hdim_qk or DK.shape[3] != hdim_qk or DV.shape[3] != hdim_vo or O.shape[3] != hdim_vo:
            raise ValueError("dQ/dK must carry hdim_qk and O/dO/dV hdim_vo")
        want = {"bf16": "torch.bfloat16", "f16": "torch.float16"}[traits.DTYPE_STR]
        for name, t in (("Q", Q), ("K", K), ("V", V), ("O", O), ("DO", DO), ("DQ", DQ), ("DK", DK), ("DV", DV)):
            if str(t.dtype) != want:
                raise ValueError(f"{name}: this build is compiled for {want}; got {t.dtype}")
        fmha.row_tensor_arg(LSE, "logsumexp", num_head_q, seqlen_q, varlen)
        if fmha.varlen_is_dense(varlen):
            # Per the word's LSE layout, as in the dK/dV launcher: a dense word
            # (`varlen=None` or `fmha.varlen_dense(...)`) may select TH.
            want = fmha.dense_row_shape(varlen, batch_size, num_head_q, seqlen_q)
            if tuple(LSE.shape) != want:
                raise ValueError(f"logsumexp must be {want}; got {tuple(LSE.shape)}")
        _vl = fmha.varlen_args(STRIDES_CONSTEXPR, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        # The grid's sequence count: the bias's batch extent.
        num_seqs = int(num_seqlens) or int(batch_size)
        *_dp, _dp_keepalive = fmha.dropout_args(
            bool(traits.ENABLE_DROPOUT),
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            device=Q.device,
            stream=stream,
        )
        if dropout_p is not None and not traits.ENABLE_DROPOUT:
            raise ValueError("this build was not compiled for dropout")
        if traits.BIAS_TYPE and bias is None:
            raise ValueError("this build has bias=True and requires a (batch, num_heads, seqlen_q, seqlen_k) tensor")
        if bias is not None and not traits.BIAS_TYPE:
            raise ValueError("this build was not compiled for bias")
        if bias is not None:
            # The split backward's contract; see `check_score_tensor`.
            check_score_tensor("bias", bias, num_seqs, num_head_q, _vl[5], _vl[6], Q.dtype)
        bias_t = bias if bias is not None else DK
        bias_st = tuple(int(x) for x in bias.stride()[:3]) if bias is not None else (0, 0, 0)
        return (
            wire_ptr(Q),
            wire_ptr(K),
            wire_ptr(V),
            wire_ptr(bias_t),
            wire_ptr(O),
            wire_ptr(DO),
            wire_ptr(DK),
            wire_ptr(DV),
            wire_ptr(DQ),
            # dB is not produced yet; the slot is in the ABI so adding it does
            # not move the wire format.
            wire_ptr(DK),
            wire_ptr(LSE, fx.Float32),
            int(batch_size),
            _vl[1],
            _vl[2],
            _vl[3],
            _vl[4],
            _vl[0],
            int(num_seqlens),
            *_vl[5:],
            *_resolve_window_args(window),
            _dp[0],
            _dp[1],
            _dp[2],
            _dp[3],
            _dp[4],
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            fmha.resolve_scale(
                Q, scale if scale is not None else BUILD_SM_SCALE, PADDED_HEAD, 1.0 / (BLOCK_DMODEL**0.5)
            ),
            # Q, K, V, O, dO, dK, dV, dQ: `prep_tensors` order.
            *st,
            *bias_st,
            0,
            0,
            0,
        ), stream

    def _launch(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        return fmha.run_compiled(
            _COMPILED,
            launch_fmha_bwd_fused_gfx950,
            *packed,
            stream if stream is not None else fx.Stream(None),
        )

    def _compile(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        return flyc.compile(launch_fmha_bwd_fused_gfx950, *packed, fx.Stream(stream))

    _launch.compile = _compile
    _launch.traits = traits
    _launch.knobs = knobs
    _launch.varlen_dense = fmha.varlen_dense
    return _launch


def build_fmha_bwd_fused_gfx950_module(*, max_seqlen_k, **kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    `FmhaInputMetadata`'s fields name the problem and `max_seqlen_k` the KV
    rung; any other keyword pins a knob. The real head dims are rounded up the
    ladder, as the dispatchers' `plan` does.
    """
    from dataclasses import fields

    meta_fields = {f.name for f in fields(FmhaInputMetadata)}
    # block_dmodel is required but nullable: None lets `plan` pick the tile.
    meta = FmhaInputMetadata(**{"block_dmodel": None, **{k: v for k, v in kwargs.items() if k in meta_fields}})
    meta, knobs = entry_plan(meta,
                             max_seqlen_k=max_seqlen_k,
                             overrides={k: v for k, v in kwargs.items() if k not in meta_fields})
    return build_fmha_bwd_fused_gfx950_module_primary(meta, knobs)
