# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention forward for gfx942, FlyDSL: the narrow body (head dims <= 128)
on `flex_gemm` and the `softmax` hooks.

No LDS transpose read on gfx942, so the kernel computes S^T = K.Q^T and
O^T += V^T.P^T: S^T's MFMA output layout is P^T's operand layout, and V is
transposed through VGPRs on its way into LDS. The kernel is a schedule over GEMM
phases and softmax hooks (flyc-gfx942-flex-gemm-design.md R2).

The forward's entry module: this file holds the narrow body, and
`build_fmha_fwd_gfx942_module_primary` picks any body from the knobs. Every
body's kernel is `fmha_fwd_gfx942_kernel`, with this file's signature.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, rocdl

import fmha_utils as fmha
from . import flex_gemm
from . import host
from . import softmax
from . import tensor_io
from . import utils
from . import tuning_fwd as tuning
from .flex_gemm import (
    gemm_dma,
    gemm_dma_plan,
    gemm_load,
    gemm_load_plan,
    gemm_mma,
    gemm_read,
    gemm_read_plan,
    gemm_stage,
)
from .utils import (
    WAVE,
    Lds,
    alias_scopes,
    band_window,
    barrier_if,
    bias_slab,
    body_builder,
    build_body_primary,
    clean_tiles,
    cluster_barrier,
    kv_tiles,
    philox_plane,
    philox_stream,
    row_band,
    rsrc_for,
    smax,
    smin,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_fwd_gfx942.lds">'
LDS_SCOPES = ('k0', 'k1', 'vt0', 'vt1')
WAVE_M = 32         # query rows per wave: the N of the 32x32x8 MFMA


@dataclass(frozen=True)
class FwdTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    num_waves: int
    block_n: int
    waves_per_eu: int
    rescale_threshold: float
    grid_axis_order: int
    bias: bool
    dropout: bool

    # make_fwd_launch's names
    @property
    def D(self):  # noqa: N802
        return self.head_dim

    @property
    def PADDED(self):  # noqa: N802
        return self.padded

    @property
    def block_size(self):
        return self.num_waves * WAVE

    @property
    def block_m(self):
        return self.num_waves * WAVE_M

    @property
    def pad_floor(self):
        """Columns below this are real in every call: a padded build is chosen by
        max(hdim_qk, hdim_vo), so the other one can be anything."""
        return 0 if self.padded else self.head_dim

    def gemms(self):
        """(QK, PV): S^T = K.Q^T and O^T += V^T.P^T, double-buffered in LDS."""
        qk = flex_gemm.trait_ab(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.pad_floor,
                                self.block_size,
                                0,
                                'k')
        pv = flex_gemm.trait_tb(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.block_size,
                                2 * qk.a.tile_bytes,
                                'vt')
        return qk, pv


def build_fmha_fwd_gfx942(head_dim,
                               causal=False,
                               causal_type=None,
                               dtype_str='bf16',
                               bias=False,
                               dropout=False,
                               num_waves=8,
                               block_n=None,
                               waves_per_eu=2,
                               rescale_threshold=0.0,
                               grid_axis_order=None,
                               padded_head=False):
    """A launcher with gfx1201/fwd.py's call interface and kernel ABI.

    `causal_type` as gfx1201's: 1/2 are host sentinels, 3 is gSWA; all compile to 3.
    `rescale_threshold` > 0 enables the lazy running max (FA4's, off by default;
    numerical lessons, #17): the max may lag that many log2 units.
    """
    assert head_dim <= 128, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if block_n is None:
        block_n = tuning.default_block_n(head_dim)
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    tr = FwdTraits(head_dim=head_dim,
                   padded=padded_head,
                   causal=causal_type != 0,
                   dtype=dtype_str,
                   num_waves=num_waves,
                   block_n=block_n,
                   waves_per_eu=waves_per_eu,
                   rescale_threshold=rescale_threshold,
                   grid_axis_order=grid_axis_order,
                   bias=bias,
                   dropout=dropout)
    qk, pv = tr.gemms()
    lds_bytes = 2 * qk.a.tile_bytes + 2 * pv.a.tile_bytes
    assert lds_bytes <= 65536, lds_bytes
    tag = 'fwd_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    block_n = tr.block_n

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    def scopes(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

    def compute_begin():
        rocdl.s_setprio(1)

    def compute_end(name=None):
        rocdl.s_setprio(0)
        cluster_barrier(name)

    @flyc.kernel(known_block_size=[tr.block_size, 1, 1])
    def fmha_fwd_gfx942_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
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
        philox_seed_output: fx.Pointer,
        philox_offset_output: fx.Pointer,
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
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
    ):
        _ = tag                     # puts every traits field in the JIT cache key
        # ctx: the frozen traits plus the run-time state. `carried` names what
        # crosses the tile loop's back edge (softmax state, O, V(j+1) in VGPRs).
        ctx = flex_gemm.Ctx(tr, carried=('m', 'corr', 'l', 's', 'o', 'v_regs'))
        ctx.qk, ctx.pv = qk, pv

        # ---- prelude: addressing, resources, per-lane plans (the order the ISA was tuned in)
        # Lane geometry and the (q tile, head) this block serves; GQA maps the
        # query head to its kv head. Varlen decodes each sequence's length and offset.
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lm = lane % 32
        lh = lane // 32
        if const_expr(tile_fastest):
            q_tile = fx.Index(gpu.block_idx.x)
            head = fx.Index(gpu.block_idx.y)
        else:
            head = fx.Index(gpu.block_idx.x)
            q_tile = fx.Index(gpu.block_idx.y)
        if const_expr(tr.causal):
            # Longest tiles first: under any causal window later rows see more keys.
            # Count tiles as the C++ grid does; gpu.grid_dim would add hidden kernargs.
            n_q_tiles = (fx.Index(max_seqlen_q) + tr.block_m - 1) // tr.block_m
            q_tile = n_q_tiles - fx.Index(1) - q_tile
        z_i32 = fx.Int32(gpu.block_idx.z)
        head_k = head // (fx.Index(num_head_q) // fx.Index(num_head_k))
        sq_i32, q_row_off, q_batch = fmha.decode_addressing(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32)
        sk_i32, k_row_off, k_batch = fmha.decode_addressing(varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32)
        sq = fx.Index(sq_i32)
        sk = fx.Index(sk_i32)

        # LDS: K(j) double-buffered at 0, V^T(j) double-buffered after it.
        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_addr = fx.Index(fx.ptrtoint(lds_p))
        ctx.lds = Lds(lds_p, elem, _LDS_DOMAIN, LDS_SCOPES)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)
        q_rsrc = rsrc_for(Q, q_batch, head, q_row_off, stride_q_batch, stride_q_head, stride_q_seq, sq, row_qk)
        o_rsrc = rsrc_for(O, q_batch, head, q_row_off, stride_o_batch, stride_o_head, stride_o_seq, sq, row_vo)
        k_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk, row_qk))
        v_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk, row_vo))
        q_row_stride = fx.Index(fx.Int32(stride_q_seq))
        k_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)
        o_row_stride = fx.Index(fx.Int32(stride_o_seq))

        # Q rows and the O store, for softmax.epilogue.
        ctx.set_q_rows(varlen_bits=varlen_bits,
                       q_batch=q_batch,
                       q_row_off=q_row_off,
                       sq=sq,
                       max_seqlen_q=max_seqlen_q,
                       seqinfo_q0=seqinfo_q0,
                       seqinfo_q1=seqinfo_q1,
                       num_seqlens=num_seqlens,
                       num_head_q=num_head_q,
                       head=head)
        ctx.set_o_store(o_rsrc=o_rsrc,
                        o_row_stride=o_row_stride,
                        o_col_limit=row_vo,
                        dropout_scale=dropout_scale)

        # Each wave owns 32 query rows; a lane holds one row (lm) and one half of
        # the 32x32x8 accumulator's columns (lh).
        q_start = q_tile * tr.block_m
        q_row = q_start + wave * WAVE_M + lm

        # Q^T stays in VGPRs for the whole loop. The plans are per-lane addresses,
        # computed once so the loop body only adds tile offsets.

        ctx.q_packs = flex_gemm.resident_packs(qk, q_rsrc, q_row, q_row_stride, lh, hdim_qk)
        k_dma_plan = gemm_dma_plan(qk, wave, lane, k_stride_b, lds_addr)
        v_load_plan = gemm_load_plan(pv, tid, v_stride_b)
        k_read_plan = gemm_read_plan(qk, lm, lh)
        vt_read_plan = gemm_read_plan(pv, lm, lh)
        ctx.k_masks = flex_gemm.a_col_masks(qk, hdim_qk, lh)

        # Softmax constants: scores go to base 2 (exp2); swap_addr reaches the lane
        # holding the other half of this row.
        ctx.c_scale = fx.Float32(sm_scale) * fx.Float32(softmax.LOG2E)
        ctx.c_neg_inf = fx.Float32(float('-inf'))
        c_neg_max = fx.Float32(-3.4028234663852886e38)
        ctx.swap_addr = fx.Int32(lane ^ fx.Index(32)) * fx.Int32(4)
        ctx.kv_lane = fx.Int32(lh * 4)

        # Tiles: masked iterations bracket the clean ones; iteration j masks tile j+1.
        win_l, win_r = band_window(tr.causal, window_left, window_right, sq_i32, sk_i32, varlen_bits)
        q_first, q_last, kv_base, n_tiles = kv_tiles(q_start, tr.block_m, block_n, sq_i32, sk_i32, win_l, win_r)
        if const_expr(tr.causal):
            # Causal: a lane's band and key are rebuilt per masked tile from the lane
            # id; kept live, their VGPRs made the masked loops spill Q.
            q_wave = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(q_start + wave * WAVE_M)))
            ctx.band = lambda: row_band(q_wave + (utils.lane_now() & fx.Int32(31)), win_l, win_r, sk_i32)
            ctx.lane_key = lambda: (utils.lane_now() >> fx.Int32(3)) & fx.Int32(4)
        else:
            band = row_band(q_row, win_l, win_r, sk_i32)
            ctx.band = lambda: band
            ctx.lane_key = lambda: ctx.kv_lane
        lead, clean_end = clean_tiles(q_first, q_last, kv_base, block_n, sk_i32, win_l, win_r)
        if const_expr(bias):
            # Only the last tile can hold the dword that straddles the bias slab's end.
            clean_end = smin(clean_end, smax(n_tiles, fx.Index(1)) - fx.Index(1))
        n_loop = (n_tiles > fx.Index(1)).select(n_tiles - fx.Index(1), fx.Index(0))
        j_clean = smin(smax(lead, fx.Index(1)) - fx.Index(1), n_loop)
        j_clean = smin(j_clean + j_clean % 2, n_loop)
        j_clean_end = smax(smin(smax(clean_end, fx.Index(1)) - fx.Index(1), n_loop), j_clean)

        # Bias: this row's slab of B and the lane's byte position in it; loads are
        # dword-aligned, so the shift picks the bf16/f16 pairs out.
        if const_expr(bias):
            ctx.b_rsrc, b_mis = bias_slab(B,
                                          q_batch,
                                          head,
                                          q_row_off,
                                          k_row_off,
                                          stride_b_batch,
                                          stride_b_head,
                                          stride_b_seq_q,
                                          sq_i32,
                                          sk_i32)
            ctx.b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + ctx.kv_lane) * fx.Int32(2)
            ctx.b_lane = ctx.b_byte & fx.Int32(-4)
            ctx.b_shift = (ctx.b_byte & fx.Int32(3)) * fx.Int32(8)
        if const_expr(dropout):
            # The Triton kernels' stream, element for element (philox.py, grid_plane).
            ctx.rng, ctx.ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            fmha.philox_report(philox_seed_output, philox_offset_output, ctx.ph_seed, ph_off)
            ph_base, ph_stride = philox_plane(ctx.rng, ph_off, z_i32, num_head_q, head, max_seqlen_q, max_seqlen_k)
            ctx.ph_row = ctx.rng.grid_offset(ph_base, ph_stride, fx.Int32(q_row), ctx.kv_lane)
            ctx.idropout_p = idropout_p

        # Waves 4..7 trail by one barrier: LDS producer->consumer needs two clusters.
        group_b = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                               fx.Int32((wave >= fx.Index(tr.num_waves // 2)).select(1, 0))))

        def kv(j):
            return kv_base + j * block_n

        # ---- memory clusters: which tile each load serves is visible here
        # They own every wait and barrier; the GEMM phases never wait.
        def mem0(kv_next, b, masked):
            """C0(j): V^T(j) -> LDS, V(j+1) -> VGPRs, K(j+1) LDS -> VGPRs, bias(j+1)."""
            gemm_stage(pv, v_load_plan, ctx.lds, scopes, ctx.v_regs, b)
            ctx.v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_next, v_stride_b)
            ctx.k_reads = gemm_read(qk, k_read_plan, ctx.lds, 1 - b)
            ctx.b_next = softmax.bias_load(ctx, kv_next, masked)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier(f'C0.{b}')

        def mem1(kv_next, b):
            """C2(j): K(j+3) -> LDS, V^T(j) LDS -> VGPRs."""
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_next + 2 * block_n, k_stride_b, 1 - b)
            ctx.vts = gemm_read(pv, vt_read_plan, ctx.lds, b)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_waitcnt(vmcnt=pv.a.n_load + qk.a.n_dma)
            cluster_barrier(f'C2.{b}')

        # Iteration -1 fills the pipeline the loop expects: K(1), K(2) in flight,
        # V(0) in VGPRs, S(0) through head(0).
        def warmup():
            """Iteration -1: K(0..2) DMAs, V(0), QK(0), head(0); opens the stagger."""
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base, k_stride_b, 0)
            rocdl.sched_barrier(0)              # the vmcnt below waits for K(0) only
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base + block_n, k_stride_b, 1)
            rocdl.s_waitcnt(vmcnt=qk.a.n_dma)
            cluster_barrier()
            barrier_if(group_b)
            k_reads = gemm_read(qk, k_read_plan, ctx.lds, 0)
            ctx.v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_base, v_stride_b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            compute_begin()
            b0 = softmax.bias_load(ctx, kv_base, True)
            s_raw = gemm_mma(qk, k_reads, ctx.q_packs, a_masks=ctx.k_masks)
            compute_end()
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base + 2 * block_n, k_stride_b, 0)
            rocdl.s_waitcnt(vmcnt=pv.a.n_load + qk.a.n_dma)
            cluster_barrier()
            compute_begin()
            ctx.m = c_neg_max               # finite: fully masked rows stay NaN-free
            softmax.head(ctx, s_raw, kv_base, True, b0)
            compute_end()
            ctx.l = fx.Float32(0.0)
            ctx.o = [fx.full(16, 0.0, fx.Float32) for _ in range(pv.out_tiles)]

        # ---- the tile loop: QK(j+1) and PV(j) with softmax(j) and (j+1) under them
        # `b` = j % 2 is a Python int (alias scopes need constants); `masked` says
        # whether tile j+1 needs the causal/window/padding mask.
        def iteration(j, b, masked):
            kv_next = kv(j + 1)
            mem0(kv_next, b, masked)
            compute_begin()                                                     # C1(j)
            # Causal masked tiles rescale O unconditionally: the skip branch kept a second O live.
            ctx.o = (softmax.rescale_always if masked and tr.causal else softmax.rescale)(ctx.o, ctx.corr)
            s_next = gemm_mma(qk, ctx.k_reads, ctx.q_packs, a_masks=ctx.k_masks)  # QK(j+1)
            p = softmax.tail(ctx, kv(j))                                        # P(j)
            compute_end(f'C1.{b}')
            mem1(kv_next, b)
            compute_begin()                                                     # C3(j)
            ctx.o = gemm_mma(pv, ctx.vts, p, ctx.o)                             # PV(j)
            softmax.head(ctx, s_next, kv_next, masked, ctx.b_next)              # max(j+1)
            compute_end(f'C3.{b}')

        # The loop leaves the last tile's PV undone; drain picks its buffer at run time.
        def finish(state, b, kv_last):
            """The last tile's PV; returns [m, l] + O. Starts from `state`: a branch
            must not see what the other branch did to ctx."""
            ctx.restore(state)
            gemm_stage(pv, v_load_plan, ctx.lds, scopes, ctx.v_regs, b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            compute_begin()
            o = softmax.rescale(ctx.o, ctx.corr)
            p = softmax.tail(ctx, kv_last)
            compute_end()
            vts = gemm_read(pv, vt_read_plan, ctx.lds, b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            o = gemm_mma(pv, vts, p, o)
            return [ctx.m, ctx.l] + o

        @flyc.jit
        def drain(n_loop):
            state = ctx.carry()
            out = [ctx.m, ctx.l] + list(ctx.o)
            kv_last = kv(n_loop)
            if n_loop % 2 == fx.Index(0):
                out = finish(state, 0, kv_last)
            else:
                out = finish(state, 1, kv_last)
            return out

        # Masked tiles run first and last, the clean middle unrolled x2 (flex_gemm.run_tiles).
        warmup()
        ctx.tiles(iteration, j_clean, j_clean_end, n_loop)
        results = drain(n_loop)
        barrier_if(fx.Int32(1) - group_b)         # close the stagger

        # ---- epilogue: O / l, LSE (l split over two lane halves)
        softmax.epilogue(ctx,
                         pv,
                         results,
                         LSE,
                         row=q_row,
                         lane=lh,
                         lane_first=lambda: lh == fx.Index(0))

    @flyc.jit
    def launch_fmha_fwd_gfx942(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
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
        philox_seed_output: fx.Pointer,
        philox_offset_output: fx.Pointer,
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
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_y = (fx.Index(max_seqlen_q) + tr.block_m - 1) // tr.block_m
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_fwd_gfx942_kernel(Q,
                               K,
                               V,
                               B,
                               O,
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
                               philox_seed_output,
                               philox_offset_output,
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
                               stride_b_batch,
                               stride_b_head,
                               stride_b_seq_q,
                               value_attrs={'rocdl.waves_per_eu': waves_per_eu,
                                            'rocdl.flat_work_group_size': f'{tr.block_size},{tr.block_size}'}).launch(
            grid=(grid_y, fx.Index(num_head_q), grid_z) if tile_fastest else (fx.Index(num_head_q), grid_y, grid_z),
            block=(tr.block_size, 1, 1),
            stream=stream)

    return host.make_fwd_launch(launch_fmha_fwd_gfx942,
                                tr,
                                causal_type,
                                host_causal_type,
                                bias,
                                dropout)


_BODY_KNOBS = ('num_waves', 'block_n', 'waves_per_eu', 'rescale_threshold')
_BODIES = {
    'narrow': (__name__, 'build_fmha_fwd_gfx942', _BODY_KNOBS),
    'wide': (f'{__package__}.fwd_wide', 'build_fmha_fwd_gfx942_wide', _BODY_KNOBS),
    'xwide': (f'{__package__}.fwd_xwide', 'build_fmha_fwd_gfx942_xwide', _BODY_KNOBS),
}


def build_fmha_fwd_gfx942_module_primary(meta, knobs):
    """The body `knobs.body` names, built for a resolved (meta, knobs) pair:
    `fmha_meta.FmhaInputMetadata` and `tuning_fwd.entry_resolve(meta)`."""
    return build_body_primary(_BODIES, meta, knobs)


def build_fmha_fwd_gfx942_by_head_dim(head_dim, body=None, **kwargs):
    """A JIT build of the compiled `head_dim` by `body`'s builder (default: the
    body the policy routes `head_dim` to), with that builder's keywords and defaults."""
    return body_builder(_BODIES, body or tuning.body(head_dim))(head_dim, **kwargs)
