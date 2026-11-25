# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The wide forward (head dims 160..256) on `flex_gemm` and the `softmax` hooks.

A schedule over GEMM phases and softmax hooks: 4 waves x 32 rows, one wave per
SIMD, so no second wave hides memory and each KV tile is two barrier-separated clusters that mix loads into their MFMAs:
    X(j): QK(j+1), softmax tail(j)
    Y(j): PV(j), softmax head(j+1)
K is double-buffered by row DMA; V^T is single-buffered (the waves run in
lock-step). Causal blocks run two q tiles so every block does the same work.
Same kernel ABI and launcher interface as the narrow body in `fwd.py`.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

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
    bias_slab,
    clean_tiles,
    cluster_barrier,
    kv_tiles,
    philox_plane,
    philox_stream,
    pin_values,
    row_band,
    rsrc_for,
    smax,
    smin,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_fwd_gfx942_wide.lds">'
LDS_SCOPES = ('k0', 'k1', 'vt')
WAVE_M = 32         # query rows per wave: the N of the 32x32x8 MFMA


@dataclass(frozen=True)
class FwdWideTraits:
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
    # Whole K / V^T tiles reach VGPRs a cluster before their MFMAs; past 224 they
    # do not fit beside O, so the MFMAs stream them, `vt_lead` reads ahead.
    k_early: bool
    vt_early: bool
    vt_lead: int = 8
    # O leaves through LDS in whole row chunks. Measured per head dim: hd256 gains
    # (B4 H32 S8k: 387 -> 400 causal, 440 -> 449); 192 and 224 lose non-causal.
    staged_o: bool = False
    valu_per_mfma: int = 3          # sched_group_barrier slot size

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
    def pair_tiles(self):
        """Causal blocks pair q tiles g and T-1-g."""
        return self.causal

    @property
    def grid_block_m(self):
        return self.block_m * (2 if self.pair_tiles else 1)

    @property
    def pad_floor(self):
        """Columns below this are real in every call: a padded build is chosen by
        max(hdim_qk, hdim_vo), so the other one can be anything."""
        return 0 if self.padded else self.head_dim

    def gemms(self):
        """(QK, PV): S^T = K.Q^T, K double-buffered by row DMA; O^T += V^T.P^T,
        V^T single-buffered, loaded 4 rows x 8 columns per thread."""
        qk = flex_gemm.trait_ab(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.pad_floor,
                                self.block_size,
                                0,
                                'k',
                                dma='rows')
        pv = flex_gemm.trait_tb(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.block_size,
                                2 * qk.a.tile_bytes,
                                'vt',
                                buffers=1,
                                unit_cols=8)
        return qk, pv


def build_fmha_fwd_gfx942_wide(head_dim,
                                    causal=False,
                                    causal_type=None,
                                    dtype_str='bf16',
                                    bias=False,
                                    dropout=False,
                                    num_waves=4,
                                    block_n=None,
                                    waves_per_eu=1,
                                    rescale_threshold=0.0,
                                    grid_axis_order=None,
                                    padded_head=False):
    """build_fmha_fwd_gfx942's interface and kernel ABI, for wide heads."""
    assert head_dim % 32 == 0 and 128 <= head_dim <= 256, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if block_n is None:
        block_n = 32
    assert block_n == 32, block_n
    if grid_axis_order is None:
        # Tile-fastest shares K/V in L2; paired tiles balance causal blocks.
        grid_axis_order = tuning.GRID_AXIS_TILE_FASTEST
    tr = FwdWideTraits(head_dim=head_dim,
                       padded=padded_head,
                       causal=causal_type != 0,
                       dtype=dtype_str,
                       num_waves=num_waves,
                       block_n=block_n,
                       waves_per_eu=waves_per_eu,
                       rescale_threshold=rescale_threshold,
                       grid_axis_order=grid_axis_order,
                       bias=bias,
                       dropout=dropout,
                       k_early=head_dim <= 224,
                       vt_early=head_dim <= 224,
                       staged_o=head_dim == 256)
    qk, pv = tr.gemms()
    assert block_n % num_waves == 0 and qk.a.row_bytes >= 256
    lds_bytes = 2 * qk.a.tile_bytes + pv.a.tile_bytes
    assert lds_bytes <= 65536, lds_bytes
    # O leaves through LDS in passes of o_width columns, 32 rows per wave.
    o_width = next(w for w in (128, 64, 32) if head_dim % w == 0 and num_waves * WAVE_M * 2 * w <= lds_bytes)
    tag = 'fwd_wide_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    block_n = tr.block_n
    k_early = tr.k_early
    vt_early = tr.vt_early
    # Per-tile op counts for the sched_group_barrier spread. V^T reads pair into
    # ds_read2_b64, hence n_vt = half the PV MFMAs.
    n_kdma = qk.a.n_dma
    n_v = pv.a.n_load
    n_vst = pv.a.n_stage
    n_k = qk.n_reads
    n_vt = pv.n_mma // 2

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    def scopes(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

    # What x_loads / y_loads issue per cluster: an early K or V^T moves its work
    # from the cluster of its MFMAs to the other one.
    x_ops = flex_gemm.ClusterOps(rd=(n_vt if vt_early else 0) + (0 if k_early else n_k),
                       wr=0 if vt_early else n_vst,
                       vmem=(n_kdma if k_early else 0) + (0 if vt_early else n_v),
                       lead=0 if k_early else tr.vt_lead)
    y_ops = flex_gemm.ClusterOps(rd=(n_k if k_early else 0) + (0 if vt_early else n_vt),
                       wr=n_vst if vt_early else 0,
                       vmem=(n_v if vt_early else 0) + (0 if k_early else n_kdma),
                       lead=0 if vt_early else tr.vt_lead)

    def spread(n_mfma, ops):
        flex_gemm.spread(n_mfma, ops, valu_per_mfma=tr.valu_per_mfma)

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

        # ---- block geometry: lanes, the (q tile(s), head) this block serves, GQA, varlen
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lm = lane % 32
        lh = lane // 32
        if const_expr(tile_fastest):
            g_tile = fx.Index(gpu.block_idx.x)
            head = fx.Index(gpu.block_idx.y)
        else:
            head = fx.Index(gpu.block_idx.x)
            g_tile = fx.Index(gpu.block_idx.y)
        if const_expr(tr.pair_tiles):
            # Block g runs q tiles g and T-1-g: causal blocks then all do the same work.
            # T odd: the middle block's second tile is T, which has no rows.
            n_q_tiles = (fx.Index(max_seqlen_q) + tr.block_m - 1) // tr.block_m
            mirror = n_q_tiles - fx.Index(1) - g_tile
            q_tiles = [mirror, (mirror == g_tile).select(n_q_tiles, g_tile)]
        else:
            q_tiles = [g_tile]
        z_i32 = fx.Int32(gpu.block_idx.z)
        head_k = head // (fx.Index(num_head_q) // fx.Index(num_head_k))
        sq_i32, q_row_off, q_batch = fmha.decode_addressing(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32)
        sk_i32, k_row_off, k_batch = fmha.decode_addressing(varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32)
        sq = fx.Index(sq_i32)

        # LDS: K(j) double-buffered at 0, one V^T tile after it.
        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_addr = fx.Index(fx.ptrtoint(lds_p))
        lds_io = Lds(lds_p, elem, _LDS_DOMAIN, LDS_SCOPES)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)
        q_rsrc = rsrc_for(Q, q_batch, head, q_row_off, stride_q_batch, stride_q_head, stride_q_seq, sq, row_qk)
        o_rsrc = rsrc_for(O, q_batch, head, q_row_off, stride_o_batch, stride_o_head, stride_o_seq, sq, row_vo)
        k_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk_i32, row_qk))
        v_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk_i32, row_vo))
        q_row_stride = fx.Index(fx.Int32(stride_q_seq))
        k_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)
        o_row_stride = fx.Index(fx.Int32(stride_o_seq))

        for q_pass in range_constexpr(len(q_tiles)):
            # One pass per q tile; everything below is per pass.
            # `carried` names what crosses the tile loop's back edge.
            ctx = flex_gemm.Ctx(tr, carried=('m', 'corr', 'l', 's', 'o', 'k_regs', 'v_regs'))
            ctx.qk, ctx.pv = qk, pv
            ctx.lds = lds_io
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
            q_tile = q_tiles[q_pass]
            q_start = q_tile * tr.block_m
            q_wave = q_start + wave * WAVE_M
            q_row = q_wave + lm

            # Q^T in VGPRs for the pass; per-lane addresses of every phase, computed once.
            ctx.q_packs = flex_gemm.resident_packs(qk, q_rsrc, q_row, q_row_stride, lh, hdim_qk)
            k_dma_plan = gemm_dma_plan(qk, wave, lane, k_stride_b, lds_addr)
            v_load_plan = gemm_load_plan(pv, tid, v_stride_b)
            k_read_plan = gemm_read_plan(qk, lm, lh)
            vt_read_plan = gemm_read_plan(pv, lm, lh)
            ctx.k_masks = flex_gemm.a_col_masks(qk, hdim_qk, lh)

            # Softmax constants: scores go to base 2; swap_addr reaches the other row half.
            ctx.c_scale = fx.Float32(sm_scale) * fx.Float32(softmax.LOG2E)
            ctx.c_neg_inf = fx.Float32(float('-inf'))
            c_neg_max = fx.Float32(-3.4028234663852886e38)
            ctx.swap_addr = fx.Int32(lane ^ fx.Index(32)) * fx.Int32(4)
            ctx.kv_lane = fx.Int32(lh * 4)

            # Tiles: iteration j masks tile j+1. Masked iterations [0, j_clean) and
            # [j_clean_end, n_loop) bracket the clean ones; j_clean is even.
            win_l, win_r = band_window(tr.causal, window_left, window_right, sq_i32, sk_i32, varlen_bits)
            q_first, q_last, kv_base, n_tiles = kv_tiles(q_start, tr.block_m, block_n, sq_i32, sk_i32, win_l, win_r)
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

            # Bias: this row's slab of B and the lane's byte position in it.
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

            def kv(j):
                return kv_base + j * block_n

            # LDS lifetimes (all waves in lock-step). Early K / V^T reach VGPRs a
            # cluster before their MFMAs; otherwise the MFMAs stream them.
            #   K(t):   k_early: DMA X(t-3) -> kb[t%2], waited end X(t-2), read Y(t-2), QK X(t-1)
            #           else:    DMA Y(t-3) -> kb[t%2], waited end Y(t-2), read X(t-1) by QK
            #   V^T(t): vt_early: loaded Y(t-2), written Y(t-1) -> vt, read X(t); PV Y(t)
            #           else:     loaded X(t-1), written X(t) -> vt, read Y(t) by PV

            # ---- loads of each cluster: which tile each serves is visible here
            def x_loads(kv_next, b, masked):
                """X(j): bias(j+1), K(j+3) DMA (early K); V^T(j) -> LDS and V(j+1)
                (streamed V^T); V^T(j) reads (early V^T); K(j+1) reads (streamed K)."""
                ctx.b_next = softmax.bias_load(ctx, kv_next, masked)
                if const_expr(k_early):
                    gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_next + 2 * block_n, k_stride_b, 1 - b)
                if const_expr(not vt_early):
                    gemm_stage(pv, v_load_plan, ctx.lds, scopes, ctx.v_regs, 0)
                    ctx.v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_next, v_stride_b)
                vts = gemm_read(pv, vt_read_plan, ctx.lds, 0) if vt_early else None
                k_reads = ctx.k_regs if k_early else gemm_read(qk, k_read_plan, ctx.lds, 1 - b)
                return vts, k_reads

            def y_loads(kv_next, b, vts):
                """Y(j): K(j+2) reads (early K); V^T(j+1) -> LDS and V(j+2) (early V^T);
                K(j+3) DMA (streamed K); V^T(j) reads (streamed V^T)."""
                if const_expr(k_early):
                    ctx.k_regs = gemm_read(qk, k_read_plan, ctx.lds, b)
                if const_expr(vt_early):
                    gemm_stage(pv, v_load_plan, ctx.lds, scopes, ctx.v_regs, 0)
                    ctx.v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_next + block_n, v_stride_b)
                if const_expr(not k_early):
                    gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_next + 2 * block_n, k_stride_b, 1 - b)
                return vts if vt_early else gemm_read(pv, vt_read_plan, ctx.lds, 0)

            # ---- cluster ends: wait for what the next cluster consumes or overwrites
            def x_end(b):
                if const_expr(k_early):
                    rocdl.s_waitcnt(vmcnt=n_v + n_kdma)     # K(j+2)'s DMA, read in Y(j)
                if const_expr(vt_early):
                    # Y(j) overwrites V^T; s_barrier does not wait for these reads.
                    rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier(f'X{b}')

            def y_end(b):
                if const_expr(not k_early):
                    rocdl.s_waitcnt(vmcnt=n_v + n_kdma)     # K(j+2)'s DMA, read in X(j+1)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier(f'Y{b}')

            # Iteration -1: K(0), K(1) land; X(-1) = QK(0); Y(-1) = softmax head(0).
            def warmup():
                gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base, k_stride_b, 0)
                gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base + block_n, k_stride_b, 1)
                # The vmcnt below counts V and bias as the newest loads: keep them after the DMAs.
                rocdl.sched_barrier(0)
                ctx.v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_base, v_stride_b)
                b0 = softmax.bias_load(ctx, kv_base, True)
                rocdl.s_waitcnt(vmcnt=n_v + (qk.ns if bias else 0))
                cluster_barrier()
                if const_expr(k_early):
                    k_reads = gemm_read(qk, k_read_plan, ctx.lds, 0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    cluster_barrier()
                if const_expr(k_early):
                    gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base + 2 * block_n, k_stride_b, 0)
                    s_raw = gemm_mma(qk, k_reads, ctx.q_packs, a_masks=ctx.k_masks)
                else:
                    s_raw = gemm_mma(qk, gemm_read(qk, k_read_plan, ctx.lds, 0), ctx.q_packs, a_masks=ctx.k_masks)
                    rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()
                if const_expr(k_early):
                    ctx.k_regs = gemm_read(qk, k_read_plan, ctx.lds, 1)
                if const_expr(vt_early):
                    gemm_stage(pv, v_load_plan, ctx.lds, scopes, ctx.v_regs, 0)
                    ctx.v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_base + block_n, v_stride_b)
                if const_expr(not k_early):
                    gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base + 2 * block_n, k_stride_b, 0)
                ctx.m = c_neg_max               # finite: fully masked rows stay NaN-free
                softmax.head(ctx, s_raw, kv_base, True, b0)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()
                if const_expr(not k_early):
                    ctx.k_regs = []
                ctx.l = fx.Float32(0.0)
                ctx.o = [fx.full(16, 0.0, fx.Float32) for _ in range(pv.out_tiles)]

            # ---- the tile loop: X(j) = QK(j+1) + tail(j), Y(j) = PV(j) + head(j+1)
            # `b` = j % 2 picks K's buffer (a Python int: alias scopes need constants).
            def iteration(j, b, masked):
                kv_cur = kv(j)
                kv_next = kv_cur + block_n
                # X(j): loads spread over QK(j+1)'s MFMAs; P(j) pinned so it stays here.
                vts, k_reads = x_loads(kv_next, b, masked)
                s_next = gemm_mma(qk, k_reads, ctx.q_packs, a_masks=ctx.k_masks)      # QK(j+1)
                p = pin_values(softmax.tail(ctx, kv_cur))                             # P(j)
                spread(qk.n_mma, x_ops)
                x_end(b)
                # Y(j): rescale O, then PV(j); head(j+1)'s VALU fills the MFMA gaps.
                ctx.o = softmax.rescale(ctx.o, ctx.corr)
                vts = y_loads(kv_next, b, vts)
                ctx.o = gemm_mma(pv, vts, p, ctx.o)                                   # PV(j)
                softmax.head(ctx,
                             s_next,
                             kv_next,
                             masked,
                             ctx.b_next,
                             sched=lambda: spread(pv.n_mma, y_ops))                   # max(j+1)
                y_end(b)

            # The loop leaves the last tile's PV undone; V^T has one buffer, so no dispatch.
            def finish(kv_last):
                if const_expr(not vt_early):
                    gemm_stage(pv, v_load_plan, ctx.lds, scopes, ctx.v_regs, 0)
                vts = gemm_read(pv, vt_read_plan, ctx.lds, 0) if vt_early else None
                p = softmax.tail(ctx, kv_last)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()
                o = softmax.rescale(ctx.o, ctx.corr)
                o = gemm_mma(pv, vts if vt_early else gemm_read(pv, vt_read_plan, ctx.lds, 0), p, o)
                return [ctx.m, ctx.l] + o

            # Masked tiles run first and last, the clean middle unrolled x2 (flex_gemm.run_tiles).
            warmup()
            ctx.tiles(iteration, j_clean, j_clean_end, n_loop)
            results = finish(kv(n_loop))

            # ---- epilogue: O / l, LSE (l split over two lane halves)
            # O through LDS: whole 16 B row chunks per store.
            def store_o(accs, inv_l):
                # Every wave's prefetch DMAs must have landed (epilogue's s_waitcnt(0)) before
                # any slab is written.
                cluster_barrier()
                tensor_io.store_acc_staged(pv,
                                           accs,
                                           ctx.o_rsrc,
                                           ctx.o_row_stride,
                                           lds=ctx.lds,
                                           scope='vt',
                                           wave_base=wave * (WAVE_M * 2 * o_width),
                                           width=o_width,
                                           row0=q_wave,
                                           lane=lane,
                                           row_lim=sq,
                                           col_limit=ctx.o_col_limit,
                                           elem=elem,
                                           pad_floor=tr.pad_floor,
                                           scale=inv_l)

            softmax.epilogue(ctx,
                             pv,
                             results,
                             LSE,
                             row=q_row,
                             lane=lh,
                             lane_first=lambda: lh == fx.Index(0),
                             store=store_o if tr.staged_o else None)
            if const_expr(tr.staged_o and q_pass + 1 < len(q_tiles)):
                # The next pass's K DMAs overwrite other waves' O slabs.
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()

    @flyc.jit
    def launch_fmha_fwd_gfx942_wide(
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
        grid_y = (fx.Index(max_seqlen_q) + tr.grid_block_m - 1) // tr.grid_block_m
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

    return host.make_fwd_launch(launch_fmha_fwd_gfx942_wide,
                                tr,
                                causal_type,
                                host_causal_type,
                                bias,
                                dropout)
