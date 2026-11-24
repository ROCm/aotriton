# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The xwide forward (head dims 384, 512) on `flex_gemm` and the `softmax` hooks.

A schedule over GEMM phases and softmax hooks: 16 query rows a wave on
v_mfma_f32_16x16x16, so Q^T + O^T fit 256 VGPRs at hd512 and two waves share a
SIMD; 16-key tiles, K and V^T double-buffered fill the 64 KB LDS. Neither tile fits in registers, so both GEMMs stream their
A from LDS (`gemm_stream`). One cluster and one barrier per tile. Lane values
the loop needs (row, key group, exchange addresses) are rebuilt at their use
behind asm, so nothing but Q^T and O^T lives across the loop.
Same kernel ABI and launcher interface as the narrow body in `fwd.py`.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as raw

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
    gemm_fix,
    gemm_fix_plan,
    gemm_load,
    gemm_load_plan,
    gemm_read_plan,
    gemm_stage,
    gemm_stream,
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
    row_band,
    rsrc_for,
    smax,
    smin,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_fwd_gfx942_xwide.lds">'
LDS_SCOPES = ('k0', 'k1', 'vt0', 'vt1')
WAVE_M = 16             # query rows per wave: the N of the 16x16x16 MFMA
BLOCK_N = 16            # keys per KV tile: the M of the QK MFMA
PREFETCH = 2            # LDS operand reads issued ahead of their MFMAs


@dataclass(frozen=True)
class FwdXwideTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    num_waves: int
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
        return 0 if self.padded else self.head_dim

    def gemms(self):
        """(QK, PV) on 16x16x16: K rows XOR-swizzled by DMA; V^T in chunks by v_perm."""
        qk = flex_gemm.trait_ab(self.dtype,
                                BLOCK_N,
                                self.head_dim,
                                self.pad_floor,
                                self.block_size,
                                0,
                                'k',
                                dma='xor',
                                mfma='16x16x16')
        pv = flex_gemm.trait_tb(self.dtype,
                                BLOCK_N,
                                self.head_dim,
                                self.block_size,
                                2 * qk.a.tile_bytes,
                                'vt',
                                layout='chunks',
                                mfma='16x16x16')
        return qk, pv


def build_fmha_fwd_gfx942_xwide(head_dim,
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
    """build_fmha_fwd_gfx942's interface and kernel ABI, for head dims 384 and 512.
    A causal build runs q tiles longest first."""
    assert head_dim % 128 == 0 and 256 <= head_dim <= 512, head_dim
    assert num_waves == 8, num_waves
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if block_n not in (None, BLOCK_N):
        raise ValueError(f'the xwide forward runs {BLOCK_N}-key tiles, got block_n={block_n}')
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    tr = FwdXwideTraits(head_dim=head_dim,
                        padded=padded_head,
                        causal=causal_type != 0,
                        dtype=dtype_str,
                        num_waves=num_waves,
                        waves_per_eu=waves_per_eu,
                        rescale_threshold=rescale_threshold,
                        grid_axis_order=grid_axis_order,
                        bias=bias,
                        dropout=dropout)
    # The grid extent the tuning policy records.
    assert tr.block_m == tuning.block_m('xwide', num_waves, tr.causal), tr.block_m
    qk, pv = tr.gemms()
    lds_bytes = 2 * qk.a.tile_bytes + 2 * pv.a.tile_bytes
    assert lds_bytes <= 65536, lds_bytes
    assert qk.a.tile_bytes * 2 % 1024 == 0
    tag = 'fwd_xwide_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    n_kdma = qk.a.n_dma
    n_v = pv.a.n_load
    n_o = pv.out_tiles

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    def scopes(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

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
        # `carried`: only the softmax state and O^T cross the loop (V lives within a tile).
        ctx = flex_gemm.Ctx(tr, carried=('m', 'corr', 'l', 's', 'o'))
        ctx.qk, ctx.pv = qk, pv

        # ---- block geometry: lr is the lane's query row (and K / V^T row in reads),
        # lg its 4-element group of the MFMA's k axis.
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lr = lane % 16
        lg = lane // 16
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

        # LDS: K double-buffered at 0, V^T double-buffered after it; 64 KB at hd512.
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
            rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk_i32, row_qk))
        v_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk_i32, row_vo))
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

        q_start = q_tile * tr.block_m
        q_row = q_start + wave * WAVE_M + lr

        # Q^T in VGPRs (slack columns zeroed; K's are zeroed in LDS). Then the
        # per-lane plans: K DMA, the padded fix, V loads, the streamed reads.
        ctx.q_packs = flex_gemm.resident_packs(qk, q_rsrc, q_row, q_row_stride, lg, hdim_qk)
        k_dma_plan = gemm_dma_plan(qk, wave, lane, k_stride_b, lds_addr)
        wave_s = k_dma_plan[0]
        if const_expr(tr.padded):
            fix_plan = gemm_fix_plan(qk, lane, lr, wave, hdim_qk)
        v_load_plan = gemm_load_plan(pv, tid, v_stride_b)
        k_read_plan = gemm_read_plan(qk, lr, lg)
        vt_read_plan = gemm_read_plan(pv, lr, lg, lane=lane)

        def k_fix(buf):
            if const_expr(tr.padded):
                gemm_fix(qk, fix_plan, ctx.lds, scopes, buf)

        ctx.c_scale = fx.Float32(sm_scale) * fx.Float32(softmax.LOG2E)
        ctx.c_neg_inf = fx.Float32(float('-inf'))
        c_neg_max = fx.Float32(-3.4028234663852886e38)

        # Lane values rebuilt at their use from vt_read_plan (lane * 16) behind asm
        # LLVM cannot hoist: kept live across the loop they would cost VGPRs.
        def peer_addr(mask):
            return fx.Int32(llvm.inline_asm(T.i32,
                                            [raw(fx.Int32(vt_read_plan))],
                                            f'v_lshrrev_b32 $0, 2, $1\n\tv_xor_b32 $0, {mask * 4}, $0',
                                            '=&v,v',
                                            has_side_effects=True))

        def lane_field(shift, width):
            return fx.Int32(llvm.inline_asm(T.i32,
                                            [raw(fx.Int32(vt_read_plan))],
                                            f'v_bfe_u32 $0, $1, {4 + shift}, {width}',
                                            '=v,v',
                                            has_side_effects=True))

        def row_now():
            """q_row, rebuilt."""
            return fx.Int32(q_start) + wave_s * fx.Int32(WAVE_M) + lane_field(0, 4)

        ctx.peer_addr = peer_addr
        ctx.lane_key = lambda: lane_field(4, 2) * fx.Int32(4)       # 4 * lg

        # Tiles: iteration j masks tile j+1. Every loop runs pairs, so an odd n_loop
        # gets one more iteration whose tile lies past the keys: all masked, P = 0.
        # (A parity branch instead made LLVM spill O around it.)
        win_l, win_r = band_window(tr.causal, window_left, window_right, sq_i32, sk_i32, varlen_bits)
        q_first, q_last, kv_base, n_tiles = kv_tiles(q_start, tr.block_m, BLOCK_N, sq_i32, sk_i32, win_l, win_r)
        ctx.band = lambda: row_band(row_now(), win_l, win_r, sk_i32)
        lead, clean_end = clean_tiles(q_first, q_last, kv_base, BLOCK_N, sk_i32, win_l, win_r)
        if const_expr(bias):
            clean_end = smin(clean_end, smax(n_tiles, fx.Index(1)) - fx.Index(1))
        n_loop = (n_tiles > fx.Index(1)).select(n_tiles - fx.Index(1), fx.Index(0))
        n_iter = n_loop + n_loop % 2
        j_clean = smin(smax(lead, fx.Index(1)) - fx.Index(1), n_loop)
        j_clean = j_clean + j_clean % 2
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
            ctx.b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + fx.Int32(lg * 4)) * fx.Int32(2)
            ctx.b_lane = ctx.b_byte & fx.Int32(-4)
            ctx.b_shift = (ctx.b_byte & fx.Int32(3)) * fx.Int32(8)

        def n_bias_loads(exact):
            return (4 if exact else 1) if bias else 0

        if const_expr(dropout):
            # The Triton kernels' stream, element for element (philox.py, grid_plane).
            ctx.rng, ctx.ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            fmha.philox_report(philox_seed_output, philox_offset_output, ctx.ph_seed, ph_off)
            ph_base, ph_stride = philox_plane(ctx.rng, ph_off, z_i32, num_head_q, head, max_seqlen_q, max_seqlen_k)
            # The wave's part of the offset is uniform; the lane adds row * stride + col // 4.
            ctx.ph_row = ctx.rng.grid_offset(ph_base, ph_stride, fx.Int32(q_start) + wave_s * fx.Int32(WAVE_M), 0)
            ph_stride32 = fx.Int32(ph_stride)
            ctx.idropout_p = idropout_p

            def zext64(x):
                return fx.Int64(arith.extui(T.i64, raw(fx.Int32(x))))

            # One v_mad_u64_u32: zero-extended row * stride + col // 4.
            ctx.ph_lane = lambda: zext64(lane_field(0, 4)) * zext64(ph_stride32) + zext64(lane_field(4, 2))

        def kv(j):
            return kv_base + j * BLOCK_N

        # Iteration -1: K(0), K(1), V(0) land; QK(0), head(0).
        def warmup():
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base, k_stride_b, 0, hdim=hdim_qk)
            # The vmcnt below counts everything after K(0) as newer: keep it after the DMA.
            rocdl.sched_barrier(0)
            b0 = softmax.bias_load(ctx, kv_base, True)
            v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_base, v_stride_b)
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_base + BLOCK_N, k_stride_b, 1, hdim=hdim_qk)
            rocdl.s_waitcnt(vmcnt=n_bias_loads(True) + n_v + n_kdma)
            k_fix(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            gemm_stage(pv, v_load_plan, ctx.lds, scopes, v_regs, 0)
            s_raw = gemm_stream(qk, k_read_plan, ctx.lds, 0, ctx.q_packs, ahead=PREFETCH)
            ctx.m = c_neg_max               # finite: fully masked rows stay NaN-free
            softmax.head(ctx, s_raw, kv_base, True, b0)
            rocdl.s_waitcnt(vmcnt=0)
            k_fix(1)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            ctx.l = fx.Float32(0.0)
            ctx.o = [fx.full(4, 0.0, fx.Float32) for _ in range(n_o)]

        # ---- the tile loop, one cluster per tile: DMA K(j+2) -> k[j%2], exp(j),
        # QK(j+1) from k[(j+1)%2], load V(j+1), PV(j) from vt[j%2], max(j+1),
        # V^T(j+1) -> vt[(j+1)%2], wait K(j+2). With no stagger, one barrier
        # separates every producer from its consumer.
        def iteration(j, b, masked):
            kv_cur = kv(j)
            kv_next = kv_cur + BLOCK_N
            b_next = softmax.bias_load(ctx, kv_next, masked)
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_next + BLOCK_N, k_stride_b, b, hdim=hdim_qk)
            ctx.o = softmax.rescale(ctx.o, ctx.corr, 4)
            p = softmax.tail(ctx, kv_cur)                                               # P(j)
            s_next = gemm_stream(qk, k_read_plan, ctx.lds, 1 - b, ctx.q_packs, ahead=PREFETCH)   # QK(j+1)
            # V(j+1) lives only through PV(j): loaded after QK(j+1), stored at the end.
            rocdl.sched_barrier(0)
            v_regs = gemm_load(pv, v_load_plan, v_rsrc, kv_next, v_stride_b)
            ctx.o = gemm_stream(pv, vt_read_plan, ctx.lds, b, p[0], ctx.o, ahead=PREFETCH)   # PV(j)
            softmax.head(ctx, s_next, kv_next, masked, b_next)                         # max(j+1)
            gemm_stage(pv, v_load_plan, ctx.lds, scopes, v_regs, 1 - b)
            rocdl.s_waitcnt(vmcnt=0)
            k_fix(b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()

        # The last tile's exp and PV; its V^T is already in vt[0] (n_iter is even).
        def finish(kv_last):
            o = softmax.rescale(ctx.o, ctx.corr, 4)
            p = softmax.tail(ctx, kv_last)
            o = gemm_stream(pv, vt_read_plan, ctx.lds, 0, p[0], o, ahead=PREFETCH)
            return [ctx.m, ctx.l] + o

        warmup()
        ctx.tile_pairs(iteration, j_clean, j_clean_end, n_iter)
        results = finish(kv(n_iter))

        # ---- epilogue: O / l, LSE (l split over four lane groups)
        # q_row and the lane group are rebuilt where used, not kept live.
        softmax.epilogue(ctx,
                         pv,
                         results,
                         LSE,
                         row=lambda: fx.Index(row_now()),
                         lane=lambda: lane_field(4, 2),
                         lane_first=lambda: lane_field(4, 2) == fx.Int32(0))

    @flyc.jit
    def launch_fmha_fwd_gfx942_xwide(
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

    return host.make_fwd_launch(launch_fmha_fwd_gfx942_xwide,
                                tr,
                                causal_type,
                                host_causal_type,
                                bias,
                                dropout)
