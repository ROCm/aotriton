# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The wide dK/dV body (head dims 160, 192) on `flex_gemm`, `softmax` and `tensor_io`.

    S  = Q.K^T                  P = exp2(sm_scale*log2e*S + log2e*bias - lse2)
    dP = dO.V^T                 dS = P * (dP - Delta)
    dV^T += dO^T.P              dK^T += Q^T.dS;  dK = sm_scale * dK

Four waves, one a SIMD, each keeping 16 *
kv_subs keys of K and V in registers while 16-row Q/dO tiles stream through LDS
on 16x16x16 MFMAs. S and dP are `trait_ab` GEMMs on row-major (RM) tiles each
wave DMAs by whole rows (`dma='rows'`); dV and dK are `trait_tb` GEMMs on Q^T and
dO^T (T), built from the RM tiles through VGPRs with v_perm (`transposed_from`,
layout 'padded'): gfx942 has no transpose read. Two barrier-separated clusters
a tile, each one streamed pair of GEMMs whose hooks carry the other work:
    X(j): S(j+1), dP(j+1); hooks: P(j), dS(j) per key column, T(j+1)'s write
    Y(j): dV, dK += tile j; hooks: DMA of RM(j+3) by rows, LSE/Delta(j+2)
RM and T are double-buffered; LSE and Delta are DMA'd beside T. The policy's
band mask zeroes P after exp2 (`mask_at` 'prob'; 'score' costs ~1% causal and
moves f16 roundings). Same kernel ABI and launcher interface as the narrow body
in `dkdv.py`.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops

import fmha_utils as fmha
from . import flex_gemm
from . import host
from . import softmax
from . import tensor_io
from . import utils
from . import tuning_dkdv as tuning
from .flex_gemm import (
    gemm_dma_plan,
    gemm_dma_rows,
    gemm_read_plan,
    gemm_stream_joint,
    gemm_transpose_plan,
    gemm_transpose_read,
    gemm_transpose_write,
)
from .utils import (
    LOG2E,
    OOB_VOFF,
    WAVE,
    Lds,
    alias_scopes,
    apply_mask,
    base_addr,
    cluster_barrier,
    col_mask,
    deal_keep_bits,
    lds_ptr_at,
    m0_at,
    pack_operand,
    philox_plane,
    philox_stream,
    pin_values,
    rsrc_for,
    smax,
    smin,
    span_bytes,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942_wide.lds">'
LDS_SCOPES = ('rm0', 'rm1', 't0', 't1', 's0', 's1')
NUM_WAVES = 4
BM = 16                 # q rows per tile: the N of the 16x16x16 MFMA
SIDE_BYTES = 512        # LSE then Delta of one tile, 256 B per DMA issue


@dataclass(frozen=True)
class DkdvWideTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    kv_subs: int                # 16-key MFMA columns per wave
    waves_per_eu: int
    grid_axis_order: int
    bias: bool
    dropout: bool
    dropout_scale_at: str       # softmax.P_DROP_PER_ELEM or P_DROP_AT_STORE
    mask_at: str                # softmax.MASK_PROB or MASK_SCORE
    tb_pos: int                 # T(j+1)'s write among X's hooks: 0 first, else last
    prefetch: int = 2           # LDS reads in flight ahead of their MFMAs

    # make_dkdv_launch's names
    @property
    def D(self):  # noqa: N802
        return self.head_dim

    @property
    def PADDED(self):  # noqa: N802
        return self.padded

    @property
    def block_size(self):
        return NUM_WAVES * WAVE

    @property
    def rows_per_wave(self):
        return 16 * self.kv_subs

    @property
    def block_kv(self):
        return NUM_WAVES * self.rows_per_wave

    @property
    def pad_floor(self):
        return 0 if self.padded else self.head_dim

    @property
    def kv_vgpr(self):
        """Pin K, V to VGPRs; padded builds leave LLVM free (with both masks the pin
        can leave the register allocator no solution, and it aborts)."""
        return not self.padded

    @property
    def q_down(self):
        """Walk q tiles last to first: a causal head's blocks then all start on its last
        tiles and share them in L2, instead of each starting at its own diagonal."""
        return self.causal

    def gemms(self):
        """(S, dP, dV, dK) and the LDS plan: RM slots [Q | dO] at 0, then T slots
        [Q^T | dO^T | LSE, Delta]."""
        def rows(lds_base):
            return flex_gemm.trait_ab(self.dtype,
                                      BM,
                                      self.head_dim,
                                      self.pad_floor,
                                      self.block_size,
                                      lds_base,
                                      'rm',
                                      dma='rows',
                                      mfma='16x16x16',
                                      b_tiles=self.kv_subs,
                                      buf_stride=2 * rm_tile)

        def trans(lds_base, src):
            return flex_gemm.trait_tb(self.dtype,
                                      BM,
                                      self.head_dim,
                                      self.block_size,
                                      lds_base,
                                      't',
                                      layout='padded',
                                      mfma='16x16x16',
                                      transposed_from=src.a,
                                      b_tiles=self.kv_subs,
                                      buf_stride=t_slot)

        rm_tile = flex_gemm.RowA(BM, self.head_dim, 0, self.block_size, 0, 'rm', dma='rows').tile_bytes
        lds_t = 4 * rm_tile
        t_tile = flex_gemm.padded_t_bytes(BM, self.head_dim)
        t_slot = 2 * t_tile + SIDE_BYTES
        qs = rows(0)
        dops = rows(rm_tile)
        return qs, dops, trans(lds_t + t_tile, dops), trans(lds_t, qs)


def build_fmha_bwd_dkdv_gfx942_wide(head_dim,
                                         causal=False,
                                         causal_type=None,
                                         dtype_str='bf16',
                                         bias=False,
                                         dropout=False,
                                         kv_subs=None,
                                         waves_per_eu=1,
                                         grid_axis_order=None,
                                         padded_head=False,
                                         dropout_scale_at=None,
                                         mask_at=None):
    """build_fmha_bwd_dkdv_gfx942's interface and kernel ABI, for wide heads."""
    assert head_dim % 32 == 0 and 64 <= head_dim <= 192, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if kv_subs is None:
        kv_subs = tuning.body_kv_subs('wide', head_dim)
    assert kv_subs == 2, kv_subs
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    tr = DkdvWideTraits(head_dim=head_dim,
                        padded=padded_head,
                        causal=causal_type != 0,
                        dtype=dtype_str,
                        kv_subs=kv_subs,
                        waves_per_eu=waves_per_eu,
                        grid_axis_order=grid_axis_order,
                        bias=bias,
                        dropout=dropout,
                        dropout_scale_at=dropout_scale_at or tuning.default_dropout_scale_at('wide'),
                        mask_at=mask_at or tuning.default_mask_at('wide'),
                        # LLVM's register allocator aborted ("Overlapping insert") on
                        # padded + bias + dropout builds otherwise (bf16 hd224).
                        tb_pos=0 if padded_head and bias and dropout else 99)
    qs, dops, dvt, dkt = tr.gemms()
    side = dkt.a.lds_base + 2 * dkt.a.tile_bytes     # LSE/Delta area of T slot 0
    lds_bytes = dkt.a.lds_base + 2 * dkt.a.buf_stride
    assert lds_bytes <= 65536, lds_bytes
    tag = 'dkdv_wide_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    NSUB = tr.kv_subs
    NT = qs.reads                   # 16 B RM reads per tensor and tile
    n_acc = dkt.n_acc
    # Y waits for its RM DMAs (both tensors) and LSE/Delta's two to land a tile later.
    n_vm_y = qs.a.n_dma + dops.a.n_dma + 2

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    def scopes(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

    @flyc.kernel(known_block_size=[tr.block_size, 1, 1])
    # The kernel ABI is bwd_dkdv_kernel's (gfx1201), so aot/flyc_bwd_dkdv.py can serve both.
    def fmha_bwd_dkdv_gfx942_kernel(
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
        _ = tag
        # ---- prelude: lane roles, this block's kv tile and kv head, sequence bounds
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lc = lane % 16              # MFMA row/column: q row of A, key of B
        gq = lane // 16             # 4-element group of the MFMA's k axis
        if const_expr(tile_fastest):
            kv_tile = fx.Index(gpu.block_idx.x)
            head_k = fx.Index(gpu.block_idx.y)
        else:
            head_k = fx.Index(gpu.block_idx.x)
            kv_tile = fx.Index(gpu.block_idx.y)
        z_i32 = fx.Int32(gpu.block_idx.z)
        sq_i32, q_row_off, q_batch = fmha.decode_addressing(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32)
        sk_i32, k_row_off, k_batch = fmha.decode_addressing(varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32)
        group = fx.Int32(num_head_q) // fx.Int32(num_head_k)
        wave_s = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave)))

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_base = fx.Index(fx.ptrtoint(lds_p))
        lds_io = Lds(lds_p, elem, _LDS_DOMAIN, LDS_SCOPES)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)

        def masked8(v8, c0, hdim):
            return apply_mask(v8, col_mask(c0, hdim), elem)

        # ---- K and V: resident B operands, key kvw + 16*sub + lc, d = 32t + 8gq + [0, 8)
        k0 = fx.Int32(kv_tile) * fx.Int32(tr.block_kv)
        kvw = k0 + wave_s * fx.Int32(tr.rows_per_wave)
        k_rsrc = rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk_i32, row_qk)
        v_rsrc = rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk_i32, row_vo)

        def resident(rsrc, ss, hdim):
            out = []
            for t in range_constexpr(NT):
                row = []
                for sub in range_constexpr(NSUB):
                    r = fx.Index(kvw + fx.Int32(16 * sub)) + lc
                    c0 = fx.Index(32 * t) + gq * 8
                    v8 = buffer_ops.buffer_load(rsrc, r * fx.Index(fx.Int32(ss)) + c0, vec_width=8, dtype=elem)
                    if const_expr(tr.padded):
                        # Slack columns may hold NaN, and 0 * NaN is NaN.
                        v8 = masked8(v8, fx.Int32(c0), hdim)
                    row.append(Vec(v8))
                out.append(row)
            return out

        def load_res():
            """The resident K and V as B packs [block][read]."""
            k_res = resident(k_rsrc, stride_k_seq, fx.Int32(hdim_qk))
            v_res = resident(v_rsrc, stride_v_seq, fx.Int32(hdim_vo))
            if const_expr(tr.kv_vgpr):
                # Keep K and V out of the AGPRs, which the accumulators fill.
                flat = pin_values([x for row in k_res + v_res for x in row])
                n_k = NT * NSUB
                k_res = [[Vec(flat[t * NSUB + sub]) for sub in range(NSUB)] for t in range(NT)]
                v_res = [[Vec(flat[n_k + t * NSUB + sub]) for sub in range(NSUB)] for t in range(NT)]
            return ([[k_res[t][sub] for t in range(NT)] for sub in range(NSUB)],
                    [[v_res[t][sub] for t in range(NT)] for sub in range(NSUB)])

        # ---- the visited q range of this key block
        k_hi = smin(k0 + fx.Int32(tr.block_kv), sk_i32) - fx.Int32(1)
        if const_expr(tr.causal):
            win_l, win_r = fmha.resolve_window(window_left, window_right, sq_i32, sk_i32, varlen_bits)
            q_lo = smax(k0 - win_r, fx.Int32(0))
            q_hi = smin(k_hi + win_l, sq_i32 - fx.Int32(1))
        else:
            q_lo = fx.Int32(0)
            q_hi = sq_i32 - fx.Int32(1)
        alive = (k0 < sk_i32) & (q_lo <= q_hi)
        t0 = alive.select(q_lo, fx.Int32(0)) // fx.Int32(BM)
        n_tiles = alive.select(q_hi // fx.Int32(BM) - t0 + fx.Int32(1), fx.Int32(0))
        n_loop = smax(n_tiles - fx.Int32(1), fx.Int32(0))
        t_last = t0 + n_loop
        n_pairs = n_loop // fx.Int32(2)

        # This lane's keys see q rows [band_lo, band_lo + band_span]: one unsigned compare per element.
        # A block with no tile still runs tile 0 (prologue, finish) under an empty band.
        band_lo = []
        if const_expr(tr.causal):
            span = win_l + win_r
            empty = (span < fx.Int32(0)) | (n_tiles == fx.Int32(0))
            band_span = empty.select(fx.Int32(0), span)
            for sub in range_constexpr(NSUB):
                kv = kvw + fx.Int32(16 * sub) + fx.Int32(lc)
                band_lo.append(empty.select(fx.Int32(-(1 << 30)), kv - win_r))
        else:
            empty = n_tiles == fx.Int32(0)
            band_span = empty.select(fx.Int32(0), fx.Int32(1 << 30))
            band_lo = [empty.select(fx.Int32(-(1 << 30)), fx.Int32(0))] * NSUB

        # ---- LDS plans: RM reads, T reads, the T build (waves 0-1 Q, 2-3 dO), the RM DMA
        rm_plan = gemm_read_plan(qs, lc, gq)
        t_rd = gemm_read_plan(dkt, lc, gq)
        t_plan = gemm_transpose_plan(dkt, tid, None, wave=wave_s)
        side_rd = fx.Index(side) + gq * 16

        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        m0_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_base + fx.Index(qs.a.lds_base))))
        dma_plan = gemm_dma_plan(qs, wave_s, lane, None, m0_base)

        zero4 = fx.full(4, 0.0, fx.Float32)
        if const_expr(dropout):
            rng, ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            assert rng.randoms_per_offset == 4

        ctx = flex_gemm.Ctx(tr, carried=('dk', 'dv', 's', 'dp', 'bias'))
        ctx.dropout_scale = dropout_scale
        # Q rows, for softmax.lse_rows.
        ctx.set_q_rows(varlen_bits=varlen_bits,
                       q_batch=q_batch,
                       q_row_off=q_row_off,
                       sq=sq_i32,
                       max_seqlen_q=max_seqlen_q,
                       seqinfo_q0=seqinfo_q0,
                       seqinfo_q1=seqinfo_q1,
                       num_seqlens=num_seqlens,
                       num_head_q=num_head_q)
        lse_tokens = softmax.lse_tokens(ctx)

        def cluster_end(name):
            """LDS reads land before the next cluster overwrites them (s_barrier does not wait)."""
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier(name)

        # ---- per q head of this kv head's group: dK/dV accumulate over all of them
        def head_pass(g, accs):
            """Every q tile of query head head_k * group + g, accumulated into accs."""
            head_q = fx.Index(fx.Int32(head_k) * group + g)
            q_base = base_addr(Q, q_batch, head_q, q_row_off, stride_q_batch, stride_q_head, stride_q_seq)
            do_base = base_addr(DO, q_batch, head_q, q_row_off, stride_do_batch, stride_do_head, stride_do_seq)
            q_span = span_bytes(sq_i32, stride_q_seq, row_qk)
            do_span = span_bytes(sq_i32, stride_do_seq, row_vo)
            q_rs_b = fx.Int32(stride_q_seq) * fx.Int32(2)
            do_rs_b = fx.Int32(stride_do_seq) * fx.Int32(2)
            q_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(q_base, num_records_bytes=q_span))
            do_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(do_base,
                                                                                        num_records_bytes=do_span))
            lse_row0, lse_pitch = softmax.lse_rows(ctx, head=head_q, tokens=lse_tokens)
            pitch_b = fx.Int32(lse_pitch) * fx.Int32(4)
            side_bytes = fx.Int64(fx.Index(sq_i32) * lse_pitch) * fx.Int64(4)
            lse_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(LSE)) + fx.Int64(lse_row0) * fx.Int64(4), num_records_bytes=side_bytes))
            delta_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(Delta)) + fx.Int64(lse_row0) * fx.Int64(4), num_records_bytes=side_bytes))
            # Side DMA: lane L < 16 copies row q0 + L; the rest read 0.
            side_dma_voff = (lane < fx.Index(BM)).select(fx.Int32(lane) * pitch_b, fx.Int32(OOB_VOFF))

            if const_expr(bias):
                b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head_q) * stride_b_head
                           + fx.Int64(q_row_off) * stride_b_seq_q)
                b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(k_row_off + sk_i32),
                              fx.Int64(0))
                b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2), num_records_bytes=b_span * fx.Int64(2)))
                b_rs_b = fx.Int32(stride_b_seq_q) * fx.Int32(2)
                b_voff = [fx.Int32(gq * 4) * b_rs_b + (k_row_off + kvw + fx.Int32(16 * sub + lc)) * fx.Int32(2)
                          for sub in range(NSUB)]

            if const_expr(dropout):
                ph_base, ph_stride = philox_plane(rng, ph_off, z_i32, num_head_q, head_q, max_seqlen_q, max_seqlen_k)

            def q0_of(j):
                if const_expr(tr.q_down):
                    return (t_last - fx.Int32(j)) * fx.Int32(BM)
                return (t0 + fx.Int32(j)) * fx.Int32(BM)

            # ---- memory: RM DMA by rows, LSE/Delta beside T, bias to VGPRs
            def rows_dma(j, slot):
                """RM(j) -> slot, one callable per row: wave w copies rows 4w..4w+3 of Q, then of dO."""
                q0 = q0_of(j)
                return (gemm_dma_rows(qs, dma_plan, q_rsrc, scopes, q0, q_rs_b, slot)
                        + gemm_dma_rows(dops, dma_plan, do_rsrc, scopes, q0, do_rs_b, slot))

            def dma(j, slot):
                for f in rows_dma(j, slot):
                    f()

            def side_dma(j, slot):
                """LSE(j) and Delta(j) -> T[slot]'s side area, 256 B each."""
                soff = q0_of(j) * pitch_b
                for i, rsrc in enumerate((lse_rsrc, delta_rsrc)):
                    m0 = fx.Int64(m0_at(m0_base, side + slot * dkt.a.buf_stride + 256 * i))
                    lds_io.tag(rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                             lds_ptr_at(m0),
                                                             fx.Int32(4),
                                                             side_dma_voff,
                                                             soff,
                                                             fx.Int32(0),
                                                             fx.Int32(0)),
                               f's{slot}')

            def side_read(slot):
                """LSE and Delta of q = q0 + 4gq + r, from T[slot]'s side area."""
                v4i = Vec.make_type(4, fx.Int32)
                lse = Vec(lds_io.load(side_rd + slot * dkt.a.buf_stride, v4i, 16, f's{slot}')).bitcast(fx.Float32)
                dlt = Vec(lds_io.load(side_rd + (slot * dkt.a.buf_stride + 256), v4i, 16, f's{slot}')).bitcast(fx.Float32)
                return [lse[r] for r in range(4)], [dlt[r] for r in range(4)]

            def bias_load(j):
                """Bias of tile j at q = q0 + 4gq + r, key kvw + 16 sub + lc: one i16 each."""
                if const_expr(not bias):
                    return []
                bsoff = q0_of(j) * b_rs_b
                out = []
                for sub in range_constexpr(NSUB):
                    for r in range_constexpr(4):
                        h = rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, b_voff[sub] + fx.Int32(r) * b_rs_b, bsoff, 0)
                        out.append(fx.Int32(fx.Int16(h)))
                return out

            def t_build(slot):
                """T[slot] from RM[slot]: the unit's rows now, its v_perm'd writes as a hook."""
                regs = gemm_transpose_read(dkt, t_plan, lds_io, slot)
                return lambda: gemm_transpose_write(dkt, t_plan, lds_io, scopes, regs, slot)

            # ---- score hooks: slack columns, the band mask, bias, dropout keep bits
            def slack(g, t, a8):
                """RM read t of S (g 0) or dP (g 1) with its slack columns zeroed."""
                if const_expr(not tr.padded):
                    return a8
                c0 = fx.Int32(32 * t) + fx.Int32(gq * 8)
                return Vec(masked8(a8, c0, fx.Int32(hdim_vo if g else hdim_qk)))

            def keep_bits(q0, kv4):
                """Dropout keep of (q0 + 4gq + r, kv4 + lc % 4) for r < 4: lane (lc, gq) draws row
                4gq + lc % 4 for keys kv4 + [0, 4), and a DPP broadcast deals its quad the bits."""
                row = q0 + fx.Int32(gq * 4) + fx.Int32(lc % 4)
                return deal_keep_bits(rng, ph_seed, ph_base, ph_stride, row, kv4, idropout_p, fx.Int32(lc % 4))

            def p_ds(j, masked, sub, s_raw, dp_raw, bvals, lse, delta):
                """P (for dV) and dS packs of tile j, key column `sub`."""
                lse2 = [fx.Float32(x) * fx.Float32(LOG2E) for x in lse]
                q0 = q0_of(j)
                sv = Vec(s_raw[sub]) * fx.full(4, fx.Float32(c_scale), fx.Float32)
                dpv = Vec(dp_raw[sub])
                mask_of = None
                if const_expr(masked):
                    rel0 = q0 + fx.Int32(gq * 4) - band_lo[sub]

                    def mask_of(v, r, fill):
                        live = fx.Uint32(rel0 + fx.Int32(r)) <= fx.Uint32(band_span)
                        return live.select(v, fill)
                keep = None
                if const_expr(dropout):
                    keep = keep_bits(q0, kvw + fx.Int32(16 * sub) + fx.Int32(4) * fx.Int32(lc // 4))

                def bias_of(r):
                    bv = Vec.from_elements([fx.Int16(bvals[4 * sub + r])], fx.Int16).bitcast(elem)
                    return fx.Float32(Vec(bv)[0].to(fx.Float32))
                ps, dss = softmax.p_ds_elems(ctx,
                                             sv,
                                             dpv,
                                             lse_l2=lse2,
                                             delta=delta,
                                             bias_of=bias_of,
                                             mask_of=mask_of,
                                             keep=keep)
                return pack_operand(ps, elem), pack_operand(dss, elem)

            # ---- the tile loop: RM slot j % 2 holds tile j+1's Q/dO when X(j) reads it,
            # T slot j % 2 tile j's Q^T/dO^T; `b` = j % 2 is a Python int.
            def iteration(j, b, masked, last=False):
                s_cur, dp_cur, bias_cur = ctx.s, ctx.dp, ctx.bias
                lse, delta = side_read(b)
                bias_next = [] if const_expr(last) else bias_load(j + 1)
                rocdl.sched_barrier(0)
                # X(j): S(j+1) = Q.K^T, dP(j+1) = dO.V^T streamed from RM; P(j), dS(j) per
                # key column and T(j+1)'s write run as the stream's hooks.
                pd = [None] * NSUB
                hooks = [lambda sub=sub: pd.__setitem__(sub, p_ds(j, masked, sub, s_cur, dp_cur, bias_cur, lse, delta))
                         for sub in range(NSUB)]
                if const_expr(last):
                    for h in hooks:
                        h()
                else:
                    hooks.insert(min(tr.tb_pos, NSUB), t_build(1 - b))
                    ctx.s, ctx.dp = gemm_stream_joint((qs, dops),
                                                      rm_plan,
                                                      lds_io,
                                                      1 - b,
                                                      (k_packs, v_packs),
                                                      ([zero4] * NSUB, [zero4] * NSUB),
                                                      tr.prefetch,
                                                      hooks,
                                                      a_mod=slack)
                cluster_end(None if last else f'X{b}')
                # Y(j): dV^T += dO^T.P, dK^T += Q^T.dS of tile j, streamed from T; the
                # DMAs of RM(j+3) (row by row) and LSE/Delta(j+2) run as its hooks.
                dmas = [] if const_expr(last) else rows_dma(j + 3, 1 - b) + [lambda: side_dma(j + 2, b)]
                ctx.dv, ctx.dk = gemm_stream_joint((dvt, dkt),
                                                   t_rd,
                                                   lds_io,
                                                   b,
                                                   ([x[0] for x in pd], [x[1] for x in pd]),
                                                   (ctx.dv, ctx.dk),
                                                   tr.prefetch,
                                                   dmas,
                                                   per_step=2)
                if const_expr(not last):
                    rocdl.s_waitcnt(vmcnt=n_vm_y)
                    bias_next = pin_values(bias_next)
                cluster_end(None if last else f'Y{b}')
                ctx.bias = bias_next

            def step(j, carried, b, masked, last=False):
                ctx.restore(carried)
                iteration(j, b, masked, last)
                return ctx.dk + ctx.dv if const_expr(last) else ctx.carry()

            @flyc.jit
            def odd_tail(j, carried):
                out = list(carried)
                if j < fx.Index(n_loop):
                    out = step(j, carried, 0, tr.causal)
                return out

            @flyc.jit
            def finish(j, carried):
                out = list(carried[:2 * n_acc])
                if j % fx.Index(2) == fx.Index(0):
                    out = step(j, carried, 0, True, last=True)
                else:
                    out = step(j, carried, 1, True, last=True)
                return out

            # Prologue: RM(0), RM(1) DMA; X(-1) = S(0), dP(0), T(0); Y(-1) = DMA RM(2).
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            dma(0, 0)
            side_dma(0, 0)
            b0 = bias_load(0)
            rocdl.sched_barrier(0)
            dma(1, 1)
            rocdl.s_waitcnt(vmcnt=qs.a.n_dma + dops.a.n_dma)
            cluster_barrier()
            s0, dp0 = gemm_stream_joint((qs, dops),
                                        rm_plan,
                                        lds_io,
                                        0,
                                        (k_packs, v_packs),
                                        ([zero4] * NSUB, [zero4] * NSUB),
                                        tr.prefetch,
                                        a_mod=slack)
            t_build(0)()
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            dma(2, 0)
            side_dma(1, 1)
            rocdl.s_waitcnt(vmcnt=n_vm_y)
            cluster_barrier()

            # One body for every tile: masking costs a few VALU an element, a second
            # (masked) body costs registers. Pairs from j = 0, then at most one even j,
            # then the last tile.
            ctx.dk = list(accs[:n_acc])
            ctx.dv = list(accs[n_acc:])
            ctx.s, ctx.dp, ctx.bias = s0, dp0, b0
            init = ctx.carry()
            res1 = init
            for jp, carried in range(0, fx.Index(n_pairs), 1, init=init):
                mid = step(jp * 2, carried, 0, tr.causal)
                res1 = yield step(jp * 2 + 1, mid, 1, tr.causal)
            res2 = odd_tail(fx.Index(n_pairs) * 2, res1)
            return finish(fx.Index(n_loop), res2)

        k_packs, v_packs = load_res()
        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range(2 * n_acc)]
        results = acc0
        for g, carried in range(0, fx.Index(group), 1, init=acc0):
            results = yield head_pass(fx.Int32(g), carried)
        rocdl.s_waitcnt(vmcnt=0)

        # ---- epilogue: dK = sm_scale * dK^T, then dV (times 1/(1-p) if dropout scales it here)
        results = list(results)

        def dk_scale():
            # dS is against the scaled scores: sm_scale lands once, here.
            return fx.full(4, fx.Float32(sm_scale), fx.Float32)

        dv_scale = None
        if const_expr(dropout and tr.dropout_scale_at == softmax.P_DROP_AT_STORE):
            dv_scale = lambda: fx.full(4, fx.Float32(dropout_scale), fx.Float32)  # noqa: E731
        tensor_io.store_acc_t_by_tensor(dkt,
                                        [(results[:n_acc],
                                          lambda: rsrc_for(DK,
                                                           k_batch,
                                                           head_k,
                                                           k_row_off,
                                                           stride_dk_batch,
                                                           stride_dk_head,
                                                           stride_dk_seq,
                                                           sk_i32,
                                                           row_qk),
                                          lambda: fx.Index(fx.Int32(stride_dk_seq)),
                                          row_qk,
                                          dk_scale),
                                         (results[n_acc:],
                                          lambda: rsrc_for(DV,
                                                           k_batch,
                                                           head_k,
                                                           k_row_off,
                                                           stride_dv_batch,
                                                           stride_dv_head,
                                                           stride_dv_seq,
                                                           sk_i32,
                                                           row_vo),
                                          lambda: fx.Index(fx.Int32(stride_dv_seq)),
                                          row_vo,
                                          dv_scale)],
                                        kvw,
                                        lc,
                                        gq,
                                        sk_i32,
                                        elem)

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx942_wide(
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
        grid_x = (fx.Index(max_seqlen_k) + tr.block_kv - 1) // tr.block_kv
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_bwd_dkdv_gfx942_kernel(Q,
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
                                    value_attrs={'rocdl.waves_per_eu': tr.waves_per_eu,
                                                 'rocdl.flat_work_group_size':
                                                 f'{tr.block_size},{tr.block_size}'}).launch(
            grid=(grid_x, fx.Index(num_head_k), grid_z) if tile_fastest else (fx.Index(num_head_k), grid_x, grid_z),
            block=(tr.block_size, 1, 1),
            stream=stream)

    _launch = host.make_dkdv_launch(launch_fmha_bwd_dkdv_gfx942_wide,
                                    tr,
                                    causal_type,
                                    host_causal_type,
                                    bias,
                                    dropout)
    _launch.varlen_bits = fmha.varlen_bits
    _launch.varlen_compact = fmha.varlen_compact
    _launch.varlen_padded = fmha.varlen_padded
    _launch.varlen_strided = fmha.varlen_strided
    return _launch
