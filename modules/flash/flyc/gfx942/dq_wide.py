# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The wide dQ/dB body (head dims 160..512) on `flex_gemm`, `softmax` and `tensor_io`.

    S^T  = K . Q^T         A = K rows (LDS),   B = Q^T (registers)
    dP^T = V . dO^T        A = V rows (LDS),   B = dO^T (registers)
    dQ^T += K^T . dS^T     A = K^T (LDS),      B = dS^T, the S^T accumulator repacked

4 waves, one per SIMD, each owning WAVE_M query
rows; WAVE_M 32 on 32x32x8 MFMAs up to head dim 256, 16 on 16x16x16 above. K
and V come in through VGPRs (`dma='stage'`): K is written row-major and, a
cluster later, as K^T (v_perm); V row-major. Single buffers, two clusters a tile:
    A(j): write K^T(j); load K(j+1), bias(j+1); S^T(j), dP^T(j); Philox(j)
    B(j): write V(j+1); load V(j+2); read K^T(j); dS(j); dQ^T += K^T(j) dS^T(j);
          write K(j+1)
P's exponent is one fma (softmax.P_EXP_FMA). Causal blocks run q tiles g and
T-1-g. Same kernel ABI and launcher interface as the narrow body in `dq.py`.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

import fmha_utils as fmha
from . import flex_gemm
from . import host
from . import softmax
from . import tensor_io
from . import utils
from . import tuning_dq as tuning
from .flex_gemm import (
    gemm_load,
    gemm_load_plan,
    gemm_mma,
    gemm_read,
    gemm_read_plan,
    gemm_stage,
)
from .utils import (
    LOG2E,
    WAVE,
    Lds,
    alias_scopes,
    band_window,
    bias_slab,
    clean_tiles,
    cluster_barrier,
    funnel_words,
    kv_tiles,
    philox_plane,
    philox_stream,
    row_band,
    rsrc_for,
    smax,
    smin,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dq_gfx942_wide.lds">'
LDS_SCOPES = ('k', 'v', 'kt')


@dataclass(frozen=True)
class DqWideTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    wave_m: int
    waves_per_eu: int
    grid_axis_order: int
    bias: bool
    dropout: bool
    # Schedule constants (measured).
    valu_per_mfma: int = 3
    lead_a: int = 4             # LDS reads issued ahead of the S^T/dP^T MFMAs
    lead_b: int = 4             # LDS reads issued ahead of the dQ^T MFMAs

    # make_dq_launch's names
    @property
    def D(self):  # noqa: N802
        return self.head_dim

    @property
    def PADDED(self):  # noqa: N802
        return self.padded

    @property
    def num_waves(self):
        return 4

    @property
    def block_size(self):
        return self.num_waves * WAVE

    @property
    def block_m(self):
        return self.num_waves * self.wave_m

    @property
    def block_n(self):
        """One S^T MFMA tile per KV tile."""
        return self.wave_m

    @property
    def pair_tiles(self):
        """Causal blocks run q tiles g and T-1-g: every block then does the same work."""
        return self.causal

    @property
    def grid_block_m(self):
        return self.block_m * (2 if self.pair_tiles else 1)

    @property
    def mfma(self):
        return '32x32x8' if self.wave_m == 32 else '16x16x16'

    @property
    def swizzle(self):
        """Row-major chunk XOR key (row // div) % mod, measured conflict-free for the MFMA reads."""
        return (8, 2) if self.wave_m == 32 else (4, 4)

    @property
    def philox_a(self):
        """Philox VALU spread over A(j)'s MFMAs (B2 H32 S4k hd192 3.52 -> 3.27 ms)."""
        return 300 if self.wave_m == 32 else 100

    @property
    def pad_floor(self):
        """Columns below this are real in every call this build serves."""
        return 0 if self.padded else self.head_dim

    def gemms(self):
        """(S, dP, dQ): K and V row-major, staged from VGPRs; K^T by v_perm. One buffer each."""
        qk = flex_gemm.trait_ab(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.pad_floor,
                                self.block_size,
                                0,
                                'k',
                                dma='stage',
                                mfma=self.mfma,
                                buffers=1,
                                swizzle=self.swizzle)
        vdo = flex_gemm.trait_ab(self.dtype,
                                 self.block_n,
                                 self.head_dim,
                                 self.pad_floor,
                                 self.block_size,
                                 qk.a.tile_bytes,
                                 'v',
                                 dma='stage',
                                 mfma=self.mfma,
                                 buffers=1,
                                 swizzle=self.swizzle)
        kt = flex_gemm.trait_tb(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.block_size,
                                2 * qk.a.tile_bytes,
                                'kt',
                                buffers=1,
                                unit_cols=8,
                                mfma=self.mfma)
        return qk, vdo, kt


def build_fmha_bwd_dq_gfx942_wide(head_dim,
                                       causal=False,
                                       causal_type=None,
                                       dtype_str='bf16',
                                       bias=False,
                                       dropout=False,
                                       num_waves=4,
                                       block_n=None,
                                       waves_per_eu=1,
                                       grid_axis_order=None,
                                       padded_head=False,
                                       wave_m=None):
    """build_fmha_bwd_dq_gfx942's interface and kernel ABI, for wide heads."""
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    wave_m = (32 if head_dim <= 256 else 16) if wave_m is None else wave_m
    assert wave_m in (16, 32), wave_m
    assert head_dim % 32 == 0 and 64 <= head_dim <= (256 if wave_m == 32 else 512), head_dim
    if grid_axis_order is None:
        grid_axis_order = tuning.GRID_AXIS_TILE_FASTEST
    tr = DqWideTraits(head_dim=head_dim,
                      padded=padded_head,
                      causal=causal_type != 0,
                      dtype=dtype_str,
                      wave_m=wave_m,
                      waves_per_eu=waves_per_eu,
                      grid_axis_order=grid_axis_order,
                      bias=bias,
                      dropout=dropout)
    if num_waves != 4 or block_n not in (None, tr.block_n):
        raise ValueError(f'this body runs 4 waves and {tr.block_n}-key tiles, got {num_waves}, {block_n}')
    # The grid extent the tuning policy records.
    assert tr.grid_block_m == tuning.block_m('wide', num_waves, tr.causal, wave_m), tr.grid_block_m
    assert tr.block_n == tuning.wide_block_n(wave_m), tr.block_n
    qk, vdo, kt = tr.gemms()
    lds_bytes = 2 * qk.a.tile_bytes + kt.a.tile_bytes
    assert lds_bytes <= 65536, lds_bytes
    tag = 'dq_wide_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    block_n = tr.block_n
    wm = tr.wave_m
    g_lanes = WAVE // wm                # lane groups: lane = group * WAVE_M + row
    ns = qk.ns                          # S^T values per lane

    # A(j) spreads its LDS reads and Philox VALU over the S^T and dP^T MFMAs.
    a_mfma = qk.n_mma + vdo.n_mma
    a_ops = flex_gemm.ClusterOps(rd=qk.n_reads + vdo.n_reads,
                                 valu=tr.philox_a if dropout else 0,
                                 lead=tr.lead_a)

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, lds_bytes, 16]

    def scopes(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

    @flyc.kernel(known_block_size=[tr.block_size, 1, 1])
    # The kernel ABI is bwd_dq_kernel's (gfx1201), so aot/flyc_bwd_dq.py can serve both.
    def fmha_bwd_dq_gfx942_kernel(
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
        _ = tag                     # puts every traits field in the JIT cache key
        # ctx: the frozen traits plus the run-time state. `carried` names what
        # crosses the tile loop's back edge: dQ^T, K(j+1) and V(j+2) in VGPRs, bias(j+1).
        ctx = flex_gemm.Ctx(tr, carried=('dq', 'k_regs', 'v_regs', 'b_raws'))
        ctx.qk = qk

        # ---- prelude: addressing, resources, per-lane plans (the order the ISA was tuned in)
        # Lane geometry: a lane holds row lm of its wave's WAVE_M and one of the
        # g_lanes lane groups (lh). Causal blocks pair q tiles g and T-1-g.
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lm = lane % wm
        lh = lane // wm
        if const_expr(tile_fastest):
            g_tile = fx.Index(gpu.block_idx.x)
            head = fx.Index(gpu.block_idx.y)
        else:
            head = fx.Index(gpu.block_idx.x)
            g_tile = fx.Index(gpu.block_idx.y)
        if const_expr(tr.pair_tiles):
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
        # Q rows, for softmax.lse_rows.
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

        # LDS: row-major K and V, then K^T; one buffer each.
        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        ctx.lds = Lds(lds_p, elem, _LDS_DOMAIN, LDS_SCOPES)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)
        q_rsrc = rsrc_for(Q, q_batch, head, q_row_off, stride_q_batch, stride_q_head, stride_q_seq, sq, row_qk)
        do_rsrc = rsrc_for(DO, q_batch, head, q_row_off, stride_do_batch, stride_do_head, stride_do_seq, sq, row_vo)
        dq_rsrc = rsrc_for(DQ, q_batch, head, q_row_off, stride_dq_batch, stride_dq_head, stride_dq_seq, sq, row_qk)
        k_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk_i32, row_qk))
        v_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk_i32, row_vo))
        q_row_stride = fx.Index(fx.Int32(stride_q_seq))
        do_row_stride = fx.Index(fx.Int32(stride_do_seq))
        dq_row_stride = fx.Index(fx.Int32(stride_dq_seq))
        k_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)

        # LSE and Delta share the logsumexp row layout.
        lse_row0, lse_pitch = softmax.lse_rows(ctx)
        lse_rsrc = softmax.lse_rsrc_at(LSE, lse_row0, lse_pitch, sq, nullable=False)
        delta_rsrc = softmax.lse_rsrc_at(Delta, lse_row0, lse_pitch, sq, nullable=False)

        # Per-thread plans: K, V and K^T share one load unit (4 KV rows x 8 columns);
        # the row-major write is tile-relative, so K and V share it.
        k_plan, v_plan, kt_plan = gemm_load_plan(qk, tid, k_stride_b, also=((vdo, v_stride_b), (kt, None)))
        if const_expr(tr.padded):
            k_umask = flex_gemm.unit_masks(qk, k_plan, hdim_qk)
            v_umask = flex_gemm.unit_masks(vdo, v_plan, hdim_vo)
        rd_plan = gemm_read_plan(qk, lm, lh)       # tile-relative: serves K and V
        kt_read_plan = gemm_read_plan(kt, lm, lh)

        def k_clean(regs):
            return flex_gemm.unit_clean(regs, k_umask) if const_expr(tr.padded) else regs

        def v_clean(regs):
            return flex_gemm.unit_clean(regs, v_umask) if const_expr(tr.padded) else regs

        # Softmax constants: scores go to base 2 (exp2); lane group lh holds keys 4*lh + ....
        ctx.c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        ctx.c_neg_inf = fx.Float32(float('-inf'))
        kv_lane = fx.Int32(lh * 4)
        ctx.lane_key = lambda: kv_lane

        win_l, win_r = band_window(tr.causal, window_left, window_right, sq_i32, sk_i32, varlen_bits)

        # Dropout: the Triton kernels' stream, element for element (philox.py, grid_plane).
        if const_expr(dropout):
            ctx.rng, ctx.ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            ph_base, ph_stride = philox_plane(ctx.rng, ph_off, z_i32, num_head_q, head, max_seqlen_q, max_seqlen_k)
            ctx.idropout_p = idropout_p
            ctx.dropout_scale = dropout_scale

        # ---- bias and dB resources
        # dB stays in this body: its 64-lane ds_bpermute deal follows this body's S^T
        # layout; the narrow body's (ds_swizzle) differs, so sharing would not pay.
        if const_expr(bias):
            b_rsrc, b_mis = bias_slab(B,
                                      q_batch,
                                      head,
                                      q_row_off,
                                      k_row_off,
                                      stride_b_batch,
                                      stride_b_head,
                                      stride_b_seq_q,
                                      sq_i32,
                                      sk_i32)
            # dB: its own strides; an all-zero triple means the caller wants no dB, and
            # an empty range then drops every store.
            store_db = ((fx.Int64(stride_db_batch) != fx.Int64(0))
                        | (fx.Int64(stride_db_head) != fx.Int64(0))
                        | (fx.Int64(stride_db_seq_q) != fx.Int64(0)))
            db_elems = (fx.Int64(q_batch) * stride_db_batch + fx.Int64(head) * stride_db_head
                        + fx.Int64(q_row_off) * stride_db_seq_q + fx.Int64(k_row_off))
            db_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_db_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            db_addr = fx.Int64(fx.ptrtoint(DB)) + db_elems * fx.Int64(2)
            db_rsrc = buffer_ops.create_buffer_resource_from_addr(db_addr, num_records_bytes=db_span * fx.Int64(2))
            # Clean tiles store 8B groups when every group starts 8B-aligned.
            db_fast = (((db_addr & fx.Int64(7)) == fx.Int64(0))
                       & ((fx.Int64(stride_db_seq_q) & fx.Int64(3)) == fx.Int64(0)))

        # ---- one pass per q tile of the block
        for q_pass in range_constexpr(len(q_tiles)):
            q_tile = q_tiles[q_pass]
            q_start = q_tile * tr.block_m
            q_row = q_start + wave * wm + lm
            q_ok = q_row < sq

            # Q^T and dO^T stay in VGPRs for the pass; slack columns are zeroed.
            ctx.q_packs = flex_gemm.resident_packs(qk, q_rsrc, q_row, q_row_stride, lh, hdim_qk)
            ctx.do_packs = flex_gemm.resident_packs(vdo, do_rsrc, q_row, do_row_stride, lh, hdim_vo)
            # Natural-base LSE; +inf (dead rows, rows past sq) makes P exactly 0.
            lse = softmax.lse_load(lse_rsrc, lse_pitch, q_row)
            ctx.delta = softmax.lse_load(delta_rsrc, lse_pitch, q_row)
            ctx.lse2 = q_ok.select(lse * fx.Float32(LOG2E), fx.Float32(float('inf')))
            ctx.neg_lse2 = fx.Float32(0.0) - ctx.lse2

            # Tiles: blocks past a short sequence, or whose band misses every key, do none.
            q_first, q_last, kv_base, n_tiles = kv_tiles(q_start, tr.block_m, block_n, sq_i32, sk_i32, win_l, win_r)
            band = row_band(q_row, win_l, win_r, sk_i32)
            ctx.band = lambda: band
            lead, clean_end = clean_tiles(q_first, q_last, kv_base, block_n, sk_i32, win_l, win_r)
            r_lead = smin(lead, n_tiles)
            if const_expr(bias):
                # Only the last tile can hold the dword that straddles the bias slab's end.
                clean_end = smin(clean_end, smax(n_tiles, fx.Index(1)) - fx.Index(1))
            r_clean = smax(smin(clean_end, n_tiles), r_lead)

            # ---- bias and dB layouts of this pass
            if const_expr(bias and wm == 32):
                # Coalesced row-piece (RP) layout, as the narrow body: in load k, lane s
                # holds 8-byte piece (s % 8) ^ 2k of wave row 8k + s/8, 8 lanes to a 64-byte
                # row segment; ds_bpermute deals group q to the S^T lanes in round q, which
                # RP lane s holds in load q ^ a, a = (s % 8) / 2.
                q_wave = q_start + wave * wm
                rp_a = (lane % 8) // 2
                rp_row = [q_wave + fx.Index(8 * k) + lane // 8 for k in range(4)]
                rp_key = [((lane % 8) ^ fx.Index(2 * k)) * 4 for k in range(4)]
                b_rp = [b_mis + (fx.Int32(rp_row[k]) * fx.Int32(stride_b_seq_q) + fx.Int32(rp_key[k])) * fx.Int32(2)
                        for k in range(4)]
                b_rd = [x & fx.Int32(-4) for x in b_rp]
                # Row parity alone sets the shift, and 8k keeps it.
                b_shift = (b_rp[0] & fx.Int32(3)) * fx.Int32(8)
                rp_pull = fx.Int32((lm % 8) * 32 + lh * 4 + (lm // 8) * 8)
                # RP stores: lane s, store k, writes the piece of S^T lane (lh = s % 2, lm = 8k + s/8).
                db_rp = [((rp_row[k] < sq) & store_db).select((fx.Int32(rp_row[k]) * fx.Int32(stride_db_seq_q)
                                                               + fx.Int32(rp_key[k])) * fx.Int32(2),
                                                              fx.Int32(0x7FFFFFF0)) for k in range(4)]
                db_raw = rocdl.get_buffer_rsrc(db_rsrc)
                db_aligned = (((db_addr & fx.Int64(3)) == fx.Int64(0))
                              & ((fx.Int64(stride_db_seq_q) & fx.Int64(1)) == fx.Int64(0)))
                rp_push = fx.Int32((lane % 2) * 128 + rp_a * 32 + (lane // 8) * 4)
                rp_a0 = (rp_a % 2) == fx.Index(1)
                rp_a1 = rp_a >= fx.Index(2)
                b_rd_n = 4
            elif const_expr(bias):
                b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + kv_lane) * fx.Int32(2)
                b_rd = [(b_byte & fx.Int32(-4)) + fx.Int32(4 * g_lanes * g * 2) for g in range(ns // 4)]
                b_shift = (b_byte & fx.Int32(3)) * fx.Int32(8)
                db_lane = fx.Int32(q_row) * fx.Int32(stride_db_seq_q) + kv_lane
                db_ok = q_ok & store_db
                b_rd_n = ns // 4

            def rp_pick(vals, c):
                """vals[c ^ a]: three v_cndmask on two lane masks."""
                lo = rp_a0.select(vals[1 ^ (c & 1)], vals[c & 1])
                hi = rp_a0.select(vals[3 ^ (c & 1)], vals[2 | (c & 1)])
                return rp_a1.select(hi, lo) if c < 2 else rp_a1.select(lo, hi)

            def bias_issue(kv_start):
                """Raw dwords of this lane's bias for one KV tile, 3 aligned dwords a group
                (odd-pitch rows funnel-shift). Issued a tile ahead of its use."""
                if const_expr(not bias):
                    return []
                soff = fx.Int32(kv_start) * fx.Int32(2)
                return [rocdl.raw_ptr_buffer_load(Vec.make_type(3, fx.Int32), b_rsrc, b_rd[i], soff, 0)
                        for i in range(b_rd_n)]

            @flyc.jit
            def bias_tail(raws, kv_start):
                """bias_issue's dwords, reloaded 2 bytes at a time on the tile holding key
                sk - 1: a dword straddling the slab's end reads 0, even its in-range half."""
                out = list(raws)
                if fx.Int32(kv_start) + fx.Int32(block_n) >= sk_i32:
                    soff = fx.Int32(kv_start) * fx.Int32(2)
                    out = []
                    for i in range_constexpr(b_rd_n):
                        hs = [fx.Int32(fx.Int16(rocdl.raw_ptr_buffer_load(T.i16,
                                                                          b_rsrc,
                                                                          b_rd[i] + fx.Int32(2 * e),
                                                                          soff,
                                                                          0))) & fx.Int32(0xFFFF)
                              for e in range(6)]
                        out.append(Vec.from_elements([hs[2 * w] | (hs[2 * w + 1] << fx.Int32(16)) for w in range(3)],
                                                     fx.Int32).ir_value())
                return out

            def bias_finish(raws):
                """bias_issue's dwords shifted (and, at WAVE_M 32, dealt) to S^T order."""
                words = []
                for i in range_constexpr(b_rd_n):
                    words.append(funnel_words(Vec(raws[i]), b_shift))
                if const_expr(wm == 32):
                    words = [[fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type,
                                                         rp_pull ^ fx.Int32(8 * q),
                                                         rp_pick([words[k][e] for k in range(4)], q)))
                              for e in range(2)] for q in range(4)]
                return [Vec.from_elements(w, fx.Int32).bitcast(elem).ir_value() for w in words]

            def db_store_rp(packs, kv_start, mode):
                """dB = dS dealt to the RP layout: 8-byte stores (`mode` 'dword'), else
                2-byte ('short', or 'masked' at keys past sk)."""
                recv = []
                for q in range_constexpr(4):
                    own = Vec(packs[q]).bitcast(fx.Int32)
                    recv.append([fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, rp_push ^ fx.Int32(32 * q), fx.Int32(own[e])))
                                 for e in range(2)])
                dealt = [Vec.from_elements([rp_pick([recv[q][e] for q in range(4)], k) for e in range(2)], fx.Int32)
                         for k in range(4)]
                soff = fx.Int32(kv_start) * fx.Int32(2)
                for k in range_constexpr(4):
                    if const_expr(mode == 'dword'):
                        rocdl.raw_ptr_buffer_store(raw(dealt[k]), db_raw, raw(db_rp[k]), raw(soff))
                        continue
                    hs = Vec(dealt[k]).bitcast(fx.Int16)
                    for e in range_constexpr(4):
                        o = db_rp[k] + fx.Int32(2 * e)
                        if const_expr(mode == 'masked'):
                            key = fx.Int32(kv_start) + fx.Int32(rp_key[k] + e)
                            o = (key < sk_i32).select(o, fx.Int32(0x7FFFFFF0))
                        rocdl.raw_ptr_buffer_store(raw(fx.Int16(hs[e])), db_raw, raw(o), raw(soff))

            if const_expr(dropout):
                # Tiles start on a multiple of 4 keys: this lane's part of the offset is loop-invariant.
                ctx.ph_row = ctx.rng.grid_offset(ph_base, ph_stride, fx.Int32(q_row), kv_lane)

            def db_store_scalar(packs, base, kv_start):
                """One element at a time: a dB row may start mid-dword and end mid-group."""
                for r in range_constexpr(ns):
                    rel = 4 * g_lanes * (r // 4) + r % 4
                    col = fx.Int32(kv_start) + kv_lane + fx.Int32(rel)
                    buffer_ops.buffer_store(Vec(packs[r // 4])[r % 4],
                                            db_rsrc,
                                            base + fx.Int32(rel),
                                            mask=db_ok & (col < sk_i32))

            def db_out(packs, kv_start, masked):
                """dB = dS as rounded for the MFMA: RP layout at WAVE_M 32, else by 4-key group."""
                if const_expr(bias and wm == 32):
                    if const_expr(masked):
                        db_store_rp(packs, kv_start, 'masked')
                    else:
                        if db_aligned:
                            db_store_rp(packs, kv_start, 'dword')
                        else:
                            db_store_rp(packs, kv_start, 'short')
                elif const_expr(bias):
                    base = db_lane + fx.Int32(kv_start)
                    if const_expr(masked):
                        db_store_scalar(packs, base, kv_start)
                    else:
                        if db_fast:
                            for i in range_constexpr(ns // 4):
                                buffer_ops.buffer_store(packs[i], db_rsrc, base + fx.Int32(4 * g_lanes * i), mask=db_ok)
                        else:
                            db_store_scalar(packs, base, kv_start)

            # ---- cluster ends: LDS reads land before the next cluster overwrites them
            # (s_barrier does not wait for them).
            def a_end():
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier('A')

            def b_end():
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier('B')

            # ---- the tile loop: S, dP, Philox in A(j); dS, dB, dQ in B(j)
            # One buffer each: K(j+1) and V(j+2) wait in VGPRs until their buffer is free.
            def iteration(j, b, masked):
                kv_cur = kv_base + j * block_n
                b_raws = ctx.b_raws
                if const_expr(bias and masked):
                    b_raws = bias_tail(b_raws, kv_cur)
                # A(j): K^T(j), then K(j+1) into the same registers; bias(j+1) first so
                # its wait does not drain them.
                gemm_stage(kt, kt_plan, ctx.lds, scopes, ctx.k_regs, 0)                          # K^T(j)
                b_next = bias_issue(kv_cur + block_n)
                rocdl.sched_barrier(0)
                ctx.k_regs = gemm_load(qk, k_plan, k_rsrc, kv_cur + block_n, k_stride_b)        # K(j+1)
                s = gemm_mma(qk, gemm_read(qk, rd_plan, ctx.lds, 0), ctx.q_packs)               # S(j)
                dp = gemm_mma(vdo, gemm_read(vdo, rd_plan, ctx.lds, 0), ctx.do_packs)           # dP(j)
                keep = softmax.keep_scale(ctx, kv_cur) if const_expr(dropout) else None         # under the MFMAs
                flex_gemm.spread(a_mfma, a_ops)
                a_end()
                # B(j): V(j+1) row-major, then V(j+2) into its registers; dS(j);
                # dQ^T += K^T(j) dS^T(j); K(j+1) row-major last.
                gemm_stage(vdo, v_plan, ctx.lds, scopes, v_clean(ctx.v_regs), 0)                 # V(j+1)
                rocdl.sched_barrier(0)
                ctx.v_regs = gemm_load(vdo, v_plan, v_rsrc, kv_cur + 2 * block_n, v_stride_b)   # V(j+2)
                kts = gemm_read(kt, kt_read_plan, ctx.lds, 0)
                rocdl.sched_barrier(0)              # the whole K^T tile lands under the dS VALU
                bias_vals = bias_finish(b_raws) if const_expr(bias) else None
                p = softmax.p_from_lse(ctx, s, kv_cur, masked, bias_vals, exponent=softmax.P_EXP_FMA)
                ds = softmax.ds_packs(ctx, p, dp, keep, exponent=softmax.P_EXP_FMA)            # dS(j)
                db_out(ds, kv_cur, masked)
                ctx.dq = gemm_mma(kt, kts, ds, ctx.dq, order='b_outer')                          # dQ += K^T dS
                rocdl.sched_barrier(0)
                ctx.k_regs = k_clean(ctx.k_regs)
                gemm_stage(qk, k_plan, ctx.lds, scopes, ctx.k_regs, 0)                           # K(j+1)
                b_end()
                ctx.b_raws = b_next

            # Prologue: K, V (tile 0) row-major; K^T(0) is written by A(0); V(1) in flight.
            ctx.k_regs = gemm_load(qk, k_plan, k_rsrc, kv_base, k_stride_b)
            ctx.v_regs = gemm_load(vdo, v_plan, v_rsrc, kv_base, v_stride_b)
            ctx.k_regs = k_clean(ctx.k_regs)
            gemm_stage(qk, k_plan, ctx.lds, scopes, ctx.k_regs, 0)
            gemm_stage(vdo, v_plan, ctx.lds, scopes, v_clean(ctx.v_regs), 0)
            rocdl.sched_barrier(0)
            ctx.v_regs = gemm_load(vdo, v_plan, v_rsrc, kv_base + block_n, v_stride_b)
            ctx.b_raws = bias_issue(kv_base)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            ctx.dq = [fx.full(ns, 0.0, fx.Float32) for _ in range(kt.out_tiles)]

            # Masked tiles first and last, the clean middle; one buffer, so no dispatch.
            ctx.tiles(iteration, r_lead, r_clean, n_tiles, buffers=1)

            # ---- epilogue: dQ = sm_scale * dQ^T, transposed back by the store
            tensor_io.store_acc(kt,
                                ctx.dq,
                                dq_rsrc,
                                q_row,
                                dq_row_stride,
                                lane=lh,
                                col_limit=row_qk,
                                row_ok=q_ok,
                                elem=elem,
                                pad_floor=tr.pad_floor,
                                scale=sm_scale,
                                offset_first=False)

    @flyc.jit
    def launch_fmha_bwd_dq_gfx942_wide(
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
        grid_y = (fx.Index(max_seqlen_q) + tr.grid_block_m - 1) // tr.grid_block_m
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_bwd_dq_gfx942_kernel(Q,
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
                                  value_attrs={'rocdl.waves_per_eu': waves_per_eu,
                                               'rocdl.flat_work_group_size': f'{tr.block_size},{tr.block_size}'}).launch(
            grid=(grid_y, fx.Index(num_head_q), grid_z) if tile_fastest else (fx.Index(num_head_q), grid_y, grid_z),
            block=(tr.block_size, 1, 1),
            stream=stream)

    _launch = host.make_dq_launch(launch_fmha_bwd_dq_gfx942_wide,
                                  tr,
                                  causal_type,
                                  host_causal_type,
                                  bias,
                                  dropout)
    _launch.block_m = tr.grid_block_m
    _launch.varlen_bits = fmha.varlen_bits
    _launch.varlen_compact = fmha.varlen_compact
    _launch.varlen_padded = fmha.varlen_padded
    return _launch
