# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention backward dQ/dB for gfx942, FlyDSL: the narrow body (head dims
<= 128) on `flex_gemm`, `softmax` and `tensor_io`.

    S^T  = K.Q^T                P = exp2(sm_scale*log2e*S + log2e*bias - lse2)
    dP^T = V.dO^T               dS = P * (dP - Delta)
    dQ^T += K^T.dS^T            dQ = sm_scale * dQ;  dB = dS

All three in the transposed form, as the forward: S and dP are its QK GEMM
(`trait_ab`), dQ derives K^T on read from the K tile (`trait_tb` with
`derived_from`). K/V rows use the 'runs' key order, so a lane's S^T values are
8-key runs of its row: bias loads and dB stores then move 16 bytes a lane.
The dQ entry module: this file holds the narrow body, and
`build_fmha_bwd_dq_gfx942_module_primary` picks any body from the knobs. Every
body's kernel is `fmha_bwd_dq_gfx942_kernel`, with this file's signature.
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
    gemm_dma,
    gemm_dma_plan,
    gemm_mma,
    gemm_read,
    gemm_read_plan,
    gemm_stream,
)
from .utils import (
    LOG2E,
    WAVE,
    Lds,
    alias_scopes,
    band_window,
    barrier_if,
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

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dq_gfx942.lds">'
NBUF = 3
LDS_SCOPES = tuple(f'{t}{b}' for t in 'kv' for b in range(NBUF))
WAVE_M = 32         # query rows per wave: the N of the 32x32x8 MFMA


@dataclass(frozen=True)
class DqTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    num_waves: int
    block_n: int
    waves_per_eu: int
    grid_axis_order: int
    lpt: bool
    bias: bool
    dropout: bool

    # make_dq_launch's names
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
    def nsub(self):
        """32x32 S^T sub-tiles per KV tile."""
        return self.block_n // 32

    @property
    def pad_floor(self):
        """Columns below this are real in every call this build serves."""
        return 0 if self.padded else self.head_dim

    def gemms(self):
        """(S, dP, dQ): S^T = K.Q^T and dP^T = V.dO^T on K and V tiles cycling
        through NBUF LDS buffers; dQ^T += K^T.dS^T, K^T derived from the K tile."""
        qk = flex_gemm.trait_ab(self.dtype,
                                self.block_n,
                                self.head_dim,
                                self.pad_floor,
                                self.block_size,
                                0,
                                'k',
                                buffers=NBUF,
                                key_order='runs')
        vdo = flex_gemm.trait_ab(self.dtype,
                                 self.block_n,
                                 self.head_dim,
                                 self.pad_floor,
                                 self.block_size,
                                 NBUF * qk.a.tile_bytes,
                                 'v',
                                 buffers=NBUF,
                                 key_order='runs')
        kds = flex_gemm.trait_tb(self.dtype,
                                 self.block_n,
                                 self.head_dim,
                                 self.block_size,
                                 0,
                                 'k',
                                 derived_from=qk.a)
        return qk, vdo, kds


def build_fmha_bwd_dq_gfx942(head_dim,
                                  causal=False,
                                  causal_type=None,
                                  dtype_str='bf16',
                                  bias=False,
                                  dropout=False,
                                  num_waves=8,
                                  block_n=None,
                                  waves_per_eu=2,
                                  grid_axis_order=None,
                                  padded_head=False,
                                  lpt=None):
    """A launcher with gfx1201/dq.py's call interface and kernel ABI.

    `causal_type` as gfx1201's: 1/2 are host sentinels, 3 is gSWA; all compile to 3.
    """
    assert head_dim in tuning.LADDER and head_dim <= 128, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if block_n is None:
        block_n = tuning.default_block_n(head_dim)
    assert block_n in (32, 64), block_n
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    if lpt is None:
        lpt = tuning.default_lpt(causal_type != 0)
    tr = DqTraits(head_dim=head_dim,
                  padded=padded_head,
                  causal=causal_type != 0,
                  dtype=dtype_str,
                  num_waves=num_waves,
                  block_n=block_n,
                  waves_per_eu=waves_per_eu,
                  grid_axis_order=grid_axis_order,
                  lpt=lpt,
                  bias=bias,
                  dropout=dropout)
    # The grid extent the tuning policy records.
    assert tr.block_m == tuning.block_m('narrow', num_waves, tr.causal), tr.block_m
    qk, vdo, kds = tr.gemms()
    lds_bytes = 2 * NBUF * qk.a.tile_bytes
    assert lds_bytes <= 65536, lds_bytes
    tag = 'dq_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    block_n = tr.block_n
    nsub = tr.nsub

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
        # crosses the tile loop's back edge: dQ^T, and bias(j+1) in flight.
        ctx = flex_gemm.Ctx(tr, carried=('dq', 'b_raws'))
        ctx.qk = qk

        # ---- prelude: addressing, resources, per-lane plans (the order the ISA was tuned in)
        # Lane geometry and the (q tile, head) this block serves; under causal, LPT
        # runs the longest q tiles first. GQA maps the query head to its kv head.
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
        if const_expr(tr.lpt):
            q_tile = (fx.Index(max_seqlen_q) + tr.block_m - 1) // tr.block_m - fx.Index(1) - q_tile
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

        # LDS: K and V tiles, each cycling through NBUF buffers.
        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_addr = fx.Index(fx.ptrtoint(lds_p))
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

        # Each wave owns 32 query rows; a lane holds one row (lm) and one half of
        # the 32x32x8 accumulator's columns (lh).
        q_start = q_tile * tr.block_m
        q_wave = q_start + wave * WAVE_M
        q_row = q_wave + lm
        q_ok = q_row < sq

        # Q^T and dO^T stay in VGPRs for the whole loop; slack columns are zeroed in
        # both operands (0 * NaN is NaN), in the resident packs and as K/V reads land.
        k_masks = flex_gemm.a_col_masks(qk, hdim_qk, lh)
        v_masks = flex_gemm.a_col_masks(vdo, hdim_vo, lh)
        ctx.q_packs = flex_gemm.resident_packs(qk, q_rsrc, q_row, q_row_stride, lh, hdim_qk, masks=k_masks)
        ctx.do_packs = flex_gemm.resident_packs(vdo, do_rsrc, q_row, do_row_stride, lh, hdim_vo, masks=v_masks)

        # LSE (natural base, +inf on rows that see no key) and Delta share one row
        # layout. A row past seqlen_q gets lse2 = +inf, so its P, dS and dB are 0.
        lse_row0, lse_pitch = softmax.lse_rows(ctx)

        def row_scalar(ptr):
            rsrc = softmax.lse_rsrc_at(ptr, lse_row0, lse_pitch, sq, nullable=False)
            return softmax.lse_load(rsrc, lse_pitch, q_row)

        ctx.lse2 = q_ok.select(row_scalar(LSE) * fx.Float32(LOG2E), fx.Float32(float('inf')))
        ctx.delta = q_ok.select(row_scalar(Delta), fx.Float32(0.0))

        # Per-lane plans: K and V DMAs share one set of M0 values (the tiles share a
        # layout); reads of S, dP and the derived K^T reads of dQ.
        k_dma_plan, v_dma_plan = gemm_dma_plan(qk, wave, lane, k_stride_b, lds_addr, also=((vdo, v_stride_b),))
        rd_plan = gemm_read_plan(qk, lm, lh)       # tile-relative: serves K and V
        # S^T value r of this lane holds tile key qk.coords[r] + kv_lane.
        kv_lane = lh * 8
        kt_read_plan = gemm_read_plan(kds, lm, lh)

        # Softmax constants: scores go to base 2 (exp2).
        ctx.c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        ctx.c_neg_inf = fx.Float32(float('-inf'))

        # Tiles: masked ones bracket the clean ones; j_clean is a multiple of NBUF so
        # the unrolled loop knows its buffers.
        win_l, win_r = band_window(tr.causal, window_left, window_right, sq_i32, sk_i32, varlen_bits)
        q_first, q_last, kv_base, n_tiles = kv_tiles(q_start, tr.block_m, block_n, sq_i32, sk_i32, win_l, win_r)
        band = row_band(q_row, win_l, win_r, sk_i32)
        ctx.band = lambda: band
        ctx.lane_key = lambda: fx.Int32(kv_lane)
        lead, clean_end = clean_tiles(q_first, q_last, kv_base, block_n, sk_i32, win_l, win_r)
        clean_end = smin(clean_end, n_tiles)
        j_clean = smin(lead + (fx.Index(NBUF) - lead % NBUF) % NBUF, n_tiles)
        j_clean_end = smax(j_clean + (smax(clean_end, j_clean) - j_clean) // NBUF * NBUF, j_clean)

        # ---- bias and dB: 16 bytes a lane, by whole 64 B segments of 16 rows
        # dB stays in this body: its lane exchange (ds_swizzle lane ^ 16 / ^ 8) follows
        # this body's S^T layout; the wide body's differs, so sharing would not pay.
        if const_expr(bias):
            # (B, H, Sq, Sk), Sk contiguous; rows offset by Q's varlen offset, columns by K's.
            b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head) * stride_b_head
                       + fx.Int64(q_row_off) * stride_b_seq_q + fx.Int64(k_row_off))
            b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2),
                num_records_bytes=b_span * fx.Int64(2)))
            # dB = dS. A caller that wants none passes all-zero dB strides (and maybe a
            # null DB): an empty range then drops every store.
            store_db = ((fx.Int64(stride_db_batch) != fx.Int64(0))
                        | (fx.Int64(stride_db_head) != fx.Int64(0))
                        | (fx.Int64(stride_db_seq_q) != fx.Int64(0)))
            db_elems = (fx.Int64(q_batch) * stride_db_batch + fx.Int64(head) * stride_db_head
                        + fx.Int64(q_row_off) * stride_db_seq_q + fx.Int64(k_row_off))
            db_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_db_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            db_addr = fx.Int64(fx.ptrtoint(DB)) + db_elems * fx.Int64(2)
            db_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                db_addr,
                num_records_bytes=store_db.select(db_span * fx.Int64(2), fx.Int64(0))))
            # Rows start on a dword: clean tiles store whole dwords.
            db_aligned = (((db_addr & fx.Int64(3)) == fx.Int64(0))
                          & ((fx.Int64(stride_db_seq_q) & fx.Int64(1)) == fx.Int64(0)))
            # Bias and dB of a 32-key sub-tile move in two accesses, A for wave rows 0-15
            # and B for 16-31, each a whole 64-byte segment of 16 rows: 16 bytes a lane,
            # rows pair_row[x], keys pair_key[x] + [0, 8). Lane lm owns keys [0, 16) of
            # them and trades keys [16, 32) with lane lm ^ 16 (ds_swizzle).
            pair_lo = lm < fx.Index(16)
            pair_row = [q_wave + lm % 16 + 16 * x for x in range(2)]
            pair_key = [fx.Int32(kv_lane) + pair_lo.select(fx.Int32(16 * x), fx.Int32(16 - 16 * x)) for x in range(2)]
            # Odd-pitch rows start mid-dword: the loads are unaligned.
            b_off = [(fx.Int32(pair_row[x]) * fx.Int32(stride_b_seq_q) + pair_key[x]) * fx.Int32(2) for x in range(2)]
            db_off = [(pair_row[x] < sq).select((fx.Int32(pair_row[x]) * fx.Int32(stride_db_seq_q) + pair_key[x])
                                                * fx.Int32(2),
                                                fx.Int32(0x7FFFFFF0)) for x in range(2)]
            # bias_issue's offsets, by access.
            b_offs = [b_off[x] + fx.Int32(64 * n) for n in range(nsub) for x in range(2)]
            if const_expr(nsub == 2):
                # A 64-key tile trades once more, sub-tile 1's access with lane lm ^ 8:
                # access (x, h) then moves whole 128-byte segments of rows
                # 16x + 8h + lm % 8, lane lm taking sub-tile n_of[h]. Bias loads and
                # clean dB stores use it.
                quad_lo = (lm % 16) < fx.Index(8)
                db_off8 = []
                b_offs = []
                for x in range_constexpr(2):
                    for h in range_constexpr(2):
                        row = q_wave + lm % 8 + 16 * x + 8 * h
                        n_of = quad_lo.select(fx.Int32(64 * h), fx.Int32(64 - 64 * h))
                        b_offs.append((fx.Int32(row) * fx.Int32(stride_b_seq_q) + pair_key[x]) * fx.Int32(2) + n_of)
                        db_off8.append((row < sq).select((fx.Int32(row) * fx.Int32(stride_db_seq_q) + pair_key[x])
                                                         * fx.Int32(2) + n_of,
                                                         fx.Int32(0x7FFFFFF0)))

        def bias_issue(kv_start):
            """Bias of one KV tile by access (b_offs). Issued a tile ahead of its use."""
            if const_expr(not bias):
                return []
            soff = fx.Int32(kv_start) * fx.Int32(2)
            return [rocdl.raw_ptr_buffer_load(Vec.make_type(4, fx.Int32), b_rsrc, o, soff, 0) for o in b_offs]

        @flyc.jit
        def bias_tail(raws, kv_start):
            """bias_issue's pieces, reloaded by element on a tile past sk: an unaligned
            dword straddling the slab's end reads 0, even its in-range half."""
            out = list(raws)
            if fx.Int32(kv_start) + fx.Int32(block_n) > sk_i32:
                soff = fx.Int32(kv_start) * fx.Int32(2)
                out = []
                for p in range_constexpr(2 * nsub):
                    hs = [fx.Int32(fx.Int16(rocdl.raw_ptr_buffer_load(T.i16,
                                                                      b_rsrc,
                                                                      b_offs[p] + fx.Int32(2 * e),
                                                                      soff,
                                                                      0))) & fx.Int32(0xFFFF)
                          for e in range(8)]
                    out.append(Vec.from_elements([hs[2 * w] | (hs[2 * w + 1] << fx.Int32(16)) for w in range(4)],
                                                 fx.Int32).ir_value())
            return out

        def swap16(x):
            """x of lane ^ 16."""
            return fx.Int32(rocdl.ds_swizzle(T.i32, raw(x), raw(fx.Int32(0x401F))))

        def swap8(x):
            """x of lane ^ 8."""
            return fx.Int32(rocdl.ds_swizzle(T.i32, raw(x), raw(fx.Int32(0x201F))))

        def untrade8(words):
            """Bias accesses (x, h) back to (n, x) of a 32-key sub-tile: undo the lane ^ 8 trade."""
            if const_expr(nsub == 1):
                return words
            mine = [[quad_lo.select(words[2 * x][w], words[2 * x + 1][w]) for w in range(4)] for x in range(2)]
            far = [[swap8(quad_lo.select(words[2 * x + 1][w], words[2 * x][w])) for w in range(4)] for x in range(2)]
            return mine + far

        def bias_finish(raws):
            """bias_issue's accesses as S^T groups of 4 keys: group 4n + i holds keys
            32n + 16(i/2) + 4(i%2) + kv_lane + [0, 4)."""
            if const_expr(not bias):
                return None
            words = untrade8([[fx.Int32(Vec(r)[w]) for w in range(4)] for r in raws])
            out = []
            for n in range_constexpr(nsub):
                acc = [words[2 * n + x] for x in range(2)]
                own = [pair_lo.select(acc[0][w], acc[1][w]) for w in range(4)]
                far = [swap16(pair_lo.select(acc[1][w], acc[0][w])) for w in range(4)]
                for run in (own, far):
                    for h in range_constexpr(2):
                        out.append(Vec.from_elements(run[2 * h:2 * h + 2], fx.Int32).bitcast(elem).ir_value())
            return out

        def db_store(ds, kv_start, mode):
            """dB = dS by accesses A and B: 16 bytes a lane (`mode` 'dword', rows
            dword-aligned), else 2 ('short', or 'masked' at keys past sk)."""
            soff = fx.Int32(kv_start) * fx.Int32(2)
            acc = []
            for n in range_constexpr(nsub):
                own, far = [[fx.Int32(Vec(ds[4 * n + 2 * p + x]).bitcast(fx.Int32)[e])
                             for x in range(2) for e in range(2)] for p in range(2)]
                far = [swap16(w) for w in far]
                acc.append([[pair_lo.select(own[w], far[w]) if x == 0 else pair_lo.select(far[w], own[w])
                             for w in range(4)] for x in range(2)])
            if const_expr(mode == 'dword' and nsub == 2):
                for x in range_constexpr(2):
                    far = [swap8(w) for w in acc[1][x]]
                    for h in range_constexpr(2):
                        v4 = Vec.from_elements([quad_lo.select(acc[0][x][w], far[w]) if h == 0
                                                else quad_lo.select(far[w], acc[0][x][w]) for w in range(4)], fx.Int32)
                        rocdl.raw_ptr_buffer_store(raw(v4), db_rsrc, raw(db_off8[2 * x + h]), raw(soff))
                return
            for n in range_constexpr(nsub):
                for x in range_constexpr(2):
                    v4 = Vec.from_elements(acc[n][x], fx.Int32)
                    off = db_off[x] + fx.Int32(64 * n)
                    if const_expr(mode == 'dword'):
                        rocdl.raw_ptr_buffer_store(raw(v4), db_rsrc, raw(off), raw(soff))
                        continue
                    hs = v4.bitcast(fx.Int16)
                    for e in range_constexpr(8):
                        o = off + fx.Int32(2 * e)
                        if const_expr(mode == 'masked'):
                            key = fx.Int32(kv_start) + pair_key[x] + fx.Int32(32 * n + e)
                            o = (key < sk_i32).select(o, fx.Int32(0x7FFFFFF0))
                        rocdl.raw_ptr_buffer_store(raw(fx.Int16(hs[e])), db_rsrc, raw(o), raw(soff))

        @flyc.jit
        def db_store_clean(ds, kv_start):
            if db_aligned:
                db_store(ds, kv_start, 'dword')
            else:
                db_store(ds, kv_start, 'short')

        def db_out(ds, kv_start, masked):
            """dB of tile j: masked tiles store by element, clean ones by dword when aligned."""
            if const_expr(bias and masked):
                db_store(ds, kv_start, 'masked')
            if const_expr(bias and not masked):
                db_store_clean(ds, kv_start)

        # Dropout: the Triton kernels' stream, element for element (philox.py,
        # grid_plane). Tiles start on a multiple of 4 keys, so this lane's part of
        # each Philox offset is loop-invariant.
        if const_expr(dropout):
            ctx.rng, ctx.ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            ph_base, ph_stride = philox_plane(ctx.rng, ph_off, z_i32, num_head_q, head, max_seqlen_q, max_seqlen_k)
            ctx.ph_row = ctx.rng.grid_offset(ph_base, ph_stride, fx.Int32(q_row), fx.Int32(kv_lane))
            ctx.idropout_p = idropout_p
            ctx.dropout_scale = dropout_scale

        # Waves 4..7 trail by one barrier, so an LDS buffer is safe two clusters
        # after its last reader / its producer's wait:
        #   A(j): S and dP read k/v[j%3]; P; waits DMA(j+1)
        #   B(j): DMA(j+2) -> k/v[(j+2)%3]; dS; dQ reads k[j%3]
        group_b = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                               fx.Int32((wave >= fx.Index(tr.num_waves // 2)).select(1, 0))))
        n_bias = 2 * nsub if bias else 0      # bias loads a tile

        def kv(j):
            return kv_base + j * block_n

        def kv_dma(kv_start, buf):
            """K and V of one tile -> LDS buffer `buf`."""
            gemm_dma(qk, k_dma_plan, k_rsrc, scopes, kv_start, k_stride_b, buf)
            gemm_dma(vdo, v_dma_plan, v_rsrc, scopes, kv_start, v_stride_b, buf)

        def a_end():
            """End of A(j): DMA(j+1) has landed (newer: dB(j-1)'s stores or bias(0)'s
            loads, then bias(j+1)'s); dB need not land yet."""
            if const_expr(bias):
                rocdl.sched_barrier(0)      # the vmcnt counts bias(j+1)'s loads: keep them above it
            rocdl.s_waitcnt(vmcnt=2 * n_bias)
            cluster_barrier()

        # Iteration -1: tiles 0 and 1 in flight, tile 0 landed; bias(0) issued.
        def warmup():
            rocdl.sched_barrier(0)
            kv_dma(kv_base, 0)
            rocdl.sched_barrier(0)          # the vmcnt below waits for tile 0 only
            kv_dma(kv_base + block_n, 1)
            rocdl.sched_barrier(0)
            ctx.b_raws = bias_issue(kv_base)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=2 * qk.a.n_dma + n_bias)
            cluster_barrier()
            barrier_if(group_b)             # open the stagger
            ctx.dq = [fx.full(16, 0.0, fx.Float32) for _ in range(kds.out_tiles)]

        # ---- the tile loop: S, P, dP in A(j); dS, dB, dQ in B(j)
        # `b` = j % 3 is a Python int (alias scopes need constants); `masked` says
        # whether tile j needs the causal/window/padding mask.
        def iteration(j, b, masked):
            kv_cur = kv(j)
            # A(j): Philox under S's MFMAs, P under dP's; bias(j+1) stays in flight.
            b_raws = bias_tail(ctx.b_raws, kv_cur) if const_expr(bias and masked) else ctx.b_raws
            bias_vals = bias_finish(b_raws)
            keep = softmax.keep_scale(ctx, kv_cur) if const_expr(dropout) else None
            s = gemm_mma(qk, gemm_read(qk, rd_plan, ctx.lds, b, masks=k_masks), ctx.q_packs)     # S(j)
            p = softmax.p_from_lse(ctx, s, kv_cur, masked, bias_vals)                                # P(j)
            ctx.b_raws = bias_issue(kv_cur + block_n)
            dp = gemm_mma(vdo, gemm_read(vdo, rd_plan, ctx.lds, b, masks=v_masks), ctx.do_packs)  # dP(j)
            a_end()
            # B(j): DMA(j+2) first, then dS, dB and dQ += K^T dS.
            kv_dma(kv_cur + 2 * block_n, (b + 2) % NBUF)
            rocdl.sched_barrier(0)
            ds = softmax.ds_packs(ctx, p, dp, keep)                                                  # dS(j)
            db_out(ds, kv_cur, masked)
            ctx.dq = gemm_stream(kds, kt_read_plan, ctx.lds, b, ds, ctx.dq)                          # dQ += K^T dS
            cluster_barrier()

        # Masked tiles run first and last, the clean middle unrolled x3 (flex_gemm.run_tiles).
        warmup()
        ctx.tiles(iteration, j_clean, j_clean_end, n_tiles, buffers=NBUF, tail_start=j_clean_end)
        barrier_if(fx.Int32(1) - group_b)         # close the stagger
        rocdl.s_waitcnt(0)

        # ---- epilogue: dQ = sm_scale * dQ^T, transposed back by the store
        tensor_io.store_acc(kds,
                            ctx.dq,
                            dq_rsrc,
                            q_row,
                            dq_row_stride,
                            lane=lh,
                            col_limit=row_qk,
                            row_ok=q_ok,
                            elem=elem,
                            pad_floor=tr.pad_floor,
                            scale=sm_scale)

    @flyc.jit
    def launch_fmha_bwd_dq_gfx942(
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
        grid_y = (fx.Index(max_seqlen_q) + tr.block_m - 1) // tr.block_m
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

    _launch = host.make_dq_launch(launch_fmha_bwd_dq_gfx942,
                                  tr,
                                  causal_type,
                                  host_causal_type,
                                  bias,
                                  dropout)
    _launch.block_m = tr.block_m
    return _launch


_BODIES = {
    'narrow': (__name__, 'build_fmha_bwd_dq_gfx942', ('num_waves', 'block_n', 'waves_per_eu', 'lpt')),
    'wide': (f'{__package__}.dq_wide',
             'build_fmha_bwd_dq_gfx942_wide',
             ('num_waves', 'block_n', 'waves_per_eu', 'wave_m')),
}


def build_fmha_bwd_dq_gfx942_module_primary(meta, knobs):
    """The body `knobs.body` names, built for a resolved (meta, knobs) pair:
    `fmha_meta.FmhaInputMetadata` and `tuning_dq.entry_resolve(meta)`."""
    return build_body_primary(_BODIES, meta, knobs)


def build_fmha_bwd_dq_gfx942_by_head_dim(head_dim, body=None, **kwargs):
    """A JIT build of the compiled `head_dim` by `body`'s builder (default: the
    body the policy routes `head_dim` to), with that builder's keywords and defaults."""
    return body_builder(_BODIES, body or tuning.body(head_dim))(head_dim, **kwargs)
