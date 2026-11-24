# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dK/dV backward for gfx942, FlyDSL: the narrow body (head dims <= 128) on
`flex_gemm`, `softmax` and `tensor_io`.

    S  = Q.K^T                  P = exp2(sm_scale*log2e*S + log2e*bias - lse2)
    dP = dO.V^T                 dS = P * (dP - Delta)
    dV^T += dO^T.P              dK^T += Q^T.dS;  dK = sm_scale * dK

The forward loop transposed: each
wave keeps 16*kv_blocks rows of K and V in registers, while 16-row Q/dO tiles
stream through LDS. S and dP are `trait_ab` GEMMs on 'swz' row-major tiles with
K/V as the resident B; on 16x16x16 they come out as one kv column and four q
rows a lane, the B operand of dV and dK, so P and dS need no repacking. dV and
dK read Q^T/dO^T built in LDS from the row tiles (`trait_tb` with
`transposed_from`): gfx942 has no transpose read.

The dK/dV entry module: this file holds the narrow body, and
`build_fmha_bwd_dkdv_gfx942_module_primary` picks any body from the knobs. Every
body's kernel is `fmha_bwd_dkdv_gfx942_kernel`, with this file's signature.
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
from . import tuning_dkdv as tuning
from .flex_gemm import (
    gemm_dma_m0,
    gemm_dma_plan,
    gemm_dma_swz,
    gemm_mma,
    gemm_read,
    gemm_read_plan,
    gemm_transpose,
    gemm_transpose_plan,
)
from .utils import (
    LOG2E,
    WAVE,
    alias_scopes,
    apply_mask,
    body_builder,
    build_body_primary,
    cluster_barrier,
    col_mask,
    philox_stream,
    rsrc_for,
    smax,
    smin,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942.lds">'
LDS_SCOPES = ('r0', 'r1', 't0', 't1')
BM = tuning.BLOCK_ROWS      # q rows per tile, and kv rows per MFMA block


@dataclass(frozen=True)
class DkdvTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    num_waves: int
    kv_blocks: int              # 16-row kv blocks per wave
    waves_per_eu: int
    interleave_valu: int
    grid_axis_order: int
    bias: bool
    dropout: bool
    dropout_scale_at: str       # softmax.P_DROP_AT_STORE or P_DROP_PER_ELEM
    mask_at: str                # softmax.MASK_SCORE or MASK_PROB

    # make_dkdv_launch's names
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
    def rows_per_wave(self):
        return BM * self.kv_blocks

    @property
    def block_n(self):
        return self.num_waves * self.rows_per_wave

    @property
    def pad_floor(self):
        """Columns below this are real in every call this build serves."""
        return 0 if self.padded else self.head_dim

    @property
    def tile_bytes(self):
        """One 16-row tile of one tensor, either orientation."""
        return BM * 2 * self.head_dim

    @property
    def slot_bytes(self):
        """Two rings of two slots, each holding Q's and dO's tile side by side:
        rows (row-major, DMA'd) then transposed (built in LDS)."""
        return 2 * self.tile_bytes

    @property
    def early(self):
        """Two waves per SIMD: the next tile's Philox runs under this tile's dV/dK
        (B2 H32 S4k d96: 2.30 vs 4.83 ms; at one wave it does not pay)."""
        return self.dropout and self.waves_per_eu >= 2

    @property
    def sched_valu(self):
        """One wave with dropout from d64: VALU-bound, the order is left to the
        scheduler (B2 H32 S8k d128: 12.88 vs 13.34 ms; d32 causal loses 7%)."""
        return 0 if self.dropout and not self.early and self.head_dim >= 64 else self.interleave_valu

    def gemms(self):
        """(S, dP, dV, dK): S = Q.K^T and dP = dO.V^T on the rows ring, K/V resident
        in kv_blocks blocks; dV^T += dO^T.P and dK^T += Q^T.dS on the transposed ring."""
        def rows(lds_base):
            return flex_gemm.trait_ab(self.dtype,
                                      BM,
                                      self.head_dim,
                                      self.pad_floor,
                                      self.block_size,
                                      lds_base,
                                      'r',
                                      dma='swz',
                                      mfma='16x16x16',
                                      b_tiles=self.kv_blocks,
                                      buf_stride=self.slot_bytes)

        def trans(lds_base, src):
            return flex_gemm.trait_tb(self.dtype,
                                      BM,
                                      self.head_dim,
                                      self.block_size,
                                      2 * self.slot_bytes + lds_base,
                                      't',
                                      mfma='16x16x16',
                                      transposed_from=src.a,
                                      b_tiles=self.kv_blocks,
                                      buf_stride=self.slot_bytes)

        qs = rows(0)
        dops = rows(self.tile_bytes)
        return qs, dops, trans(self.tile_bytes, dops), trans(0, qs)


def build_fmha_bwd_dkdv_gfx942(head_dim,
                                    causal=False,
                                    causal_type=None,
                                    dtype_str='bf16',
                                    bias=False,
                                    dropout=False,
                                    num_waves=None,
                                    kv_blocks=None,
                                    waves_per_eu=None,
                                    grid_axis_order=None,
                                    interleave_valu=None,
                                    padded_head=False,
                                    dropout_scale_at=None,
                                    mask_at=None):
    """A launcher with gfx1201/dkdv.py's `_launch` interface and kernel ABI.

    `causal_type` as gfx1201's: 1/2 are host sentinels, 3 is gSWA; all compile to 3.
    `dropout_scale_at`: where dropout's 1/(1-p) meets P for dV (softmax.P_DROP_*);
    `mask_at`: where the band mask lands (softmax.MASK_*). None: the policy's (tuning_dkdv).
    """
    assert head_dim in tuning.LADDER and head_dim <= 128, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if num_waves is None:
        num_waves = tuning.default_num_waves(head_dim)
    if kv_blocks is None:
        kv_blocks = tuning.default_kv_blocks(head_dim)
    if waves_per_eu is None:
        waves_per_eu = tuning.default_waves_per_eu(head_dim, bias)
    if interleave_valu is None:
        interleave_valu = tuning.default_interleave_valu(head_dim)
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    tr = DkdvTraits(head_dim=head_dim,
                    padded=padded_head,
                    causal=causal_type != 0,
                    dtype=dtype_str,
                    num_waves=num_waves,
                    kv_blocks=kv_blocks,
                    waves_per_eu=waves_per_eu,
                    interleave_valu=interleave_valu,
                    grid_axis_order=grid_axis_order,
                    bias=bias,
                    dropout=dropout,
                    dropout_scale_at=dropout_scale_at or tuning.default_dropout_scale_at('narrow'),
                    mask_at=mask_at or tuning.default_mask_at('narrow'))
    qs, dops, dvt, dkt = tr.gemms()
    lds_bytes = 4 * tr.slot_bytes
    assert lds_bytes <= 65536, lds_bytes
    tag = 'dkdv_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)
    KVB = tr.kv_blocks
    DB = head_dim // 16
    sched_valu = tr.sched_valu
    early = tr.early
    # MFMAs a phase spreads its VALU over: A runs S and dP, B runs dV and dK.
    n_mfma_a = qs.n_mma + dops.n_mma
    n_mfma_b = dvt.n_mma + dkt.n_mma

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
        lc = lane % 16              # MFMA row/column of this lane
        lg = lane // 16             # MFMA k group: k = 4 * lg + [0, 4)
        if const_expr(tile_fastest):
            kv_tile = fx.Index(gpu.block_idx.x)
            head_k = fx.Index(gpu.block_idx.y)
        else:
            head_k = fx.Index(gpu.block_idx.x)
            kv_tile = fx.Index(gpu.block_idx.y)
        z_i32 = fx.Int32(gpu.block_idx.z)
        group = fx.Index(num_head_q) // fx.Index(num_head_k)
        sq_i32, q_row_off, q_batch = fmha.decode_addressing(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32)
        sk_i32, k_row_off, k_batch = fmha.decode_addressing(varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32)
        sq = fx.Index(sq_i32)
        sk = fx.Index(sk_i32)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_base = fx.Index(fx.ptrtoint(lds.buf.ptr))

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)

        kv0 = kv_tile * tr.block_n
        kv0_i32 = fx.Int32(kv0)
        kv_wave = kv0 + wave * tr.rows_per_wave             # this wave's first kv row

        # ---- resident K and V: block b, read p: row kv_wave + 16b + lc,
        # d = 32p + 8lg + [0, 8); S/dP's k-steps take the low and high halves.
        k_rsrc = rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk, row_qk)
        v_rsrc = rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk, row_vo)
        k_row_stride = fx.Index(fx.Int32(stride_k_seq))
        v_row_stride = fx.Index(fx.Int32(stride_v_seq))
        qk_masks = {}
        vo_masks = {}
        for p in range_constexpr(qs.reads):
            if const_expr(32 * p + 32 > tr.pad_floor):
                c0 = fx.Int32(32 * p) + fx.Int32(lg * 8)
                qk_masks[p] = col_mask(c0, hdim_qk)
                vo_masks[p] = col_mask(c0, hdim_vo)
        k_packs = []
        v_packs = []
        for b in range_constexpr(KVB):
            row = kv_wave + fx.Index(16 * b) + lc
            kb = []
            vb = []
            for p in range_constexpr(qs.reads):
                col = fx.Index(32 * p) + lg * 8
                k8 = buffer_ops.buffer_load(k_rsrc, row * k_row_stride + col, vec_width=8, dtype=elem)
                v8 = buffer_ops.buffer_load(v_rsrc, row * v_row_stride + col, vec_width=8, dtype=elem)
                if const_expr(p in qk_masks):
                    # Slack columns may hold NaN, and 0 * NaN is NaN: zero both sides of S and dP.
                    k8 = apply_mask(k8, qk_masks[p], elem)
                    v8 = apply_mask(v8, vo_masks[p], elem)
                kb.append(k8)
                vb.append(v8)
            k_packs.append(kb)
            v_packs.append(vb)

        # ---- LDS plans: the rows ring's DMA (one M0 set for Q and dO), the
        # transposes that build Q^T/dO^T, and the read bases of both rings.
        lds_base_i32 = fx.Int32(lds_base)
        wave_u, dma_m0 = gemm_dma_m0(qs, wave, lds_base_i32)
        q_row_stride_b = fx.Int32(stride_q_seq) * fx.Int32(2)
        do_row_stride_b = fx.Int32(stride_do_seq) * fx.Int32(2)
        q_dma = gemm_dma_plan(qs, wave, lane, q_row_stride_b, wave_u)
        do_dma = gemm_dma_plan(dops, wave, lane, do_row_stride_b, wave_u)
        t_plan = gemm_transpose_plan(dkt, tid, lds_base_i32)
        rm_plan = gemm_read_plan(qs, lc, lg, lds_addr=lds_base_i32)
        t_rd = gemm_read_plan(dkt, lc, lg, lds_addr=lds_base_i32)

        # ---- per q head: Q, dO, LSE, Delta, bias and the dropout plane
        def head_state(head_q):
            q_rsrc = rocdl.get_buffer_rsrc(rsrc_for(Q,
                                                    q_batch,
                                                    head_q,
                                                    q_row_off,
                                                    stride_q_batch,
                                                    stride_q_head,
                                                    stride_q_seq,
                                                    sq,
                                                    row_qk))
            do_rsrc = rocdl.get_buffer_rsrc(rsrc_for(DO,
                                                     q_batch,
                                                     head_q,
                                                     q_row_off,
                                                     stride_do_batch,
                                                     stride_do_head,
                                                     stride_do_seq,
                                                     sq,
                                                     row_vo))
            row0, pitch = softmax.lse_rows(ctx, head=head_q, tokens=lse_tokens)
            rsrcs = softmax.lse_delta_rsrcs(LSE, Delta, row0, pitch, sq)
            lse_rsrc, delta_rsrc = [rocdl.get_buffer_rsrc(r) for r in rsrcs]
            return q_rsrc, do_rsrc, lse_rsrc, delta_rsrc, fx.Int32(pitch) * fx.Int32(4)

        ctx = flex_gemm.Ctx(tr, carried=('dk', 'dv', 'sd', 'rows', 'nibs'))
        # Q rows, for softmax.lse_rows.
        ctx.set_q_rows(varlen_bits=varlen_bits,
                       q_batch=q_batch,
                       q_row_off=q_row_off,
                       sq=sq,
                       max_seqlen_q=max_seqlen_q,
                       seqinfo_q0=seqinfo_q0,
                       seqinfo_q1=seqinfo_q1,
                       num_seqlens=num_seqlens,
                       num_head_q=num_head_q)
        lse_tokens = softmax.lse_tokens(ctx)

        # This lane's LSE/Delta rows: 4 * lg + [0, 4) of the tile; r rides in soffset.
        def row_loads(rsrc, pitch_b, q_start):
            voff = fx.Int32(lg * 4) * pitch_b
            return [fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type,
                                                       rsrc,
                                                       voff,
                                                       (fx.Int32(q_start) + fx.Int32(r)) * pitch_b,
                                                       0)) for r in range(4)]

        ctx.c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        ctx.dropout_scale = dropout_scale

        # ---- the q range: row i keeps key j iff 0 <= j < sk and i - win_l <= j <= i + win_r
        # (gSWA, docs/Generalized Sliding Window Attention.md, A-2), resolved per sequence.
        if const_expr(tr.causal):
            win_l, win_r = fmha.resolve_window(window_left, window_right, sq_i32, sk_i32, varlen_bits)
        else:
            win_l = sq_i32
            win_r = sk_i32
        kv_end = smin(kv0_i32 + fx.Int32(tr.block_n), sk_i32)
        q_lo = smax(kv0_i32 - win_r, fx.Int32(0))
        q_hi = smin(kv_end - fx.Int32(1) + win_l + fx.Int32(1), sq_i32)
        alive = (kv0_i32 < sk_i32) & (q_hi > q_lo)
        t0 = fx.Index(alive.select(q_lo, fx.Int32(0))) // BM
        q_base = t0 * BM
        n_tiles = alive.select((fx.Index(q_hi) + BM - 1) // BM - t0, fx.Index(0))

        def n_rel(x):
            return fx.Index(smax(x, fx.Int32(0)))

        # Tile r (from t0) is clean, every (q, kv) inside the band, iff lead <= r < clean_end.
        # Masked tiles [0, j_clean) and the tail bracket the clean pairs; j_clean is even.
        lead = smin((n_rel(kv_end - fx.Int32(1) - win_r - fx.Int32(q_base)) + BM - 1) // BM, n_tiles)
        ce_x = kv0_i32 + win_l - fx.Int32(BM - 1) - fx.Int32(q_base)
        clean_end = (ce_x < fx.Int32(0)).select(fx.Index(0), fx.Index(ce_x) // BM + fx.Index(1))
        clean_end = smin(smax(clean_end, lead), n_tiles)
        j_clean = smin(lead + lead % 2, n_tiles)
        j_clean_end = smax(clean_end, j_clean)
        n_pairs = (j_clean_end - j_clean) // 2
        j_tail = j_clean + n_pairs * 2

        # ---- score hooks: the band mask, the bias and the dropout keep bits
        # Mask: (kv - q + win_l) as unsigned <= win_l + win_r; an empty band fails all.
        _span = win_l + win_r
        _empty = _span < fx.Int32(0)
        band_span = _empty.select(fx.Int32(0), _span)
        band_lane = _empty.select(fx.Int32(-(1 << 30)),
                                  fx.Int32(kv_wave + lc) - fx.Int32(lg * 4) + win_l)

        def mask_scores(s_vals, q_start, b, fill):
            rel0 = band_lane - fx.Int32(q_start) + fx.Int32(16 * b)
            out = []
            for r in range_constexpr(4):
                rel = rel0 - fx.Int32(r)
                out.append((fx.Uint32(rel) > fx.Uint32(band_span)).select(fill, s_vals[r]))
            return out

        if const_expr(bias):
            # (B, H, Sq, Sk), Sk contiguous; rows offset by Q's varlen offset, columns by K's.
            b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(k_row_off + sk_i32), fx.Int64(0))
            b_lane = (fx.Int32(lg * 4) * fx.Int32(stride_b_seq_q) + k_row_off + fx.Int32(kv_wave + lc)) * fx.Int32(2)

        def bias_rsrc_for(head_q):
            b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head_q) * stride_b_head
                       + fx.Int64(q_row_off) * stride_b_seq_q)
            b_addr = fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2)
            return rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(b_addr,
                                                                                     num_records_bytes=b_span * fx.Int64(2)))

        def bias_load(b_rsrc, q_start, b):
            halves = []
            for r in range_constexpr(4):
                soff = (fx.Int32(q_start) + fx.Int32(r)) * fx.Int32(stride_b_seq_q) * fx.Int32(2)
                halves.append(fx.Int16(rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, b_lane + fx.Int32(32 * b), soff, 0)))
            v = Vec(Vec.from_elements(halves, fx.Int16).bitcast(elem))
            return [fx.Float32(v[r].to(fx.Float32)) for r in range(4)]

        if const_expr(dropout):
            # The Triton kernels' stream (philox.py, grid_plane): offset(row, col) =
            # base + row * stride + col // 4, the element at slot col % 4. Lane (lg, lc)
            # draws q row 4lg + lc % 4, keys 4 * (lc // 4) + [0, 4) of a 16x16 block, so
            # each lane quad holds the 4x4 block its S rows need: a DPP broadcast deals it.
            rng, ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            ph_q = fx.Int32(lg * 4 + lc % 4)
            ph_col = fx.Int32(kv_wave) + fx.Int32((lc // 4) * 4)
            keep_bit = fx.Int32(1) << fx.Int32(lc % 4)

        def dropout_nibs(hs, q_start):
            """Per kv block, this lane's 4 keep bits of tile q_start."""
            if const_expr(not dropout):
                return []
            ph_tile = hs['ph_lane'] + fx.Int64(q_start) * hs['ph_stride']
            nibs = []
            for b in range_constexpr(KVB):
                vals = rng.u32(ph_seed, ph_tile + fx.Int64(4 * b))
                bits = [(fx.Int32(vals[i]) > fx.Int32(idropout_p)).select(fx.Int32(1 << i), fx.Int32(0))
                        for i in range(4)]
                nibs.append((bits[0] | bits[1]) | (bits[2] | bits[3]))
            return nibs

        def dropout_keep(nib):
            """Keep flags of S rows r = 0..3 of a block, from the quad's nibbles."""
            keep = []
            for r in range_constexpr(4):
                # quad_perm [r, r, r, r]: row r's nibble, from lane r of the quad.
                got = fx.Int32(rocdl.update_dpp(fx.Int32.ir_type, raw(fx.Int32(0)), raw(nib), 0x55 * r, 0xF, 0xF, True))
                keep.append((got & keep_bit) != fx.Int32(0))
            return keep

        # ---- memory: tile rows (LSE, Delta) to VGPRs, Q/dO to the rows ring, and
        # the S/dP GEMMs that read them.
        def interleave(n_mfma, n_other, mask):
            """Ask the scheduler for one MFMA, then `n_other` of `mask`, repeated."""
            if const_expr(sched_valu > 0):
                for _ in range_constexpr(n_mfma):
                    rocdl.sched_group_barrier(0x008, 1, 0)
                    rocdl.sched_group_barrier(mask, n_other, 0)

        def rows_load(hs, q_start):
            return (row_loads(hs['lse_rsrc'], hs['pitch_b'], q_start)
                    + row_loads(hs['delta_rsrc'], hs['pitch_b'], q_start))

        def rows_dma(hs, q_start, slot):
            gemm_dma_swz(qs, q_dma, dma_m0, hs['q_rsrc'], scopes, q_start, q_row_stride_b, slot)
            gemm_dma_swz(dops, do_dma, dma_m0, hs['do_rsrc'], scopes, q_start, do_row_stride_b, slot)

        def gemm_sdp(slot):
            """S and dP of the tile in rows slot `slot`, one accumulator per kv block."""
            qr = gemm_read(qs, rm_plan, scopes, slot, masks=qk_masks)
            dor = gemm_read(dops, rm_plan, scopes, slot, masks=vo_masks)
            return gemm_mma(qs, qr, k_packs) + gemm_mma(dops, dor, v_packs)

        def lse_delta(rows):
            lse_l2 = [fx.Int32(x).bitcast(fx.Float32) * fx.Float32(LOG2E) for x in rows[:4]]
            delta = [fx.Int32(x).bitcast(fx.Float32) for x in rows[4:8]]
            return lse_l2, delta

        # ---- the tile loop. Rows slot j % 2 holds tile j's Q/dO, transposed slot
        # j % 2 its Q^T/dO^T; `b` = j % 2 is a Python int (alias scopes need constants).
        def iteration(j, b, masked, hs):
            q_j = q_base + j * BM
            nibs = ctx.nibs if const_expr(early) else dropout_nibs(hs, q_j)
            # A: S/dP(j+1) on the MFMA pipe while P/dS(j) runs on the VALU; rows of
            # j+1 load, Q/dO of j+2 DMA; dV/dK's operands read early to hide latency.
            rows_next = rows_load(hs, q_j + BM)
            rows_dma(hs, q_j + 2 * BM, b)
            sd_next = gemm_sdp(1 - b)
            dot = gemm_read(dvt, t_rd, scopes, b)
            qt = gemm_read(dkt, t_rd, scopes, b)
            lse_l2, delta = lse_delta(ctx.rows)
            p, ds = softmax.p_ds_kv(ctx,
                                    ctx.sd[:KVB],
                                    ctx.sd[KVB:],
                                    lse_l2,
                                    delta,
                                    bias_of=lambda kb: bias_load(hs['b_rsrc'], q_j, kb),
                                    mask_of=lambda vals, kb, fill: mask_scores(vals, q_j, kb, fill),
                                    keep_of=lambda kb: dropout_keep(nibs[kb]),
                                    masked=masked)
            interleave(n_mfma_a, sched_valu, 0x002 | 0x400)
            rocdl.sched_barrier(0)
            # B: dV/dK(j) while tile j+1's transposed half is built in LDS.
            ctx.dv = gemm_mma(dvt, dot, p, ctx.dv)                    # dV^T += dO^T P
            ctx.dk = gemm_mma(dkt, qt, ds, ctx.dk)                    # dK^T += Q^T dS
            gemm_transpose(dkt, t_plan, scopes, 1 - b, 1 - b)
            ctx.nibs = dropout_nibs(hs, q_j + BM) if const_expr(early) else []
            interleave(n_mfma_b, 1, 0x002 | 0x200)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(0)
            cluster_barrier()
            ctx.sd = sd_next
            ctx.rows = rows_next

        # ---- per q head of this kv head's group: dK/dV accumulate over all of them
        init = [fx.full(4, 0.0, fx.Float32) for _ in range(dkt.n_acc + dvt.n_acc)]
        res_h = init
        for hh, accs_h in range(0, group, 1, init=init):
            head_q = head_k * group + hh
            q_rsrc, do_rsrc, lse_rsrc, delta_rsrc, pitch_b = head_state(head_q)
            hs = dict(q_rsrc=q_rsrc, do_rsrc=do_rsrc, lse_rsrc=lse_rsrc, delta_rsrc=delta_rsrc, pitch_b=pitch_b)
            if const_expr(bias):
                hs['b_rsrc'] = bias_rsrc_for(head_q)
            if const_expr(dropout):
                plane = z_i32 * fx.Int32(num_head_q) + fx.Int32(head_q)
                ph_base, hs['ph_stride'] = rng.grid_plane(ph_off, plane, max_seqlen_q, max_seqlen_k)
                hs['ph_lane'] = rng.grid_offset(ph_base, hs['ph_stride'], ph_q, ph_col)

            # Prologue: rows of tiles 0 and 1, transposed tile 0, S/dP(0).
            rows0 = rows_load(hs, q_base)
            rows_dma(hs, q_base, 0)
            rows_dma(hs, q_base + BM, 1)
            rocdl.s_waitcnt(0)
            cluster_barrier()
            gemm_transpose(dkt, t_plan, scopes, 0, 0)
            rocdl.s_waitcnt(0)
            cluster_barrier()
            sd0 = gemm_sdp(0)
            # Iteration 0 refills rows slot 0: every wave must be done reading it.
            rocdl.s_waitcnt(0)
            cluster_barrier()
            accs_h = list(accs_h)
            ctx.dk = accs_h[:dkt.n_acc]
            ctx.dv = accs_h[dkt.n_acc:]
            ctx.sd = sd0
            ctx.rows = rows0
            ctx.nibs = dropout_nibs(hs, q_base) if const_expr(early) else []
            ctx.tiles(lambda j, b, masked: iteration(j, b, masked, hs),
                      j_clean,
                      j_clean_end,
                      n_tiles,
                      tail_start=j_tail,
                      n_trips=n_pairs)
            res_h = yield ctx.dk + ctx.dv

        # ---- epilogue: dK = sm_scale * dK^T, dV (times 1/(1-p) if dropout scales it here), transposed by the store
        dk_rsrc = rsrc_for(DK, k_batch, head_k, k_row_off, stride_dk_batch, stride_dk_head, stride_dk_seq, sk, row_qk)
        dv_rsrc = rsrc_for(DV, k_batch, head_k, k_row_off, stride_dv_batch, stride_dv_head, stride_dv_seq, sk, row_vo)
        dk_row_stride = fx.Index(fx.Int32(stride_dk_seq))
        dv_row_stride = fx.Index(fx.Int32(stride_dv_seq))
        scale_v = fx.full(4, fx.Float32(sm_scale), fx.Float32)
        dv_scale = None
        if const_expr(dropout and tr.dropout_scale_at == softmax.P_DROP_AT_STORE):
            dv_scale = lambda: fx.full(4, fx.Float32(dropout_scale), fx.Float32)  # noqa: E731
        res_h = list(res_h)
        tensor_io.store_acc_t(dkt,
                              [(res_h[:dkt.n_acc], dk_rsrc, dk_row_stride, row_qk, scale_v),
                               (res_h[dkt.n_acc:], dv_rsrc, dv_row_stride, row_vo, dv_scale)],
                              kv_wave,
                              lc,
                              lg,
                              sk,
                              elem,
                              tr.pad_floor)

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx942(
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
        n_kv = (fx.Index(max_seqlen_k) + tr.block_n - 1) // tr.block_n
        value_attrs = {'rocdl.waves_per_eu': tr.waves_per_eu,
                       'rocdl.flat_work_group_size': f'{tr.block_size},{tr.block_size}'}
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
                                    value_attrs=value_attrs).launch(
            grid=(n_kv, fx.Index(num_head_k), grid_z) if tile_fastest else (fx.Index(num_head_k), n_kv, grid_z),
            block=(tr.block_size, 1, 1),
            stream=stream)

    return host.make_dkdv_launch(launch_fmha_bwd_dkdv_gfx942,
                                 tr,
                                 causal_type,
                                 host_causal_type,
                                 bias,
                                 dropout)


_MASK_KNOBS = ('mask_at', 'dropout_scale_at')
_WIDE_KNOBS = ('kv_subs', 'waves_per_eu') + _MASK_KNOBS
_BODIES = {
    'narrow': (__name__,
               'build_fmha_bwd_dkdv_gfx942',
               ('num_waves', 'kv_blocks', 'waves_per_eu', 'interleave_valu') + _MASK_KNOBS),
    'wide': (f'{__package__}.dkdv_wide', 'build_fmha_bwd_dkdv_gfx942_wide', _WIDE_KNOBS),
    'xwide': (f'{__package__}.dkdv_xwide', 'build_fmha_bwd_dkdv_gfx942_xwide', _WIDE_KNOBS),
    'xxwide': (f'{__package__}.dkdv_xxwide', 'build_fmha_bwd_dkdv_gfx942_xxwide', _WIDE_KNOBS),
}


def build_fmha_bwd_dkdv_gfx942_module_primary(meta, knobs):
    """The body `knobs.body` names, built for a resolved (meta, knobs) pair:
    `fmha_meta.FmhaInputMetadata` and `tuning_dkdv.entry_resolve(meta)`."""
    return build_body_primary(_BODIES, meta, knobs)


def build_fmha_bwd_dkdv_gfx942_by_head_dim(head_dim, body=None, **kwargs):
    """A JIT build of the compiled `head_dim` by `body`'s builder (default: the
    body the policy routes `head_dim` to), with that builder's keywords and defaults."""
    return body_builder(_BODIES, body or tuning.body(head_dim))(head_dim, **kwargs)
