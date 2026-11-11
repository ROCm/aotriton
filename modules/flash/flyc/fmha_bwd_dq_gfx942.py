# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention backward, dQ and dB, for gfx942, FlyDSL.

    S^T  = K.Q^T                P = exp2(sm_scale*log2e*S + log2e*bias - lse2)
    dP^T = V.dO^T               dS = P * (dP - Delta)
    dQ^T += K^T.dS^T            dQ = sm_scale * dQ;  dB = dS

All three in the transposed form, as the gfx942 forward: S^T's MFMA output
layout is dS^T's operand layout. GEMM3 reads K^T out of the row-major K tile:
MFMA `dt` of a d-interleave group owns d = g*m + dt, so one ds_read of g
adjacent columns at 4 KV rows feeds g MFMAs after 2g v_perm.

The kernel ABI and `_launch` interface are fmha_bwd_dq_gfx1201_kernel's.
"""


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
import fmha_tuning_bwd_dq_gfx942 as tuning
from philox import Philox

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dq_gfx942.lds">'
NBUF = 3
LDS_SCOPES = tuple(f'{t}{b}' for t in 'kv' for b in range(NBUF))
LOG2E = 1.4426950408889634
WAVE = 64
WAVE_M = 32         # query rows per wave: the N of the 32x32x8 MFMA


def kt_groups(head_dim):
    """GEMM3's d-interleave groups: (first column, g, first accumulator)."""
    widths = {32: (1,), 64: (2,), 96: (2, 1), 128: (4,)}[head_dim]
    out, d0, a0 = [], 0, 0
    for g in widths:
        out.append((d0, g, a0))
        d0 += 32 * g
        a0 += g
    return tuple(out)


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, num_waves, block_n, padded_head=False):
        assert head_dim in tuning.LADDER, head_dim
        assert block_n in (32, 64), block_n
        self.D = head_dim
        self.PADDED = padded_head
        # Columns below PAD_FLOOR are real in every call this build serves.
        self.PAD_FLOOR = 0 if padded_head else head_dim
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NUM_WAVES = num_waves
        self.BLOCK_SIZE = num_waves * WAVE
        self.BLOCK_M = num_waves * WAVE_M
        self.BLOCK_N = block_n
        self.NSUB = block_n // 32               # 32x32 S^T sub-tiles per KV tile
        self.NS = 16 * self.NSUB                # S values per lane
        self.QK_STEPS = head_dim // 8           # MFMA k-steps of GEMM1 and GEMM2
        self.KV_STEPS = block_n // 8            # MFMA k-steps of GEMM3
        self.D_TILES = head_dim // 32           # dQ^T accumulators
        self.KT_GROUPS = kt_groups(head_dim)
        # K and V rows sit an odd number of 16B chunks apart (the forward's K tile).
        self.ROW_BYTES = head_dim * 2
        chunks = head_dim // 8
        self.STRIDE = (chunks + 1 - chunks % 2) * 16
        self.DMA_ROUND_BYTES = self.BLOCK_SIZE * 4
        self.DMA_ROUNDS = -(-block_n * self.STRIDE // self.DMA_ROUND_BYTES)
        self.TILE_BYTES = self.DMA_ROUNDS * self.DMA_ROUND_BYTES
        self.LDS_K_BASE = 0
        self.LDS_V_BASE = NBUF * self.TILE_BYTES
        self.LDS_BYTES = 2 * NBUF * self.TILE_BYTES
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        # The JIT cache keys on scalar closure values; the kernel references this.
        self.CACHE_TAG = f'D{head_dim}_c{int(causal)}_{dtype_str}_w{num_waves}_n{block_n}_p{int(padded_head)}'


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
    """A launcher with fmha_bwd_dq_gfx1201_kernel's call interface and kernel ABI.

    `causal_type` as gfx1201's: 1/2 are host sentinels, 3 is gSWA; all compile to 3.
    """
    host_causal_type = (1 if causal else 0) if causal_type is None else causal_type
    if host_causal_type not in (0, 1, 2, 3):
        raise ValueError(f'causal_type must be 0-3, got {host_causal_type}')
    causal_type = 0 if host_causal_type == 0 else 3
    if bias and causal_type:
        raise ValueError('bias with a causal mask is not served (aot/_common.py flash_disabled)')
    if block_n is None:
        block_n = tuning.default_block_n(head_dim)
    tr = Traits(head_dim, causal_type != 0, dtype_str, num_waves, block_n, padded_head)
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(tr.CAUSAL)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    has_bias = bool(bias)
    if lpt is None:
        lpt = tuning.default_lpt(tr.CAUSAL)
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}_l{int(lpt)}'
    elem = fx.BFloat16 if dtype_str == 'bf16' else fx.Float16

    def v4e():
        return Vec.make_type(4, elem)

    def v8e():
        return Vec.make_type(8, elem)

    def v16f():
        return Vec.make_type(16, fx.Float32)

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, tr.LDS_BYTES, 16]

    def mfma(a, b, c):
        if const_expr(tr.DTYPE == 'bf16'):
            return rocdl.mfma_f32_32x32x8bf16_1k(v16f(), [Vec(a).bitcast(fx.Int16), Vec(b).bitcast(fx.Int16), c])
        return rocdl.mfma_f32_32x32x8f16(v16f(), [a, b, c])

    def exp2(x):
        return fx.Float32(rocdl.exp2(T.f32, raw(x)))

    def pack4(vals):
        """Four f32 -> one operand. bf16: round to nearest, ties away, NaN-unsafe,
        as fmha_common_gfx1201.bf16_round_nonan_pack_v8; never truncate."""
        if const_expr(tr.DTYPE == 'bf16'):
            words = []
            for h in range_constexpr(2):
                rnd = [fx.Float32(x).bitcast(fx.Int32) + fx.Int32(0x8000) for x in (vals[2 * h], vals[2 * h + 1])]
                words.append(fx.Int32(rocdl.perm_b32(rnd[1], rnd[0], fx.Int32(0x07060302))))
            return Vec.from_elements(words, fx.Int32).bitcast(elem).ir_value()
        return Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem).ir_value()

    def lds_ptr_at(base_i64):
        return buffer_ops.create_llvm_ptr(fx.Int64(base_i64), address_space=3)

    def scope_attrs(name):
        """LDS alias scopes; without them LLVM waits vmcnt(0) for any DMA before any LDS read."""
        def arr(names):
            return ir.Attribute.parse('[' + ', '.join(
                f'#llvm.alias_scope<id = "{n}", domain = {_LDS_DOMAIN}>' for n in names) + ']')
        return arr([name]), arr([n for n in LDS_SCOPES if n != name])

    def cluster_barrier():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    def barrier_if(flag_i32):
        """s_barrier on the waves whose wave-uniform `flag_i32` is nonzero. Clobbers SCC:
        undeclared, a live SCC across it misroutes a branch and the barriers mismatch."""
        llvm.inline_asm(ir.Type.parse('!llvm.void'),
                        [raw(flag_i32)],
                        's_cmp_eq_u32 $0, 0\n\ts_cbranch_scc1 1f\n\ts_barrier\n\t1:',
                        's,~{scc}',
                        has_side_effects=True)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
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
        _ = cache_tag
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
        if const_expr(lpt):
            # Longest first: under causal a q tile's work grows with its index.
            q_tile = (fx.Index(max_seqlen_q) + tr.BLOCK_M - 1) // tr.BLOCK_M - fx.Index(1) - q_tile
        z_i32 = fx.Int32(gpu.block_idx.z)
        head_k = head // (fx.Index(num_head_q) // fx.Index(num_head_k))
        sq_i32, q_row_off, q_batch = fmha.decode_addressing(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32)
        sk_i32, k_row_off, k_batch = fmha.decode_addressing(varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32)
        sq = fx.Index(sq_i32)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_base = fx.Index(fx.ptrtoint(lds_p))

        def lds_view(off, e, align):
            return fx.recast_iter(fx.PointerType.get(e.ir_type, fx.AddressSpace.Shared, align),
                                  lds_p + fx.Int64(off))

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)

        def rsrc_for(ptr, batch, h, row_off, sb, sh, ss, rows, row_elems):
            # One sequence of one head, ending on its last row's last owned column.
            elems = fx.Int64(batch) * sb + fx.Int64(h) * sh + fx.Int64(row_off) * ss
            base = fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(2)
            span = (fx.Int64(rows) - fx.Int64(1)) * ss + fx.Int64(row_elems)
            span = (fx.Int32(rows) > fx.Int32(0)).select(span, fx.Int64(0))
            return buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=span * fx.Int64(2))

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
        k_row_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_row_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)

        q_start = q_tile * tr.BLOCK_M
        q_wave = q_start + wave * WAVE_M
        q_row = q_wave + lm
        q_ok = q_row < sq

        # Operand packs: k-step 2t+e reads d = 16t + 8*lh + 4e + [0, 4).
        def col_mask(t, hdim):
            """Dword masks keeping this lane's columns 16t + 8lh + [0, 8) below hdim."""
            c0 = fx.Int32(16 * t) + fx.Int32(lh * 8)
            words = []
            for w in range_constexpr(4):
                lo = (c0 + fx.Int32(2 * w) < fx.Int32(hdim)).select(fx.Int32(0xFFFF), fx.Int32(0))
                hi = (c0 + fx.Int32(2 * w + 1) < fx.Int32(hdim)).select(fx.Int32(-65536), fx.Int32(0))
                words.append(lo | hi)
            return Vec.from_elements(words, fx.Int32)

        def masked8(v8, mask):
            return Vec(Vec(v8).bitcast(fx.Int32) & mask).bitcast(elem).ir_value()

        qk_masks = {t: col_mask(t, hdim_qk) for t in range(tr.QK_STEPS // 2) if 16 * t + 16 > tr.PAD_FLOOR}
        vo_masks = {t: col_mask(t, hdim_vo) for t in range(tr.QK_STEPS // 2) if 16 * t + 16 > tr.PAD_FLOOR}

        def row_packs(rsrc, row_stride, masks):
            # Slack columns may hold NaN, and 0 * NaN is NaN: both operands are zeroed there.
            packs = []
            for t in range_constexpr(tr.QK_STEPS // 2):
                off = q_row * row_stride + fx.Index(16 * t) + lh * 8
                v8 = buffer_ops.buffer_load(rsrc, off, vec_width=8, dtype=elem)
                if const_expr(t in masks):
                    v8 = masked8(v8, masks[t])
                packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
                packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())
            return packs

        q_packs = row_packs(q_rsrc, q_row_stride, qk_masks)
        do_packs = row_packs(do_rsrc, do_row_stride, vo_masks)

        # LSE (natural base, +inf on rows that see no key) and Delta share one layout.
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)
        lse_row0, lse_pitch = fmha.lse_row_addressing(varlen_bits,
                                                      fx.Index(q_batch),
                                                      head,
                                                      num_head_q,
                                                      lse_tokens,
                                                      fx.Index(q_row_off))

        def row_scalar(ptr):
            addr = fx.Int64(fx.ptrtoint(ptr))
            rsrc = buffer_ops.create_buffer_resource_from_addr(addr + fx.Int64(lse_row0) * fx.Int64(4),
                                                               num_records_bytes=fx.Int64(sq * lse_pitch) * fx.Int64(4))
            return fx.Float32(buffer_ops.buffer_load(rsrc, q_row * lse_pitch, vec_width=1, dtype=fx.Float32))

        # A row past seqlen_q gets lse2 = +inf, so its P, dS and dB are exactly 0.
        lse2 = q_ok.select(row_scalar(LSE) * fx.Float32(LOG2E), fx.Float32(float('inf')))
        delta = q_ok.select(row_scalar(Delta), fx.Float32(0.0))

        # K and V DMA, the forward's K path. The tile offset rides in soffset.
        dma_voff_k = []
        dma_voff_v = []
        dma_m0 = []
        for i in range_constexpr(tr.DMA_ROUNDS):
            lds_line = fx.Index(i * tr.DMA_ROUND_BYTES) + wave * (WAVE * 4)
            p = lds_line + lane * 4
            r = p // tr.STRIDE
            c = p % tr.STRIDE
            pad = (c >= fx.Index(tr.ROW_BYTES)) | (r >= fx.Index(tr.BLOCK_N))
            dma_voff_k.append(pad.select(fx.Int32(0), fx.Int32(r) * k_row_stride_b + fx.Int32(c)))
            dma_voff_v.append(pad.select(fx.Int32(0), fx.Int32(r) * v_row_stride_b + fx.Int32(c)))
            dma_m0.append(fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_base + lds_line))))

        def kv_dma(kv_start, buf):
            for rsrc, stride_b, voffs, base, name in ((k_rsrc, k_row_stride_b, dma_voff_k, tr.LDS_K_BASE, 'k'),
                                                      (v_rsrc, v_row_stride_b, dma_voff_v, tr.LDS_V_BASE, 'v')):
                soff = fx.Int32(kv_start) * stride_b
                al, noal = scope_attrs(f'{name}{buf}')
                for i in range_constexpr(tr.DMA_ROUNDS):
                    m0 = fx.Int64(dma_m0[i] + fx.Int32(base + buf * tr.TILE_BYTES))
                    op = rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                       lds_ptr_at(m0),
                                                       fx.Int32(4),
                                                       voffs[i],
                                                       soff,
                                                       fx.Int32(0),
                                                       fx.Int32(0))
                    op = getattr(op, 'owner', op)
                    op = getattr(op, 'operation', op)
                    op.attributes['alias_scopes'] = al
                    op.attributes['noalias_scopes'] = noal

        def lds_load(off, ty, align, scope):
            al, noal = scope_attrs(scope)
            return llvm.LoadOp(ty,
                               lds_view(off, elem, align).llvm_ptr,
                               alignment=align,
                               alias_scopes=al,
                               noalias_scopes=noal).result

        # GEMM1/GEMM2 A operand: KV row lm (+32n), 16B chunk 2t + lh.
        row_rd_off = [lm * tr.STRIDE + (fx.Index(2 * t) + lh) * 16 for t in range(tr.QK_STEPS // 2)]

        def tile_read(base, buf, name, masks):
            out = []
            for n in range_constexpr(tr.NSUB):
                for t in range_constexpr(tr.QK_STEPS // 2):
                    off = row_rd_off[t] + base + buf * tr.TILE_BYTES + n * 32 * tr.STRIDE
                    v8 = lds_load(off, v8e(), 16, f'{name}{buf}')
                    if const_expr(t in masks):
                        v8 = masked8(v8, masks[t])
                    out.append(v8)
            return out

        def gemm_t(rows8, packs):
            """[v16f32] * NSUB: rows8 . packs^T over the head dim."""
            out = []
            nt = tr.QK_STEPS // 2
            for n in range_constexpr(tr.NSUB):
                s = fx.full(16, 0.0, fx.Float32)
                for t in range_constexpr(nt):
                    r8 = Vec(rows8[n * nt + t])
                    s = mfma(r8.shuffle(r8, [0, 1, 2, 3]).ir_value(), packs[2 * t], s)
                    s = mfma(r8.shuffle(r8, [4, 5, 6, 7]).ir_value(), packs[2 * t + 1], s)
                out.append(s)
            return out

        # GEMM3 A operand: K[kv = 8s + 4lh + j][d0 + g*lm + dt], j = 0..3.
        kt_rd_off = {g: lh * (4 * tr.STRIDE) + lm * (2 * g) for g in {g for _, g, _ in tr.KT_GROUPS}}
        sel_lo = 0x05040100
        sel_hi = 0x07060302

        def kt_read(buf, d0, g, s):
            base = kt_rd_off[g] + tr.LDS_K_BASE + buf * tr.TILE_BYTES + 8 * s * tr.STRIDE + 2 * d0
            rows = [base + j * tr.STRIDE for j in range(4)]
            scope = f'k{buf}'
            if const_expr(g == 1):
                hs = [fx.Int16(lds_load(r, T.i16, 2, scope)) for r in rows]
                return [Vec.from_elements(hs, fx.Int16).bitcast(elem).ir_value()]
            if const_expr(g == 2):
                words = [[fx.Int32(lds_load(r, T.i32, 4, scope))] for r in rows]
            else:
                words = []
                for r in rows:
                    w2 = Vec(lds_load(r, Vec.make_type(2, fx.Int32), 8, scope))
                    words.append([fx.Int32(w2[0]), fx.Int32(w2[1])])
            ops = []
            for dt in range_constexpr(g):
                w = dt // 2
                sel = fx.Int32(sel_hi if dt % 2 else sel_lo)
                lo = fx.Int32(rocdl.perm_b32(words[1][w], words[0][w], sel))
                hi = fx.Int32(rocdl.perm_b32(words[3][w], words[2][w], sel))
                ops.append(Vec.from_elements([lo, hi], fx.Int32).bitcast(elem).ir_value())
            return ops

        def gemm3(dq_accs, ds_packs, buf):
            out = list(dq_accs)
            for d0, g, a0 in tr.KT_GROUPS:
                for s in range_constexpr(tr.KV_STEPS):
                    ops = kt_read(buf, d0, g, s)
                    for dt in range_constexpr(g):
                        out[a0 + dt] = mfma(ops[dt], ds_packs[s], out[a0 + dt])
            return out

        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        c_neg_inf = fx.Float32(float('-inf'))
        kv_lane = fx.Int32(lh * 4)

        def smin(x, y):
            return (x < y).select(x, y)

        def smax(x, y):
            return (x > y).select(x, y)

        # Row i keeps key j iff 0 <= j < sk and i - win_l <= j <= i + win_r (gSWA,
        # docs/Generalized Sliding Window Attention.md, A-2), resolved per sequence.
        if const_expr(tr.CAUSAL):
            win_l, win_r = fmha.resolve_window(window_left, window_right, sq_i32, sk_i32, varlen_bits)
        else:
            win_l = sq_i32
            win_r = sk_i32
        q_first = fx.Int32(q_start)
        q_last = smin(q_first + fx.Int32(tr.BLOCK_M), sq_i32) - fx.Int32(1)
        kv_lo = smax(q_first - win_l, fx.Int32(0))
        kv_hi = smin(q_last + win_r + fx.Int32(1), sk_i32)
        alive = (q_first < sq_i32) & (kv_hi > kv_lo)
        t0 = fx.Index(alive.select(kv_lo, fx.Int32(0))) // tr.BLOCK_N
        kv_base = t0 * tr.BLOCK_N
        n_tiles = alive.select((fx.Index(kv_hi) + tr.BLOCK_N - 1) // tr.BLOCK_N - t0, fx.Index(0))

        # This lane's row keeps keys [band_lo, band_lo + band_span]: one unsigned compare per key.
        _row = fx.Int32(q_row)
        _hi = smin(_row + win_r, sk_i32 - fx.Int32(1))
        _lo = smax(_row - win_l, fx.Int32(0))
        _empty = _hi < _lo
        band_lo = _empty.select(fx.Int32(-(1 << 30)), _lo)
        band_span = _empty.select(fx.Int32(0), _hi - _lo)

        def kv_of(r):
            """KV row of S^T value r, relative to the tile, without this lane's 4*lh."""
            return 32 * (r // 16) + 8 * ((r % 16) // 4) + r % 4

        def mask_scores(s_vals, kv_start):
            rel0 = fx.Int32(kv_start) + kv_lane - band_lo
            return [(fx.Uint32(rel0 + fx.Int32(kv_of(r))) > fx.Uint32(band_span)).select(c_neg_inf, s_vals[r])
                    for r in range(tr.NS)]

        # Tile r (from t0) is clean, i.e. unmasked for every row of the block, iff lead <= r < clean_end.
        def n_rel(kv):
            return fx.Index(smax(smin(kv, sk_i32), fx.Int32(0)))

        lead = (n_rel(q_last - win_l - fx.Int32(kv_base)) + tr.BLOCK_N - 1) // tr.BLOCK_N
        clean_end = n_rel(smin(q_first + win_r, sk_i32 - fx.Int32(1)) + fx.Int32(1) - fx.Int32(kv_base)) // tr.BLOCK_N
        clean_end = smin(clean_end, n_tiles)
        if const_expr(bias):
            # Only the last tile can hold the dword that straddles the bias slab's end.
            clean_end = smin(clean_end, smax(n_tiles, fx.Index(1)) - fx.Index(1))
        # Masked tiles [0, j_clean) and [j_clean_end, n_tiles) bracket the clean ones;
        # j_clean is a multiple of NBUF so the unrolled loop knows its buffers.
        j_clean = smin(lead + (fx.Index(NBUF) - lead % NBUF) % NBUF, n_tiles)
        j_clean_end = smax(j_clean + (smax(clean_end, j_clean) - j_clean) // NBUF * NBUF, j_clean)

        if const_expr(bias):
            # (B, H, Sq, Sk), Sk contiguous; rows offset by Q's varlen offset, columns by K's.
            b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head) * stride_b_head
                       + fx.Int64(q_row_off) * stride_b_seq_q)
            # Rows of an odd-pitch bias start mid-dword: the descriptor base is aligned
            # down, and each lane funnel-shifts its 4 keys out of 3 aligned dwords.
            b_addr = fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2)
            b_mis = fx.Int32(b_addr & fx.Int64(3))
            b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                b_addr - fx.Int64(b_mis),
                num_records_bytes=b_span * fx.Int64(2) + fx.Int64(b_mis)))
            b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + k_row_off + kv_lane) * fx.Int32(2)
            b_lane = b_byte & fx.Int32(-4)
            b_shift = (b_byte & fx.Int32(3)) * fx.Int32(8)
            # dB = dS. A caller that wants none passes all-zero dB strides (and maybe a
            # null DB): an empty range then drops every store.
            store_db = ((fx.Int64(stride_db_batch) != fx.Int64(0))
                        | (fx.Int64(stride_db_head) != fx.Int64(0))
                        | (fx.Int64(stride_db_seq_q) != fx.Int64(0)))
            db_elems = (fx.Int64(q_batch) * stride_db_batch + fx.Int64(head) * stride_db_head
                        + fx.Int64(q_row_off) * stride_db_seq_q)
            db_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_db_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            db_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(DB)) + db_elems * fx.Int64(2),
                num_records_bytes=store_db.select(db_span * fx.Int64(2), fx.Int64(0))))
            db_lane = (fx.Int32(q_row) * fx.Int32(stride_db_seq_q) + kv_lane) * fx.Int32(2)

        def bias_load(kv_start, exact):
            """This lane's bias for one KV tile, in S^T order: 4 keys per 8-key group.

            The hardware drops a dword not wholly in range, which can take the slab's
            last element with it; `exact` tiles read 2 bytes at a time instead."""
            if const_expr(not bias):
                return None
            soff = fx.Int32(kv_start) * fx.Int32(2)
            out = []
            for g in range_constexpr(tr.NS // 4):
                if const_expr(exact):
                    off = b_byte + fx.Int32((32 * (g // 4) + 8 * (g % 4)) * 2)
                    hs = [rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, off + fx.Int32(2 * e), soff, 0) for e in range(4)]
                    out.append(Vec.from_elements([fx.Int16(h) for h in hs], fx.Int16).bitcast(elem).ir_value())
                    continue
                off = b_lane + fx.Int32((32 * (g // 4) + 8 * (g % 4)) * 2)
                dw = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(3, fx.Int32), b_rsrc, off, soff, 0))
                words = [fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.fshr.i32',
                                                      [raw(dw[w + 1]), raw(dw[w]), raw(b_shift)], [], []))
                         for w in range(2)]
                out.append(Vec.from_elements(words, fx.Int32).bitcast(elem).ir_value())
            return out

        def db_store(ds_packs, kv_start, masked):
            """dB = dS, the packed GEMM3 operand: 4 keys per k-step, 2 bytes at a time
            since a dB row may start mid-dword."""
            soff = fx.Int32(kv_start) * fx.Int32(2)
            for s in range_constexpr(tr.KV_STEPS):
                hs = Vec(ds_packs[s]).bitcast(fx.Int16)
                for e in range_constexpr(4):
                    off = db_lane + fx.Int32((8 * s + e) * 2)
                    ok = q_ok
                    if const_expr(masked):
                        ok = ok & (fx.Int32(kv_start) + kv_lane + fx.Int32(8 * s + e) < sk_i32)
                    off = ok.select(off, fx.Int32(0x7FFFFFFF))
                    rocdl.raw_ptr_buffer_store(raw(fx.Int16(hs[e])), db_rsrc, raw(off), raw(soff))

        if const_expr(dropout):
            # The Triton kernels' stream, element for element (philox.py, grid_plane).
            rng = Philox.for_arch('gfx942')
            ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            ph_plane = z_i32 * fx.Int32(num_head_q) + fx.Int32(head)
            ph_base, ph_stride = rng.grid_plane(ph_off, ph_plane, max_seqlen_q, max_seqlen_k)

        def dropout_keep(kv_start):
            """Keep flags in S^T order: each 4-key group is one Philox call."""
            keep = []
            for g in range_constexpr(tr.NS // 4):
                col = fx.Int32(kv_start) + kv_lane + fx.Int32(32 * (g // 4) + 8 * (g % 4))
                vals = rng.u32(ph_seed, rng.grid_offset(ph_base, ph_stride, fx.Int32(q_row), col))
                keep.extend(fx.Int32(v) > fx.Int32(idropout_p) for v in vals)
            return keep

        def p_of(s_raw, kv_start, masked, bias_vals):
            """P = exp2(sm_scale*log2e*S + log2e*bias - lse2), [v16f32] * NSUB."""
            # Scale once in fp32, as the forward: P must be the forward's P bit for bit.
            c_scale_v = fx.full(16, fx.Float32(c_scale), fx.Float32)
            scaled = [Vec(sr) * c_scale_v for sr in s_raw]
            s_vals = [scaled[r // 16][r % 16] for r in range(tr.NS)]
            if const_expr(bias):
                s_vals = [s_vals[r] + fx.Float32(Vec(bias_vals[r // 4])[r % 4].to(fx.Float32)) * fx.Float32(LOG2E)
                          for r in range(tr.NS)]
            if const_expr(masked):
                s_vals = mask_scores(s_vals, kv_start)
            # Whole-accumulator vector math, so v_pk_* ops pair adjacent registers.
            lse2_v = fx.full(16, fx.Float32(lse2), fx.Float32)
            out = []
            for n in range_constexpr(tr.NSUB):
                x = Vec.from_elements([s_vals[16 * n + r] for r in range(16)], fx.Float32) - lse2_v
                out.append(Vec.from_elements([exp2(x[r]) for r in range(16)], fx.Float32).ir_value())
            return out

        def ds_packs_of(p_vals, dp_raw, kv_start):
            """dS = P * (dP - Delta), packed as GEMM3's B operand."""
            delta_v = fx.full(16, fx.Float32(delta), fx.Float32)
            if const_expr(dropout):
                keep_all = dropout_keep(kv_start)
            ds = []
            for n in range_constexpr(tr.NSUB):
                dp = Vec(dp_raw[n])
                if const_expr(dropout):
                    # The mask and 1/(1-p) hit dP; P stays undropped (numerical lessons, #9).
                    keep = keep_all[16 * n:16 * n + 16]
                    dp = Vec.from_elements([keep[r].select(fx.Float32(dp[r]) * fx.Float32(dropout_scale),
                                                           fx.Float32(0.0)) for r in range(16)],
                                           fx.Float32)
                dsv = Vec(p_vals[n]) * (dp - delta_v)
                ds.extend(dsv[r] for r in range(16))
            return [pack4(ds[4 * i:4 * i + 4]) for i in range(tr.KV_STEPS)]

        # Per tile j, two clusters; waves 4..7 trail by one barrier, so an LDS buffer
        # is safe two clusters after its last reader / its producer's wait:
        #   A(j): GEMM1+GEMM2 read k/v[j%3]; P; waits DMA(j+1)
        #   B(j): DMA(j+2) -> k/v[(j+2)%3]; dS; GEMM3 reads k[j%3]
        # Unrolled by three so buffer indices (and alias scopes) are constants.
        group_b = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                               fx.Int32((wave >= fx.Index(tr.NUM_WAVES // 2)).select(1, 0))))

        rocdl.sched_barrier(0)
        kv_dma(kv_base, 0)
        # The vmcnt below waits for tile 0 only: keep tile 1 after it.
        rocdl.sched_barrier(0)
        kv_dma(kv_base + tr.BLOCK_N, 1)
        rocdl.s_waitcnt(vmcnt=2 * tr.DMA_ROUNDS)
        cluster_barrier()
        barrier_if(group_b)                       # open the skew

        def iteration(j, dq_accs, b, masked):
            """Tile j; `b` = j % 3, known at trace time."""
            kv = kv_base + j * tr.BLOCK_N
            # A(j)
            bias_vals = bias_load(kv, masked)
            s_raw = gemm_t(tile_read(tr.LDS_K_BASE, b, 'k', qk_masks), q_packs)
            # P here, under GEMM2's MFMAs; dS and its pack stay in B.
            p_vals = p_of(s_raw, kv, masked, bias_vals)
            dp_raw = gemm_t(tile_read(tr.LDS_V_BASE, b, 'v', vo_masks), do_packs)
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            # B(j)
            kv_dma(kv + 2 * tr.BLOCK_N, (b + 2) % NBUF)
            rocdl.sched_barrier(0)
            ds_packs = ds_packs_of(p_vals, dp_raw, kv)
            if const_expr(bias):
                db_store(ds_packs, kv, masked)
            dq_accs = gemm3(dq_accs, ds_packs, b)
            cluster_barrier()
            return dq_accs

        @flyc.jit
        def masked_iteration(j, carried):
            out = list(carried)
            if j % NBUF == fx.Index(0):
                out = iteration(j, carried, 0, True)
            else:
                if j % NBUF == fx.Index(1):
                    out = iteration(j, carried, 1, True)
                else:
                    out = iteration(j, carried, 2, True)
            return out

        def as_list(x):
            # A loop carrying one value yields it bare.
            return list(x) if isinstance(x, (list, tuple)) else [x]

        init = [fx.full(16, 0.0, fx.Float32) for _ in range(tr.D_TILES)]
        results0 = init
        for j, carried in range(0, j_clean, 1, init=init):
            results0 = yield masked_iteration(j, carried)
        results0 = as_list(results0)
        n_trips = (j_clean_end - j_clean) // NBUF
        results1 = results0
        for jt, carried in range(0, n_trips, 1, init=results0):
            acc = carried
            for u in range_constexpr(NBUF):
                acc = iteration(j_clean + jt * NBUF + u, acc, u, False)
            results1 = yield acc
        results1 = as_list(results1)
        results2 = results1
        for j, carried in range(j_clean_end, n_tiles, 1, init=results1):
            results2 = yield masked_iteration(j, carried)
        results2 = as_list(results2)
        barrier_if(fx.Int32(1) - group_b)         # close the skew
        rocdl.s_waitcnt(0)

        # dQ = sm_scale * dQ^T, transposed back by the store: accumulator element i
        # of group (d0, g), MFMA dt, is column d0 + g*(8*(i//4) + 4*lh + i%4) + dt.
        dq_scale = fx.full(16, fx.Float32(sm_scale), fx.Float32)
        accs = [Vec(results2[a]) * dq_scale for a in range(tr.D_TILES)]
        for d0, g, a0 in tr.KT_GROUPS:
            for i4 in range_constexpr(4):
                vals = [accs[a0 + dt][4 * i4 + r] for r in range(4) for dt in range(g)]
                for h in range_constexpr(max(1, g // 2)):
                    part = vals[8 * h:8 * h + 8] if g >= 2 else vals
                    d = fx.Index(d0 + g * 8 * i4 + 8 * h) + lh * (4 * g)
                    packed = Vec.from_elements([fx.Float32(x).to(elem) for x in part], elem)
                    ok = q_ok
                    if const_expr(d0 + g * 8 * i4 + 8 * h + 4 * g + len(part) > tr.PAD_FLOOR):
                        ok = ok & (fx.Int32(d) < row_qk)
                    buffer_ops.buffer_store(packed.ir_value(), dq_rsrc, q_row * dq_row_stride + d, mask=ok)

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
        grid_y = (fx.Index(max_seqlen_q) + tr.BLOCK_M - 1) // tr.BLOCK_M
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
                                               'rocdl.flat_work_group_size': f'{tr.BLOCK_SIZE},{tr.BLOCK_SIZE}'}).launch(
            grid=(grid_y, fx.Index(num_head_q), grid_z) if tile_fastest else (fx.Index(num_head_q), grid_y, grid_z),
            block=(tr.BLOCK_SIZE, 1, 1),
            stream=stream)

    compiled = abi.new_compiled_cache()

    def _args(Q,
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
              principal_diagonal=None,
              dropout_p=None,
              philox_seed=0,
              philox_offset1=None,
              philox_offset2=0,
              bias=None,
              dbias=None):
        """fmha_bwd_dq_gfx1201_kernel's `_args`: every kernel argument but the stream."""
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = abi.prep_tensors([("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)],
                                          q_heads=("DO", "DQ"))
        if tr.PADDED:
            if not (0 < max(meta[2], meta[3]) <= tr.D):
                raise ValueError(f'head dims {meta[2:]} do not fit a padded {tr.D} build')
        elif meta[2] != tr.D or meta[3] != tr.D:
            raise ValueError(f'head dims {meta[2:]} on an unpadded {tr.D} build; pass padded_head=True')
        lp = abi.row_tensor_arg(lse, "logsumexp", meta[0], seqlen_q, varlen)
        dp = abi.row_tensor_arg(delta, "delta", meta[0], seqlen_q, varlen)
        wl, wr = abi.resolve_window(causal_type, host_causal_type, window, seqlen_q, seqlen_k)
        vb, sq0, sq1, sk0, sk1, mq, mk = abi.varlen_args(False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        if principal_diagonal is not None:
            pd_mask = abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT
            vb = (vb & ~pd_mask) | abi.varlen_bits(principal_diagonal=principal_diagonal)
        if dropout_p and not dropout:
            raise ValueError('dropout_p= on a build without dropout')
        # `hold` keeps a materialised philox scalar alive across the launch (abi.u64_scalar).
        ps, po1, po2, ip, dsc, hold = abi.dropout_args(dropout,
                                                      dropout_p,
                                                      philox_seed,
                                                      philox_offset1,
                                                      philox_offset2,
                                                      Q.device,
                                                      stream)
        bp, dbp, bs, dbs = abi.bias_args(has_bias, True, bias, dbias, Q)
        packed = (*ptrs[:3],
                  bp,
                  ptrs[3],
                  ptrs[4],
                  dbp,
                  lp,
                  dp,
                  sq0,
                  sq1,
                  sk0,
                  sk1,
                  vb,
                  batch_size,
                  num_seqlens,
                  mq,
                  mk,
                  wl,
                  wr,
                  ps,
                  po1,
                  po2,
                  ip,
                  dsc,
                  *meta,
                  abi.resolve_scale(Q, scale, tr.PADDED, tr.D ** -0.5),
                  *st,
                  *bs,
                  *dbs)
        return packed, hold, stream

    def _launch(*args, **kwargs):
        packed, _hold, stream = _args(*args, **kwargs)
        abi.run_compiled(compiled, launch_fmha_bwd_dq_gfx942, *packed, fx.Stream(stream))

    _launch.traits = tr
    _launch.block_m = tr.BLOCK_M
    return _launch
