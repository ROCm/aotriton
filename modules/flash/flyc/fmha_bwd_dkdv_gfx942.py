# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dK/dV backward for gfx942, FlyDSL, head dims 32-128.

The forward loop transposed: each wave keeps 16*kv_blocks rows of K and V in
registers and accumulates dK^T and dV^T there, while 16-row Q/dO tiles stream
through LDS. With v_mfma_f32_16x16x16, S = Q.K^T comes out as one kv column
and four consecutive q rows per lane, which is exactly the B operand of the
q-contracted dV^T = dO^T.P and dK^T = Q^T.dS, so P and dS need no repacking.

gfx942 has no LDS transpose read, so LDS holds Q and dO twice: row-major (DMA,
for the d-contracted S and dP) and transposed (built in LDS from the row-major
tile by ds_read + v_perm + ds_write, for the q-contracted dV and dK).

Software pipeline, one wave per SIMD: iteration j runs S/dP of tile j+1 on the
MFMA pipe while the softmax of tile j runs on the VALU, then dV/dK of tile j
while tile j+1's transposed half is built; tile j+2's DMA is in flight.
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
import fmha_tuning_bwd_dkdv_gfx942 as tuning
from philox import Philox

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942.lds">'
LDS_SCOPES = ('r0', 'r1', 't0', 't1')
LOG2E = 1.4426950408889634
WAVE = 64
BM = tuning.BLOCK_ROWS      # q rows per tile, and kv rows per MFMA block


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, num_waves, kv_blocks, padded_head=False):
        assert head_dim in tuning.LADDER, head_dim
        self.D = head_dim
        self.PADDED = padded_head
        # Columns below PAD_FLOOR are real in every call this build serves.
        self.PAD_FLOOR = 0 if padded_head else head_dim
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NUM_WAVES = num_waves
        self.KVB = kv_blocks                    # 16-row kv blocks per wave
        self.R = BM * kv_blocks                 # kv rows per wave
        self.BLOCK_N = num_waves * self.R
        self.BLOCK_SIZE = num_waves * WAVE
        self.DB = head_dim // 16                # 16-wide d blocks
        self.DP = head_dim // 32                # b128 reads per row-major operand row
        # One 16-row tile of one tensor, either orientation.
        self.ROW_BYTES = 2 * head_dim
        self.TILE_BYTES = BM * self.ROW_BYTES
        self.DMA_LINES = self.TILE_BYTES // 256
        self.DMA_ROUNDS = -(-self.DMA_LINES // num_waves)
        # Row-major tiles XOR their 16B chunks so 8 lanes reading 8 rows hit 8
        # bank groups: by row % 8 for 128B-multiple rows, else by (row / 2) % 4.
        self.SWZ_WIDE = self.ROW_BYTES % 128 == 0
        # Transposed unit: 4 q rows x 2 d columns of Q or dO, one dword per row.
        self.T_UNITS = (BM // 4) * (head_dim // 2)
        self.T_UNITS_PER_THREAD = -(-2 * self.T_UNITS // self.BLOCK_SIZE)
        # Two rings of two slots: rows (Q and dO row-major) and transposed (Q^T, dO^T).
        # Tile j's rows are read one iteration before its transposed half.
        self.QR = 0
        self.DOR = self.TILE_BYTES
        self.QT = 0
        self.DOT = self.TILE_BYTES
        self.SLOT_BYTES = 2 * self.TILE_BYTES
        self.LDS_BYTES = 4 * self.SLOT_BYTES
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        self.CACHE_TAG = f'v7_D{head_dim}_c{int(causal)}_{dtype_str}_w{num_waves}_kvb{kv_blocks}_p{int(padded_head)}'

    def row_slot(self, s):
        return s * self.SLOT_BYTES

    def t_slot(self, s):
        return (2 + s) * self.SLOT_BYTES

    def swz(self, row):
        """Physical-chunk XOR of a row-major tile row (Python int or traced)."""
        if self.SWZ_WIDE:
            return row % 8
        return (row // 2) % 4


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
                               padded_head=False):
    """A launcher with fmha_bwd_dkdv_gfx1201_kernel's `_launch` interface and kernel ABI.

    `causal_type` as gfx1201's: 1/2 are host sentinels, 3 is gSWA; all compile to 3.
    """
    host_causal_type = (1 if causal else 0) if causal_type is None else causal_type
    if host_causal_type not in (0, 1, 2, 3):
        raise ValueError(f'causal_type must be 0-3, got {host_causal_type}')
    causal_type = 0 if host_causal_type == 0 else 3
    if bias and causal_type:
        raise ValueError('bias with a causal mask is not served (aot/_common.py flash_disabled)')
    if num_waves is None:
        num_waves = tuning.default_num_waves(head_dim)
    if kv_blocks is None:
        kv_blocks = tuning.default_kv_blocks(head_dim)
    if waves_per_eu is None:
        waves_per_eu = tuning.default_waves_per_eu(head_dim)
    if interleave_valu is None:
        interleave_valu = tuning.default_interleave_valu(head_dim)
    tr = Traits(head_dim, causal_type != 0, dtype_str, num_waves, kv_blocks, padded_head)
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(tr.CAUSAL)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    has_bias = bias
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_il{interleave_valu}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}'
    elem = fx.BFloat16 if dtype_str == 'bf16' else fx.Float16

    def v4e():
        return Vec.make_type(4, elem)

    def v8e():
        return Vec.make_type(8, elem)

    def v4f():
        return Vec.make_type(4, fx.Float32)

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, tr.LDS_BYTES, 16]

    def mfma(a, b, c):
        if const_expr(tr.DTYPE == 'bf16'):
            return rocdl.mfma_f32_16x16x16bf16_1k(v4f(), [Vec(a).bitcast(fx.Int16), Vec(b).bitcast(fx.Int16), c])
        return rocdl.mfma_f32_16x16x16f16(v4f(), [a, b, c])

    def exp2(x):
        return fx.Float32(rocdl.exp2(T.f32, raw(x)))

    def pack4(vals):
        """Four f32 -> one operand. bf16: round to nearest, ties away, NaN-unsafe, as
        fmha_common_gfx1201.bf16_round_nonan_pack_v8; see the numerical lessons, #16."""
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
            scopes = [f'#llvm.alias_scope<id = "{n}", domain = {_LDS_DOMAIN}>' for n in names]
            return ir.Attribute.parse('[' + ', '.join(scopes) + ']')
        return arr([name]), arr([n for n in LDS_SCOPES if n != name])

    def cluster_barrier():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
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
        _ = cache_tag
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
        lds_p = lds.buf.ptr
        lds_base = fx.Index(fx.ptrtoint(lds_p))

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)

        def rsrc_for(ptr, batch, h, row_off, sb, sh, ss, rows, row_elems):
            # One sequence of one head, ending on its last row's last owned column
            # (fmha_dualwave_gfx950._slab_span_elems): past it reads 0 and stores drop.
            elems = fx.Int64(batch) * sb + fx.Int64(h) * sh + fx.Int64(row_off) * ss
            base = fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(2)
            span = (fx.Int64(rows) - fx.Int64(1)) * ss + fx.Int64(row_elems)
            span = (fx.Int32(rows) > fx.Int32(0)).select(span, fx.Int64(0))
            return buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=span * fx.Int64(2))

        def raw_rsrc(r):
            return rocdl.get_buffer_rsrc(r)

        kv0 = kv_tile * tr.BLOCK_N
        kv0_i32 = fx.Int32(kv0)
        kv_wave = kv0 + wave * tr.R             # this wave's first kv row

        def col_mask(c0, hdim):
            """Dword masks keeping columns c0 + [0, 8) below hdim."""
            words = []
            for w in range_constexpr(4):
                lo = (c0 + fx.Int32(2 * w) < fx.Int32(hdim)).select(fx.Int32(0xFFFF), fx.Int32(0))
                hi = (c0 + fx.Int32(2 * w + 1) < fx.Int32(hdim)).select(fx.Int32(-65536), fx.Int32(0))
                words.append(lo | hi)
            return Vec.from_elements(words, fx.Int32)

        def masked8(v8, mask):
            return Vec(Vec(v8).bitcast(fx.Int32) & mask).bitcast(elem).ir_value()

        # Resident K and V. Block b, pair p: row kv_wave + 16b + lc, d = 32p + 8lg + [0, 8);
        # the S k-steps take the low and high halves, and so does the Q/dO read.
        k_rsrc = rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk, row_qk)
        v_rsrc = rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk, row_vo)
        k_row_stride = fx.Index(fx.Int32(stride_k_seq))
        v_row_stride = fx.Index(fx.Int32(stride_v_seq))
        qk_masks = {}
        vo_masks = {}
        for p in range_constexpr(tr.DP):
            if const_expr(32 * p + 32 > tr.PAD_FLOOR):
                c0 = fx.Int32(32 * p) + fx.Int32(lg * 8)
                qk_masks[p] = col_mask(c0, hdim_qk)
                vo_masks[p] = col_mask(c0, hdim_vo)
        k_packs = []
        v_packs = []
        for b in range_constexpr(tr.KVB):
            row = kv_wave + fx.Index(16 * b) + lc
            kb = []
            vb = []
            for p in range_constexpr(tr.DP):
                col = fx.Index(32 * p) + lg * 8
                k8 = buffer_ops.buffer_load(k_rsrc, row * k_row_stride + col, vec_width=8, dtype=elem)
                v8 = buffer_ops.buffer_load(v_rsrc, row * v_row_stride + col, vec_width=8, dtype=elem)
                if const_expr(p in qk_masks):
                    # Slack columns may hold NaN, and 0 * NaN is NaN: zero both sides of S and dP.
                    k8 = masked8(k8, qk_masks[p])
                    v8 = masked8(v8, vo_masks[p])
                kb.append(k8)
                vb.append(v8)
            k_packs.append(kb)
            v_packs.append(vb)

        # LDS pointers: an absolute i32 base plus a constant GEP, which the backend folds
        # into the DS offset field (as fmha_dualwave_gfx950._lds_ptr_with_imm).
        lds_ptr_ty = ir.Type.parse('!llvm.ptr<3>')
        lds_base_i32 = fx.Int32(lds_base)

        def lds_ptr(base_i32, imm):
            p = llvm.inttoptr(lds_ptr_ty, raw(fx.Int32(base_i32)))
            if imm == 0:
                return p
            return llvm.getelementptr(lds_ptr_ty,
                                      p,
                                      [],
                                      [imm],
                                      ir.IntegerType.get_signless(8),
                                      llvm.GEPNoWrapFlags.inbounds)

        def lds_load(base_i32, imm, ty, align, scope):
            al, noal = scope_attrs(scope)
            return llvm.LoadOp(ty,
                               lds_ptr(base_i32, imm),
                               alignment=align,
                               alias_scopes=al,
                               noalias_scopes=noal).result

        def lds_store(val, base_i32, imm, align, scope):
            al, noal = scope_attrs(scope)
            llvm.StoreOp(val, lds_ptr(base_i32, imm), alignment=align, alias_scopes=al, noalias_scopes=noal)

        wave_u = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave)))
        # Row-major DMA: round i moves line L = i*W + wave (mod the tile's lines) of 256B;
        # lane writes LDS byte 256L + 4*lane, fetched from its unswizzled source. When a
        # line holds whole rows, the line's first row rides in soffset and the voffset
        # depends only on L % 8, so a few VGPRs serve every round.
        whole_rows = 256 % tr.ROW_BYTES == 0 or tr.ROW_BYTES % 256 == 0

        def dma_plan(row_stride_b):
            """Per round: (voffset VGPR, uniform first-row multiplier of the line, i.e. rows per line)."""
            voffs = []
            cache = {}
            for i in range_constexpr(tr.DMA_ROUNDS):
                if const_expr(whole_rows and tr.DMA_LINES % tr.NUM_WAVES == 0):
                    rpl = max(1, 256 // tr.ROW_BYTES)
                    key = (i * tr.NUM_WAVES * rpl) % 8
                    if key not in cache:
                        lrow = (lane * 4) // tr.ROW_BYTES
                        urow8 = (fx.Index(key) + wave * rpl) % 8
                        row8 = (urow8 + lrow) % 8
                        pc = ((lane * 4) % tr.ROW_BYTES) // 16
                        ch = pc ^ tr.swz(row8)
                        cache[key] = fx.Int32(lrow) * row_stride_b + fx.Int32(ch * 16 + (lane * 4) % 16)
                    voffs.append(cache[key])
                else:
                    line = (fx.Index(i * tr.NUM_WAVES) + wave) % tr.DMA_LINES
                    pb = line * 256 + lane * 4
                    row = pb // tr.ROW_BYTES
                    pc = (pb % tr.ROW_BYTES) // 16
                    ch = pc ^ tr.swz(row)
                    voffs.append(fx.Int32(row) * row_stride_b + fx.Int32(ch * 16 + pb % 16))
            return voffs

        def dma_row0(i):
            """Uniform first row of round i's line, added to the tile row in soffset."""
            if const_expr(whole_rows and tr.DMA_LINES % tr.NUM_WAVES == 0):
                rpl = max(1, 256 // tr.ROW_BYTES)
                return (fx.Int32(i * tr.NUM_WAVES) + wave_u) * fx.Int32(rpl)
            return fx.Int32(0)

        dma_m0 = []
        for i in range_constexpr(tr.DMA_ROUNDS):
            line = (fx.Int32(i * tr.NUM_WAVES) + wave_u) % fx.Int32(tr.DMA_LINES)
            dma_m0.append(fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, lds_base_i32 + line * fx.Int32(256))))

        q_row_stride_b = fx.Int32(stride_q_seq) * fx.Int32(2)
        do_row_stride_b = fx.Int32(stride_do_seq) * fx.Int32(2)
        q_dma_voff = dma_plan(q_row_stride_b)
        do_dma_voff = dma_plan(do_row_stride_b)

        def dma(rsrc, voffs, row_stride_b, q_start, region, scope):
            al, noal = scope_attrs(scope)
            for i in range_constexpr(tr.DMA_ROUNDS):
                soff = (fx.Int32(q_start) + dma_row0(i)) * row_stride_b
                m0 = fx.Int64(dma_m0[i] + fx.Int32(region))
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

        # Transposed tiles, built in LDS from the row-major ones (gfx942 has no transpose
        # read). Unit u of Q then dO is q rows 4*g4 + [0, 4) at d columns 2*cp + [0, 2);
        # threads past the last unit redo an earlier one. Q^T row d holds 16 q in four
        # 8B groups, group g at g ^ ((d / 4) % 4).
        t_units = []
        for u in range_constexpr(tr.T_UNITS_PER_THREAD):
            ui = (tid + fx.Index(u * tr.BLOCK_SIZE)) % (2 * tr.T_UNITS)
            region = (ui // tr.T_UNITS) * tr.TILE_BYTES
            unit = ui % tr.T_UNITS
            g4 = unit // (tr.D // 2)
            cp = unit % (tr.D // 2)
            src = []
            for r in range_constexpr(4):
                row = g4 * 4 + fx.Index(r)
                src.append(lds_base_i32 + fx.Int32(region + row * tr.ROW_BYTES + ((cp // 4) ^ tr.swz(row)) * 16
                                                   + (cp % 4) * 4))
            dst = []
            for e in range_constexpr(2):
                d = cp * 2 + fx.Index(e)
                dst.append(lds_base_i32 + fx.Int32(region + d * 32 + (g4 ^ ((d // 4) % 4)) * 8))
            t_units.append((src, dst))

        def t_build(rows_slot, t_slot):
            """Q^T and dO^T of the tile in rows slot `rows_slot`, into transposed slot `t_slot`."""
            sel_lo = fx.Int32(0x05040100)
            sel_hi = fx.Int32(0x07060302)
            for src, dst in t_units:
                w = [fx.Int32(lds_load(a, tr.row_slot(rows_slot), T.i32, 4, f'r{rows_slot}')) for a in src]
                d0 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_lo)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_lo))]
                d1 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_hi)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_hi))]
                for e, dw in ((0, d0), (1, d1)):
                    lds_store(raw(Vec.from_elements(dw, fx.Int32)), dst[e], tr.t_slot(t_slot), 8, f't{t_slot}')

        # Row-major read: lane row lc, logical chunk 4p + lg.
        if const_expr(tr.SWZ_WIDE):
            s2 = (lc % 8) // 4
            s01 = lc % 4
            rm_even = lds_base_i32 + fx.Int32(lc * tr.ROW_BYTES + (lg ^ s01) * 16 + s2 * 64)
            rm_odd = lds_base_i32 + fx.Int32(lc * tr.ROW_BYTES + (lg ^ s01) * 16 + (fx.Index(1) - s2) * 64)

            def rm_at(p):
                return [rm_even, rm_odd][p % 2], 128 * (p // 2)
        else:
            rm_base = lds_base_i32 + fx.Int32(lc * tr.ROW_BYTES + (lg ^ ((lc // 2) % 4)) * 16)

            def rm_at(p):
                return rm_base, 64 * p
        # Transposed read: row 16 * db + lc, group lg.
        t_rd = lds_base_i32 + fx.Int32(lc * 32 + (lg ^ ((lc // 4) % 4)) * 8)

        def rm_read(region, scope, masks):
            out = []
            for p in range_constexpr(tr.DP):
                base, imm = rm_at(p)
                v = lds_load(base, imm + region, v8e(), 16, scope)
                if const_expr(p in masks):
                    v = masked8(v, masks[p])
                out.append(v)
            return out

        def t_read(region, scope):
            return [lds_load(t_rd, region + db * 512, v4e(), 8, scope) for db in range(tr.DB)]

        def gemm_d(rows, packs):
            """[q][kv] over d for each kv block: A = rows (Q or dO), B = resident K or V."""
            out = []
            for b in range_constexpr(tr.KVB):
                acc = fx.full(4, 0.0, fx.Float32)
                for p in range_constexpr(tr.DP):
                    a = Vec(rows[p])
                    k8 = Vec(packs[b][p])
                    acc = mfma(a.shuffle(a, [0, 1, 2, 3]).ir_value(), k8.shuffle(k8, [0, 1, 2, 3]).ir_value(), acc)
                    acc = mfma(a.shuffle(a, [4, 5, 6, 7]).ir_value(), k8.shuffle(k8, [4, 5, 6, 7]).ir_value(), acc)
                out.append(acc)
            return out

        # Per q head: Q, dO, LSE, Delta, bias and the dropout plane.
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        def head_state(head_q):
            q_rsrc = raw_rsrc(rsrc_for(Q,
                                       q_batch,
                                       head_q,
                                       q_row_off,
                                       stride_q_batch,
                                       stride_q_head,
                                       stride_q_seq,
                                       sq,
                                       row_qk))
            do_rsrc = raw_rsrc(rsrc_for(DO,
                                        q_batch,
                                        head_q,
                                        q_row_off,
                                        stride_do_batch,
                                        stride_do_head,
                                        stride_do_seq,
                                        sq,
                                        row_vo))
            row0, pitch = fmha.lse_row_addressing(varlen_bits,
                                                  fx.Index(q_batch),
                                                  head_q,
                                                  num_head_q,
                                                  lse_tokens,
                                                  fx.Index(q_row_off))
            nbytes = fx.Int64(sq * pitch) * fx.Int64(4)
            row0_b = fx.Int64(row0) * fx.Int64(4)
            lse_rsrc = raw_rsrc(buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(LSE)) + row0_b,
                                                                            num_records_bytes=nbytes))
            delta_rsrc = raw_rsrc(buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(Delta)) + row0_b,
                                                                              num_records_bytes=nbytes))
            return q_rsrc, do_rsrc, lse_rsrc, delta_rsrc, fx.Int32(pitch) * fx.Int32(4)

        # This lane's LSE/Delta rows: 4 * lg + [0, 4) of the tile; r rides in soffset.
        def row_loads(rsrc, pitch_b, q_start):
            voff = fx.Int32(lg * 4) * pitch_b
            return [fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type,
                                                       rsrc,
                                                       voff,
                                                       (fx.Int32(q_start) + fx.Int32(r)) * pitch_b,
                                                       0)) for r in range(4)]

        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        c_neg_inf = fx.Float32(float('-inf'))

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
        kv_end = smin(kv0_i32 + fx.Int32(tr.BLOCK_N), sk_i32)
        q_lo = smax(kv0_i32 - win_r, fx.Int32(0))
        q_hi = smin(kv_end - fx.Int32(1) + win_l + fx.Int32(1), sq_i32)
        alive = (kv0_i32 < sk_i32) & (q_hi > q_lo)
        t0 = fx.Index(alive.select(q_lo, fx.Int32(0))) // BM
        q_base = t0 * BM
        n_tiles = alive.select((fx.Index(q_hi) + BM - 1) // BM - t0, fx.Index(0))

        def n_rel(x):
            return fx.Index(smax(x, fx.Int32(0)))

        # Tile r (from t0) is clean, every (q, kv) inside the band, iff lead <= r < clean_end.
        lead = smin((n_rel(kv_end - fx.Int32(1) - win_r - fx.Int32(q_base)) + BM - 1) // BM, n_tiles)
        ce_x = kv0_i32 + win_l - fx.Int32(BM - 1) - fx.Int32(q_base)
        clean_end = (ce_x < fx.Int32(0)).select(fx.Index(0), fx.Index(ce_x) // BM + fx.Index(1))
        clean_end = smin(smax(clean_end, lead), n_tiles)
        # Masked tiles [0, j_clean) and [j_tail, n) bracket the clean pairs; j_clean
        # is even so the unrolled loop knows its stages.
        j_clean = smin(lead + lead % 2, n_tiles)
        j_clean_end = smax(clean_end, j_clean)
        n_pairs = (j_clean_end - j_clean) // 2
        j_tail = j_clean + n_pairs * 2

        # Mask: (kv - q + win_l) as unsigned <= win_l + win_r; an empty band fails all.
        _span = win_l + win_r
        _empty = _span < fx.Int32(0)
        band_span = _empty.select(fx.Int32(0), _span)
        band_lane = _empty.select(fx.Int32(-(1 << 30)),
                                  fx.Int32(kv_wave + lc) - fx.Int32(lg * 4) + win_l)

        def mask_scores(s_vals, q_start, b):
            rel0 = band_lane - fx.Int32(q_start) + fx.Int32(16 * b)
            out = []
            for r in range_constexpr(4):
                rel = rel0 - fx.Int32(r)
                out.append((fx.Uint32(rel) > fx.Uint32(band_span)).select(c_neg_inf, s_vals[r]))
            return out

        if const_expr(bias):
            # (B, H, Sq, Sk), Sk contiguous; rows offset by Q's varlen offset, columns by K's.
            b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(k_row_off + sk_i32), fx.Int64(0))
            b_lane = (fx.Int32(lg * 4) * fx.Int32(stride_b_seq_q) + k_row_off + fx.Int32(kv_wave + lc)) * fx.Int32(2)

        def bias_rsrc_for(head_q):
            b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head_q) * stride_b_head
                       + fx.Int64(q_row_off) * stride_b_seq_q)
            b_addr = fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2)
            return raw_rsrc(buffer_ops.create_buffer_resource_from_addr(b_addr, num_records_bytes=b_span * fx.Int64(2)))

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
            rng = Philox.for_arch('gfx942')
            ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            ph_q = fx.Int32(lg * 4 + lc % 4)
            ph_col = fx.Int32(kv_wave) + fx.Int32((lc // 4) * 4)
            keep_bit = fx.Int32(1) << fx.Int32(lc % 4)

        def dropout_nibs(hs, q_start):
            """Per kv block, this lane's 4 keep bits of tile q_start."""
            if const_expr(not dropout):
                return []
            ph_tile = hs['ph_lane'] + fx.Int64(q_start) * hs['ph_stride']
            nibs = []
            for b in range_constexpr(tr.KVB):
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

        def tile_math(s_accs, dp_accs, lse_l2, delta, q_start, masked, hs, nibs):
            """P and dS packs per kv block, from S, dP and this tile's rows."""
            c_scale_v = fx.full(4, fx.Float32(c_scale), fx.Float32)
            lse_v = Vec.from_elements(lse_l2, fx.Float32)
            delta_v = Vec.from_elements(delta, fx.Float32)
            p_packs = []
            ds_packs = []
            for b in range_constexpr(tr.KVB):
                # Scale once in fp32, as the forward does; the mask lands on the scaled score.
                s = Vec(s_accs[b]) * c_scale_v
                s_vals = [s[r] for r in range(4)]
                if const_expr(bias):
                    bv = bias_load(hs['b_rsrc'], q_start, b)
                    s_vals = [s_vals[r] + bv[r] * fx.Float32(LOG2E) for r in range(4)]
                if const_expr(masked and tr.CAUSAL):
                    s_vals = mask_scores(s_vals, q_start, b)
                e = Vec.from_elements(s_vals, fx.Float32) - lse_v
                p = [exp2(e[r]) for r in range(4)]
                dp = Vec(dp_accs[b])
                if const_expr(dropout):
                    keep = dropout_keep(nibs[b])
                    dsc = fx.Float32(dropout_scale)
                    # 1/(1-p) of P joins dV at the store.
                    p_drop = [keep[r].select(p[r], fx.Float32(0.0)) for r in range(4)]
                    dp_vals = [keep[r].select(dp[r] * dsc, fx.Float32(0.0)) for r in range(4)]
                    dpd = Vec.from_elements(dp_vals, fx.Float32) - delta_v
                else:
                    p_drop = p
                    dpd = dp - delta_v
                ds = Vec.from_elements(p, fx.Float32) * dpd
                p_packs.append(pack4(p_drop))
                ds_packs.append(pack4([ds[r] for r in range(4)]))
            return p_packs, ds_packs

        n_acc = 2 * tr.DB * tr.KVB
        n_sd = 2 * tr.KVB
        n_mfma_a = 2 * tr.KVB * 2 * tr.DP
        n_mfma_b = 2 * tr.DB * tr.KVB
        N_ROWS = 8
        # Two waves per SIMD: the next tile's Philox runs under this tile's dV/dK
        # (B2 H32 S4k d96: 2.30 vs 4.83 ms; at one wave it does not pay).
        early = dropout and waves_per_eu >= 2
        # One wave with dropout from d64: VALU-bound, the order is left to the
        # scheduler (B2 H32 S8k d128: 12.88 vs 13.34 ms; d32 causal loses 7%).
        sched_valu = 0 if dropout and not early and head_dim >= 64 else interleave_valu

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
            base = tr.row_slot(slot)
            dma(hs['q_rsrc'], q_dma_voff, q_row_stride_b, q_start, base + tr.QR, f'r{slot}')
            dma(hs['do_rsrc'], do_dma_voff, do_row_stride_b, q_start, base + tr.DOR, f'r{slot}')

        def gemm_sdp(slot):
            """S and dP accumulators of the tile in rows slot `slot`."""
            base = tr.row_slot(slot)
            qr = rm_read(base + tr.QR, f'r{slot}', qk_masks)
            dor = rm_read(base + tr.DOR, f'r{slot}', vo_masks)
            return gemm_d(qr, k_packs) + gemm_d(dor, v_packs)

        def lse_delta(rows):
            lse_l2 = [fx.Int32(x).bitcast(fx.Float32) * fx.Float32(LOG2E) for x in rows[:4]]
            delta = [fx.Int32(x).bitcast(fx.Float32) for x in rows[4:8]]
            return lse_l2, delta

        def iteration(j, carried, par, masked, hs):
            """P/dS and dV/dK of tile j, S/dP of tile j + 1, loads of tiles j + 1 and j + 2.

            Rows slot r(t % 2) holds tile t's Q/dO, transposed slot t(t % 2) its Q^T/dO^T;
            `par` = j % 2, known at trace time."""
            accs = list(carried[:n_acc])
            sd = list(carried[n_acc:n_acc + n_sd])
            rows = list(carried[n_acc + n_sd:n_acc + n_sd + N_ROWS])
            q_j = q_base + j * BM
            if const_expr(early):
                nibs = list(carried[n_acc + n_sd + N_ROWS:])
            else:
                nibs = dropout_nibs(hs, q_j)
            # Phase A: S/dP(j + 1) on the MFMA pipe while P/dS(j) runs on the VALU.
            rows_next = rows_load(hs, q_j + BM)
            rows_dma(hs, q_j + 2 * BM, par)
            sd_next = gemm_sdp(1 - par)
            # Phase B's operands, read now so their latency hides under phase A.
            tb = tr.t_slot(par)
            dot = t_read(tb + tr.DOT, f't{par}')
            qt = t_read(tb + tr.QT, f't{par}')
            lse_l2, delta = lse_delta(rows)
            p_packs, ds_packs = tile_math(sd[:tr.KVB], sd[tr.KVB:], lse_l2, delta, q_j, masked, hs, nibs)
            interleave(n_mfma_a, sched_valu, 0x002 | 0x400)
            rocdl.sched_barrier(0)
            # Phase B: dV/dK(j) while tile j + 1's transposed half is built in LDS.
            dk = accs[:tr.DB * tr.KVB]
            dv = accs[tr.DB * tr.KVB:]
            for db in range_constexpr(tr.DB):
                for b in range_constexpr(tr.KVB):
                    dv[db * tr.KVB + b] = mfma(dot[db], p_packs[b], dv[db * tr.KVB + b])
            for db in range_constexpr(tr.DB):
                for b in range_constexpr(tr.KVB):
                    dk[db * tr.KVB + b] = mfma(qt[db], ds_packs[b], dk[db * tr.KVB + b])
            t_build(1 - par, 1 - par)
            nibs_next = []
            if const_expr(early):
                nibs_next = dropout_nibs(hs, q_j + BM)
            interleave(n_mfma_b, 1, 0x002 | 0x200)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(0)
            cluster_barrier()
            return dk + dv + sd_next + rows_next + nibs_next

        init = [fx.full(4, 0.0, fx.Float32) for _ in range(n_acc)]
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

            @flyc.jit
            def masked_iteration(j, carried):
                out = list(carried)
                if j % 2 == fx.Index(0):
                    out = iteration(j, carried, 0, True, hs)
                else:
                    out = iteration(j, carried, 1, True, hs)
                return out

            # Prologue: rows of tiles 0 and 1, transposed tile 0, S/dP(0).
            rows0 = rows_load(hs, q_base)
            rows_dma(hs, q_base, 0)
            rows_dma(hs, q_base + BM, 1)
            rocdl.s_waitcnt(0)
            cluster_barrier()
            t_build(0, 0)
            rocdl.s_waitcnt(0)
            cluster_barrier()
            sd0 = gemm_sdp(0)
            # Iteration 0 refills rows slot 0: every wave must be done reading it.
            rocdl.s_waitcnt(0)
            cluster_barrier()
            init_t = list(accs_h) + sd0 + rows0
            if const_expr(early):
                init_t = init_t + dropout_nibs(hs, q_base)
            r0 = init_t
            for j, carried in range(0, j_clean, 1, init=init_t):
                r0 = yield masked_iteration(j, carried)
            r1 = r0
            for jp, carried in range(0, n_pairs, 1, init=r0):
                mid = iteration(j_clean + jp * 2, carried, 0, False, hs)
                r1 = yield iteration(j_clean + jp * 2 + 1, mid, 1, False, hs)
            r2 = r1
            for j, carried in range(j_tail, n_tiles, 1, init=r1):
                r2 = yield masked_iteration(j, carried)
            res_h = yield list(r2[:n_acc])

        # dK^T / dV^T: lane holds kv row kv_wave + 16b + lc, d = 16db + 4lg + [0, 4).
        dk_rsrc = rsrc_for(DK, k_batch, head_k, k_row_off, stride_dk_batch, stride_dk_head, stride_dk_seq, sk, row_qk)
        dv_rsrc = rsrc_for(DV, k_batch, head_k, k_row_off, stride_dv_batch, stride_dv_head, stride_dv_seq, sk, row_vo)
        dk_row_stride = fx.Index(fx.Int32(stride_dk_seq))
        dv_row_stride = fx.Index(fx.Int32(stride_dv_seq))
        scale_v = fx.full(4, fx.Float32(sm_scale), fx.Float32)
        for b in range_constexpr(tr.KVB):
            row = kv_wave + fx.Index(16 * b) + lc
            row_ok = row < sk
            for db in range_constexpr(tr.DB):
                d = fx.Index(16 * db) + lg * 4
                dk_v = Vec(res_h[db * tr.KVB + b]) * scale_v
                dv_v = Vec(res_h[tr.DB * tr.KVB + db * tr.KVB + b])
                if const_expr(dropout):
                    dv_v = dv_v * fx.full(4, fx.Float32(dropout_scale), fx.Float32)
                ok_k = row_ok
                ok_v = row_ok
                if const_expr(16 * db + 16 > tr.PAD_FLOOR):
                    ok_k = ok_k & (fx.Int32(d) < row_qk)
                    ok_v = ok_v & (fx.Int32(d) < row_vo)
                dk_e = Vec.from_elements([fx.Float32(dk_v[r]).to(elem) for r in range(4)], elem)
                dv_e = Vec.from_elements([fx.Float32(dv_v[r]).to(elem) for r in range(4)], elem)
                buffer_ops.buffer_store(dk_e.ir_value(), dk_rsrc, row * dk_row_stride + d, mask=ok_k)
                buffer_ops.buffer_store(dv_e.ir_value(), dv_rsrc, row * dv_row_stride + d, mask=ok_v)

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
        n_kv = (fx.Index(max_seqlen_k) + tr.BLOCK_N - 1) // tr.BLOCK_N
        value_attrs = {'rocdl.waves_per_eu': waves_per_eu,
                       'rocdl.flat_work_group_size': f'{tr.BLOCK_SIZE},{tr.BLOCK_SIZE}'}
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
            block=(tr.BLOCK_SIZE, 1, 1),
            stream=stream)

    compiled = abi.new_compiled_cache()

    def _launch(Q,
                K,
                V,
                DO,
                DK,
                DV,
                L,
                Delta,
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
                bias=None):
        """fmha_bwd_dkdv_gfx1201_kernel's `_launch`, argument for argument."""
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = abi.prep_tensors([("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV)],
                                          q_heads=("DO",),
                                          k_heads=("DK", "DV"))
        if tr.PADDED:
            if not (0 < max(meta[2], meta[3]) <= tr.D):
                raise ValueError(f'head dims {meta[2:]} do not fit a padded {tr.D} build')
        elif meta[2] != tr.D or meta[3] != tr.D:
            raise ValueError(f'head dims {meta[2:]} on an unpadded {tr.D} build; pass padded_head=True')
        lp = abi.row_tensor_arg(L, "logsumexp", meta[0], seqlen_q, varlen)
        dp = abi.row_tensor_arg(Delta, "delta", meta[0], seqlen_q, varlen)
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
        bp, _, bs, _ = abi.bias_args(has_bias, False, bias, None, Q)
        abi.run_compiled(compiled,
                         launch_fmha_bwd_dkdv_gfx942,
                         *ptrs[:3],
                         bp,
                         *ptrs[3:],
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
                         fx.Stream(stream))
        del hold

    _launch.traits = tr
    return _launch
