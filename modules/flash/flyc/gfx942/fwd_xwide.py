# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention forward for gfx942 at head dims 384 and 512, FlyDSL.

fwd.py's algorithm (S^T = K.Q^T, O^T += V^T.P^T, V transposed
through VGPRs) cannot hold hd512: at 32 query rows a wave, O^T alone is 256
fp32 per lane and Q^T another 128. This body runs 16 rows a wave on
v_mfma_f32_16x16x16, whose S^T output layout is still P^T's operand layout:
Q^T (D/8 regs) + O^T (D/4 regs) = 192 at hd512, so a wave fits 256 registers
and two share a SIMD. KV tiles are 16 keys: double-buffered K and V^T then take
64 KB at hd512, exactly the LDS. Neither tile fits in registers, so the MFMAs
stream their A operands from LDS, one 16B read per two MFMAs.

Why not the alternatives: 1 wave/SIMD with O in AGPRs serializes softmax and
MFMA in one wave, and rescaling AGPRs costs three ops an element; splitting D
across waves (gfx950's VO_SHARDS) duplicates QK; QK shards need an S exchange
through an LDS that is already full. The price of 16 rows is LDS traffic: each
wave reads all of K and V^T, 256 KB a tile per CU, but that streams in the
shadow of the MFMAs (removing every read measured +0%).

LDS layouts, chosen so every read is conflict-free (SQ_LDS_BANK_CONFLICT = 0):
- K: row-major, 16B chunk c of row r at chunk c ^ (r % 8).
- V^T: chunk (p, g, r) at p * 1024 + g * 256 + r * 16 holds rows 32p + 2r and
  32p + 2r + 1, keys 4g..4g+3; a read of p is one contiguous 1 KB. MFMA 2p + e
  computes O^T rows d = 32p + 2r + e.

hd512 fits 256 VGPRs only because nothing else lives across the loop: V is
loaded and stored within a tile, the masks rebuild their lane values, and a
lone spill would cost a vmcnt(0) that drains the K DMA every tile.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith, llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

import fmha_utils as fmha
from .utils import (
    LN2,
    LOG2E,
    OOB_VOFF,
    WAVE,
    Lds,
    alias_scopes,
    band_window,
    bias_load4,
    bias_slab,
    clean_tiles,
    cluster_barrier,
    col_mask_words,
    exp2,
    kv_tiles,
    lds_ptr_at,
    m0_at,
    mask_band,
    mfma_16x16x16,
    pack_operand,
    philox_plane,
    philox_stream,
    pipelined,
    row_band,
    rsrc_for,
    set_alias_scopes,
    smax,
    smin,
)
from . import tuning_fwd as tuning

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_fwd_gfx942_xwide.lds">'
LDS_SCOPES = ('k0', 'k1', 'vt0', 'vt1')
WAVE_M = 16             # query rows per wave: the N of the 16x16x16 MFMA
BLOCK_N = 16            # keys per KV tile: the M of the QK MFMA
# LDS operand reads issued ahead of their MFMAs.
PREFETCH = 2


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, num_waves, padded_head=False):
        assert head_dim % 128 == 0 and 256 <= head_dim <= 512, head_dim
        assert num_waves == 8, num_waves
        self.D = head_dim
        self.PADDED = padded_head
        self.PAD_FLOOR = 0 if padded_head else head_dim
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NUM_WAVES = num_waves
        self.BLOCK_SIZE = num_waves * WAVE
        self.BLOCK_M = num_waves * WAVE_M
        self.BLOCK_N = BLOCK_N
        self.NT = head_dim // 32                # 16B K reads per tile, two MFMAs each
        self.NP = head_dim // 32                # 16B V^T reads per tile, two MFMAs each
        self.K_ROW_BYTES = head_dim * 2
        self.K_TILE_BYTES = BLOCK_N * self.K_ROW_BYTES
        # K DMA: wave w fills rows w and w + 8, 256B a round, so its XOR key is
        # always w and one voffset serves every round.
        self.K_QUARTERS = self.K_ROW_BYTES // 256
        self.K_DMA_ROUNDS = 2 * self.K_QUARTERS
        self.VT_TILE_BYTES = BLOCK_N * head_dim * 2
        # V global load: one unit = 4 keys x 2 columns. Threads past the last unit
        # redo an earlier one: the same bytes to the same place.
        self.V_UNITS = (BLOCK_N // 4) * (head_dim // 2)
        self.V_UNITS_PER_THREAD = -(-self.V_UNITS // self.BLOCK_SIZE)
        self.LDS_K_BASE = 0
        self.LDS_VT_BASE = 2 * self.K_TILE_BYTES
        self.LDS_BYTES = self.LDS_VT_BASE + 2 * self.VT_TILE_BYTES
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        self.CACHE_TAG = f'X{head_dim}_c{int(causal)}_{dtype_str}_w{num_waves}_p{int(padded_head)}'


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
    """build_fmha_fwd_gfx942's interface for head dims 384 and 512 (padded: any
    head dims up to `head_dim`).

    `block_n` must be None or 16; `num_waves` must be 8. The rest as
    build_fmha_fwd_gfx942. A causal build runs q tiles longest first.
    """
    host_causal_type, causal_type = fmha.builder_causal_types(causal, causal_type, bias)
    if block_n not in (None, BLOCK_N):
        raise ValueError(f'the xwide forward runs {BLOCK_N}-key tiles, got block_n={block_n}')
    tr = Traits(head_dim, causal_type != 0, dtype_str, num_waves, padded_head)
    if grid_axis_order is None:
        # As the narrow body: tile-fastest non-causal (B4 H32 S8k d512: 510 vs 499
        # TFLOPS), head-fastest causal, which with longest-first tiles is globally LPT
        # (414 vs 387).
        grid_axis_order = tuning.default_grid_axis_order(tr.CAUSAL)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_thr{rescale_threshold}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}'
    elem = fx.BFloat16 if dtype_str == 'bf16' else fx.Float16

    def v4e():
        return Vec.make_type(4, elem)

    def v8e():
        return Vec.make_type(8, elem)

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, tr.LDS_BYTES, 16]

    def mfma(a, b, c):
        return mfma_16x16x16(a, b, c, elem)

    def pack_p(vals):
        return pack_operand(vals, elem)

    def scope_attrs(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
    # The kernel ABI is flash_attn_aiw_kernel's (gfx1201), as fmha_fwd_gfx942_kernel's.
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
        _ = cache_tag
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lr = lane % 16              # query row in the wave; K row / V^T row in reads
        lg = lane // 16             # 4-element group of the MFMA's k axis
        if const_expr(tile_fastest):
            q_tile = fx.Index(gpu.block_idx.x)
            head = fx.Index(gpu.block_idx.y)
        else:
            head = fx.Index(gpu.block_idx.x)
            q_tile = fx.Index(gpu.block_idx.y)
        if const_expr(tr.CAUSAL):
            # Longest tiles first: under any causal window later rows see more keys.
            # Count tiles as the C++ grid does; gpu.grid_dim would add hidden kernargs.
            n_q_tiles = (fx.Index(max_seqlen_q) + tr.BLOCK_M - 1) // tr.BLOCK_M
            q_tile = n_q_tiles - fx.Index(1) - q_tile
        z_i32 = fx.Int32(gpu.block_idx.z)
        head_k = head // (fx.Index(num_head_q) // fx.Index(num_head_k))
        sq_i32, q_row_off, q_batch = fmha.decode_addressing(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32)
        sk_i32, k_row_off, k_batch = fmha.decode_addressing(varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32)
        sq = fx.Index(sq_i32)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_base = fx.Index(fx.ptrtoint(lds_p))

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
        k_row_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_row_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)
        o_row_stride = fx.Index(fx.Int32(stride_o_seq))

        q_start = q_tile * tr.BLOCK_M
        q_row = q_start + wave * WAVE_M + lr

        def col_mask(c0):
            return Vec.from_elements(col_mask_words(c0, hdim_qk), fx.Int32)

        # Q^T packs: k-step 2t+e reads d = 32t + 8*lg + 4e + [0, 4), as K below.
        q_packs = []
        for t in range_constexpr(tr.NT):
            c0 = fx.Index(32 * t) + lg * 8
            v8 = buffer_ops.buffer_load(q_rsrc, q_row * q_row_stride + c0, vec_width=8, dtype=elem)
            if const_expr(32 * t + 32 > tr.PAD_FLOOR):
                # Slack columns may hold NaN, and 0 * NaN is NaN: zero Q here, K in LDS.
                v8 = Vec(Vec(v8).bitcast(fx.Int32) & col_mask(fx.Int32(c0))).bitcast(elem).ir_value()
            q_packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
            q_packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())

        # K DMA: wave w fills rows w and w + 8; 16B chunk c of row r sits at c ^ (r % 8).
        # Round (k, q) moves row w + 8k, bytes 256q..256q+255; both ride in soffset,
        # which the range check includes.
        wave_s = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave)))
        k_lane_chunk = (fx.Int32(lane // 4) ^ wave_s)
        k_voff = k_lane_chunk * fx.Int32(16) + fx.Int32(lane % 4) * fx.Int32(4)
        k_lane_col = k_lane_chunk * fx.Int32(8) + fx.Int32(lane % 4) * fx.Int32(2)
        k_m0 = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_base + fx.Index(tr.LDS_K_BASE))))
        k_m0 = k_m0 + wave_s * fx.Int32(tr.K_ROW_BYTES)

        def k_m0_at(const):
            """k_m0 + const in the loop: hoisted, the 16 sums would pin 16 SGPRs."""
            return m0_at(k_m0, const)

        def k_dma(kv_start, buf):
            al, noal = scope_attrs(f'k{buf}')
            for k in range_constexpr(2):
                soff = (fx.Int32(kv_start) + wave_s + fx.Int32(8 * k)) * k_row_stride_b
                for q in range_constexpr(tr.K_QUARTERS):
                    voff = k_voff
                    if const_expr(tr.PADDED):
                        # Dwords wholly past hdim_qk read 0; the one it splits is fixed in LDS.
                        voff = (k_lane_col < fx.Int32(hdim_qk) - fx.Int32(128 * q)).select(voff, fx.Int32(OOB_VOFF))
                    m0 = k_m0_at(buf * tr.K_TILE_BYTES + 8 * k * tr.K_ROW_BYTES + 256 * q)
                    op = rocdl.raw_ptr_buffer_load_lds(k_rsrc,
                                                       lds_ptr_at(fx.Int64(m0)),
                                                       fx.Int32(4),
                                                       voff,
                                                       soff + fx.Int32(256 * q),
                                                       fx.Int32(0),
                                                       fx.Int32(0))
                    set_alias_scopes(op, al, noal)

        if const_expr(tr.PADDED):
            # An odd hdim_qk leaves its slack column in a dword with a real one: wave
            # lr % 8, which DMA'd row lr, zeroes that column after its vmcnt.
            h = fx.Int32(hdim_qk)
            fix_pos = (fx.Int32(lr) * fx.Int32(tr.K_ROW_BYTES)
                       + (((h >> fx.Int32(3)) ^ fx.Int32(lr % 8)) << fx.Int32(4))
                       + (h & fx.Int32(7)) * fx.Int32(2))
            fix_ok = (((h & fx.Int32(1)) == fx.Int32(1)) & (lane < fx.Index(16)) & (lr % 8 == wave))
            fix_off = fx.Index(fix_pos) + fx.Index(tr.LDS_K_BASE)

        def make_k_fix(buf):
            al, noal = scope_attrs(f'k{buf}')

            def fix_ptr():
                # A function, not lds_io itself: the jit body below sees only these.
                return lds_io.view(fix_off + fx.Index(buf * tr.K_TILE_BYTES), fx.Int16, 2).llvm_ptr

            @flyc.jit
            def k_fix_buf():
                if fix_ok:
                    llvm.StoreOp(raw(fx.Int16(0)),
                                 fix_ptr(),
                                 alignment=2,
                                 alias_scopes=al,
                                 noalias_scopes=noal)
            return k_fix_buf

        def k_fix(buf):
            if const_expr(tr.PADDED):
                make_k_fix(buf)()

        # V: each unit is 4 keys x 2 columns, transposed by v_perm into one V^T chunk.
        # When a thread's units are whole key groups apart, one base serves them all.
        v_affine = tr.V_UNITS % tr.BLOCK_SIZE == 0 and tr.BLOCK_SIZE % (tr.D // 2) == 0
        v_base = []
        vt_wr_off = []
        for u in range_constexpr(1 if v_affine else tr.V_UNITS_PER_THREAD):
            unit = (tid + fx.Index(u * tr.BLOCK_SIZE)) % tr.V_UNITS
            g = unit // (tr.D // 2)
            cp = unit % (tr.D // 2)
            v_base.append(fx.Int32(g * 4) * v_row_stride_b + fx.Int32(cp * 4))
            vt_wr_off.append(fx.Index(tr.LDS_VT_BASE) + (cp // 16) * 1024 + g * 256 + (cp % 16) * 16)
        g_step = tr.BLOCK_SIZE // (tr.D // 2)   # key groups between a thread's units, if affine

        def v_load(kv_start):
            out = []
            for u in range_constexpr(tr.V_UNITS_PER_THREAD):
                for kk in range_constexpr(4):
                    if const_expr(v_affine):
                        soff = (fx.Int32(kv_start) + fx.Int32(4 * g_step * u + kk)) * v_row_stride_b
                        base = v_base[0]
                    else:
                        soff = (fx.Int32(kv_start) + fx.Int32(kk)) * v_row_stride_b
                        base = v_base[u]
                    out.append(fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, v_rsrc, base, soff, 0)))
            return out

        def vt_wr_at(u):
            if const_expr(v_affine):
                return vt_wr_off[0] + g_step * 256 * u
            return vt_wr_off[u]

        def v_store_lds(regs, buf):
            sel_lo = fx.Int32(0x05040100)
            sel_hi = fx.Int32(0x07060302)
            al, noal = scope_attrs(f'vt{buf}')
            for u in range_constexpr(tr.V_UNITS_PER_THREAD):
                w = regs[4 * u:4 * u + 4]
                dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_lo)),
                      fx.Int32(rocdl.perm_b32(w[3], w[2], sel_lo)),
                      fx.Int32(rocdl.perm_b32(w[1], w[0], sel_hi)),
                      fx.Int32(rocdl.perm_b32(w[3], w[2], sel_hi))]
                llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                             lds_io.view(vt_wr_at(u) + buf * tr.VT_TILE_BYTES, fx.Int32, 16).llvm_ptr,
                             alignment=16,
                             alias_scopes=al,
                             noalias_scopes=noal)

        def opaque(x):
            """x through a v_mov LLVM cannot see into, so constant offsets fold into the
            ds_read immediate instead of a v_or per read."""
            return fx.Index(fx.Int32(llvm.inline_asm(T.i32, [raw(fx.Int32(x))], 'v_mov_b32 $0, $1', '=v,v')))

        # K read t: row lr, chunk 4t + lg, at chunk (4t + lg) ^ (lr % 8).
        k_rd = [opaque(fx.Index(tr.LDS_K_BASE) + lr * tr.K_ROW_BYTES + ((fx.Index(4 * e) + lg) ^ (lr % 8)) * 16)
                for e in range(2)]
        vt_rd = opaque(fx.Index(tr.LDS_VT_BASE) + lane * 16)

        def stream(n, read, use):
            """LLVM alone keeps one buffer and waits on every read."""
            pipelined(n, PREFETCH, lambda t: Vec(read(t)), use)

        def qk(buf):
            """S^T of one tile: v4f32, keys 4*lg + i of query row lr."""
            acc = [fx.full(4, 0.0, fx.Float32)]

            def read(t):
                return lds_io.load(k_rd[t % 2] + (buf * tr.K_TILE_BYTES + 128 * (t // 2)), v8e(), 16, f'k{buf}')

            def use(t, k8):
                acc[0] = mfma(k8.shuffle(k8, [0, 1, 2, 3]).ir_value(), q_packs[2 * t], acc[0])
                acc[0] = mfma(k8.shuffle(k8, [4, 5, 6, 7]).ir_value(), q_packs[2 * t + 1], acc[0])

            stream(tr.NT, read, use)
            return acc[0]

        def pv(o_accs, p_pack, buf):
            o_accs = list(o_accs)

            def read(p):
                return lds_io.load(vt_rd + (buf * tr.VT_TILE_BYTES + 1024 * p), v8e(), 16, f'vt{buf}')

            def use(p, vt):
                o_accs[2 * p] = mfma(vt.shuffle(vt, [0, 1, 2, 3]).ir_value(), p_pack, o_accs[2 * p])
                o_accs[2 * p + 1] = mfma(vt.shuffle(vt, [4, 5, 6, 7]).ir_value(), p_pack, o_accs[2 * p + 1])

            stream(tr.NP, read, use)
            return o_accs

        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        c_neg_inf = fx.Float32(float('-inf'))
        c_neg_max = fx.Float32(-3.4028234663852886e38)
        assert tr.LDS_VT_BASE % 1024 == 0

        def peer(x, mask):
            """x of lane ^ mask. The address comes from vt_rd (lane in bits 4..9) in
            asm, so it is not hoisted into a register live across the loop."""
            addr = llvm.inline_asm(T.i32,
                                   [raw(fx.Int32(vt_rd))],
                                   f'v_lshrrev_b32 $0, 2, $1\n\tv_xor_b32 $0, {mask * 4}, $0',
                                   '=&v,v',
                                   has_side_effects=True)
            xi = fx.Float32(x).bitcast(fx.Int32)
            return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, fx.Int32(addr), xi)).bitcast(fx.Float32)

        def row_max(x):
            x = fx.maxnumf(x, peer(x, 16))
            return fx.maxnumf(x, peer(x, 32))

        def lane_field(shift, width):
            """Bits of the lane id, from vt_rd (lane * 16), behind asm LLVM cannot hoist: a
            value derived from them is rebuilt where used instead of living across loops."""
            return fx.Int32(llvm.inline_asm(T.i32,
                                            [raw(fx.Int32(vt_rd))],
                                            f'v_bfe_u32 $0, $1, {4 + shift}, {width}',
                                            '=v,v',
                                            has_side_effects=True))

        def kv_lane():
            """This lane's first key in a tile: 4 * lg."""
            return lane_field(4, 2) * fx.Int32(4)

        def row_now():
            """q_row, rebuilt."""
            return fx.Int32(q_start) + wave_s * fx.Int32(WAVE_M) + lane_field(0, 4)

        win_l, win_r = band_window(tr.CAUSAL, window_left, window_right, sq_i32, sk_i32, varlen_bits)
        q_first, q_last, kv_base, n_tiles = kv_tiles(q_start, tr.BLOCK_M, tr.BLOCK_N, sq_i32, sk_i32, win_l, win_r)

        def mask_scores(s_vals, kv_start):
            band_lo, band_span = row_band(row_now(), win_l, win_r, sk_i32)
            return mask_band(s_vals, fx.Int32(kv_start) + kv_lane() - band_lo, range(4), band_span, c_neg_inf)

        lead, clean_end = clean_tiles(q_first, q_last, kv_base, tr.BLOCK_N, sk_i32, win_l, win_r)
        if const_expr(bias):
            clean_end = smin(clean_end, smax(n_tiles, fx.Index(1)) - fx.Index(1))
        n_loop = (n_tiles > fx.Index(1)).select(n_tiles - fx.Index(1), fx.Index(0))
        # Iteration j masks tile j+1. Masked iterations [0, j_clean) and [j_tail, n_iter)
        # bracket the clean ones. Every loop runs pairs, so an odd n_loop gets one more
        # iteration, whose tile n_tiles lies past kv_hi: all masked, P = 0. A parity
        # branch instead made LLVM spill O around it.
        n_iter = n_loop + n_loop % 2
        j_clean = smin(smax(lead, fx.Index(1)) - fx.Index(1), n_loop)
        j_clean = j_clean + j_clean % 2
        j_clean_end = smax(smin(smax(clean_end, fx.Index(1)) - fx.Index(1), n_loop), j_clean)

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
            b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + fx.Int32(lg * 4)) * fx.Int32(2)
            b_lane = b_byte & fx.Int32(-4)
            b_shift = (b_byte & fx.Int32(3)) * fx.Int32(8)

        def n_bias_loads(exact):
            return (4 if exact else 1) if bias else 0

        def bias_load(kv_start, exact):
            """This lane's 4 bias values of one tile; `exact` tiles read 2 bytes at a time:
            the slab's last element may end mid-dword."""
            if const_expr(not bias):
                return None
            soff = fx.Int32(kv_start) * fx.Int32(2)
            return bias_load4(b_rsrc, b_byte if exact else b_lane, soff, b_shift, exact, elem)

        def wave_any(pred):
            return fx.Int64(rocdl.ballot(fx.Int64.ir_type, raw(pred))) != fx.Int64(0)

        def softmax_head(s_raw, m_old, kv_start, masked, bias_vals=None):
            """Scale, mask, and update the running max."""
            # Scale once in fp32; the max and the exponent share that rounding.
            sv = Vec(s_raw) * fx.full(4, fx.Float32(c_scale), fx.Float32)
            s_vals = [sv[i] for i in range(4)]
            if const_expr(bias):
                # Upcast first: log2(e) in the input dtype would differ from the backward's.
                s_vals = [s_vals[i] + fx.Float32(Vec(bias_vals)[i].to(fx.Float32)) * fx.Float32(LOG2E)
                          for i in range(4)]
            if const_expr(masked):
                s_vals = mask_scores(s_vals, kv_start)
            local = fx.maxnumf(fx.maxnumf(s_vals[0], s_vals[1]), fx.maxnumf(s_vals[2], s_vals[3]))
            m_new, corr = max_update(local, m_old)
            return s_vals, m_new, corr

        @flyc.jit
        def max_update(local, m_old):
            """Skips the cross-lane exchange when no quarter-row max passed the threshold;
            exact at threshold 0, since then no max grew."""
            m_new = m_old
            corr = fx.Float32(1.0)
            if wave_any(local > m_old + fx.Float32(rescale_threshold)):
                m_new = fx.maxnumf(m_old, row_max(local))
                corr = exp2(m_old - m_new)
            return m_new, corr

        @flyc.jit
        def rescale_o(o_accs, corr):
            out = list(o_accs)
            if wave_any(corr != fx.Float32(1.0)):
                corr_v = fx.full(4, fx.Float32(corr), fx.Float32)
                out = [Vec(o) * corr_v for o in o_accs]
            return out

        if const_expr(dropout):
            # The Triton kernels' stream, element for element (philox.py, grid_plane).
            rng, ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            fmha.philox_report(philox_seed_output, philox_offset_output, ph_seed, ph_off)
            ph_base, ph_stride = philox_plane(rng, ph_off, z_i32, num_head_q, head, max_seqlen_q, max_seqlen_k)
            # The wave's part of the offset is uniform; the lane adds row * stride + col // 4.
            ph_wave = rng.grid_offset(ph_base, ph_stride, fx.Int32(q_start) + wave_s * fx.Int32(WAVE_M), 0)
            ph_stride32 = fx.Int32(ph_stride)

        def zext64(x):
            return fx.Int64(arith.extui(T.i64, raw(fx.Int32(x))))

        def softmax_tail(s_vals, m_new, corr, l_old, kv_start):
            p = [exp2(s_vals[i] - m_new) for i in range(4)]
            l_new = l_old * corr + ((p[0] + p[1]) + (p[2] + p[3]))
            if const_expr(dropout):
                # l sums P before the mask; 1/(1-p) joins 1/l at the end.
                # One v_mad_u64_u32: zero-extended row * stride + col // 4.
                lane_off = zext64(lane_field(0, 4)) * zext64(ph_stride32) + zext64(lane_field(4, 2))
                vals = rng.u32(ph_seed, ph_wave + fx.Int64(fx.Int32(kv_start) >> fx.Int32(2)) + lane_off)
                p = [(fx.Int32(vals[i]) > fx.Int32(idropout_p)).select(p[i], fx.Float32(0.0)) for i in range(4)]
            return pack_p(p), l_new

        # One cluster and one barrier per tile; iteration j: DMA K(j+2) -> k[j%2],
        # exp(j), QK(j+1) from k[(j+1)%2], load V(j+1), PV(j) from vt[j%2], max(j+1),
        # V^T(j+1) -> vt[(j+1)%2], wait K(j+2). Unrolled by two so buffer indices
        # (and alias scopes) are constants. With no stagger, one barrier separates
        # every producer from its consumer.
        n_kdma = tr.K_DMA_ROUNDS
        n_v = tr.V_UNITS_PER_THREAD * 4

        # Prologue: K(0), K(1), V(0); then iteration -1's half.
        k_dma(kv_base, 0)
        # The vmcnt below counts everything after K(0) as newer: keep it after the DMA.
        rocdl.sched_barrier(0)
        b0 = bias_load(kv_base, True)
        v_regs = v_load(kv_base)
        k_dma(kv_base + tr.BLOCK_N, 1)
        rocdl.s_waitcnt(vmcnt=n_bias_loads(True) + n_v + n_kdma)
        k_fix(0)
        rocdl.s_waitcnt(lgkmcnt=0)
        cluster_barrier()
        v_store_lds(v_regs, 0)
        s_raw = qk(0)
        # A finite initial max keeps fully masked rows NaN-free.
        s0, m0, corr0 = softmax_head(s_raw, c_neg_max, kv_base, True, b0)
        rocdl.s_waitcnt(vmcnt=0)
        k_fix(1)
        rocdl.s_waitcnt(lgkmcnt=0)
        cluster_barrier()

        n_o = tr.D // 16

        def iteration(j, carried, b, masked):
            """Tile j's PV and tile j+1's QK; `b` = j % 2, known at trace time."""
            m_cur = carried[0]
            corr = carried[1]
            l_old = carried[2]
            s_cur = carried[3:7]
            o_accs = list(carried[7:7 + n_o])
            kv_cur = kv_base + j * tr.BLOCK_N
            kv_next = kv_cur + tr.BLOCK_N
            b_next = bias_load(kv_next, masked)
            k_dma(kv_next + tr.BLOCK_N, b)
            o_accs = rescale_o(o_accs, corr)
            p_pack, l_new = softmax_tail(s_cur, m_cur, corr, l_old, kv_cur)
            s_next = qk(1 - b)
            # V(j+1) lives only through PV(j): loaded after QK(j+1), stored at the end.
            rocdl.sched_barrier(0)
            v_regs = v_load(kv_next)
            o_accs = pv(o_accs, p_pack, b)
            s_vals, m_next, corr_next = softmax_head(s_next, m_cur, kv_next, masked, b_next)
            v_store_lds(v_regs, 1 - b)
            rocdl.s_waitcnt(vmcnt=0)
            k_fix(b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            return [m_next, corr_next, l_new] + s_vals + o_accs

        def finish(carried, b, kv_last):
            """The last tile's exp and PV; its V^T is already in vt[b]."""
            o_last = rescale_o(list(carried[7:7 + n_o]), carried[1])
            p_pack, l_last = softmax_tail(carried[3:7], carried[0], carried[1], carried[2], kv_last)
            o_last = pv(o_last, p_pack, b)
            return [carried[0], l_last] + o_last

        # Every loop runs pairs (even, odd), so buffer indices are trace-time constants.
        init = [m0, corr0, fx.Float32(0.0)] + s0 + [fx.full(4, 0.0, fx.Float32) for _ in range(n_o)]
        results0 = init
        for jp, carried in range(0, j_clean // 2, 1, init=init):
            mid = iteration(jp * 2, carried, 0, True)
            results0 = yield iteration(jp * 2 + 1, mid, 1, True)
        n_pairs = (j_clean_end - j_clean) // 2
        results1 = results0
        for jp, carried in range(0, n_pairs, 1, init=results0):
            mid = iteration(j_clean + jp * 2, carried, 0, False)
            results1 = yield iteration(j_clean + jp * 2 + 1, mid, 1, False)
        j_tail = j_clean + n_pairs * 2
        results2 = results1
        for jp, carried in range(0, (n_iter - j_tail) // 2, 1, init=results1):
            mid = iteration(j_tail + jp * 2, carried, 0, True)
            results2 = yield iteration(j_tail + jp * 2 + 1, mid, 1, True)
        results = finish(results2, 0, kv_base + n_iter * tr.BLOCK_N)

        rocdl.s_waitcnt(0)
        m_fin = results[0]
        l_fin = results[1]
        l_fin = l_fin + peer(l_fin, 16)
        l_fin = l_fin + peer(l_fin, 32)
        inv_l = fx.Float32(1.0) / l_fin
        inv_l = (l_fin > fx.Float32(0.0)).select(inv_l, fx.Float32(0.0))
        if const_expr(dropout):
            inv_l = inv_l * fx.Float32(dropout_scale)
        q_row = fx.Index(row_now())
        q_ok = q_row < sq
        inv_v = fx.full(4, fx.Float32(inv_l), fx.Float32)
        lane_g = lane_field(4, 2)
        for p in range_constexpr(tr.NP):
            # Lane holds rows d = 32p + 8*lg + 2i + e of MFMA 2p + e: 8 consecutive columns.
            o0 = Vec(results[2 + 2 * p]) * inv_v
            o1 = Vec(results[3 + 2 * p]) * inv_v
            vals = []
            for i in range_constexpr(4):
                vals += [o0[i], o1[i]]
            packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
            d = fx.Index(32 * p) + fx.Index(lane_g) * 8
            ok = q_ok
            if const_expr(32 * p + 32 > tr.PAD_FLOOR):
                ok = ok & (fx.Int32(d) < row_vo)
            buffer_ops.buffer_store(packed.ir_value(), o_rsrc, q_row * o_row_stride + d, mask=ok)
        # LSE: natural base, +inf for fully masked rows; a null L gets an empty range.
        lse = (m_fin + fx.Float32(fx.math.log2(l_fin))) * fx.Float32(LN2)
        lse = (l_fin > fx.Float32(0.0)).select(lse, fx.Float32(float('inf')))
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)
        lse_row0, lse_pitch = fmha.lse_row_addressing(varlen_bits,
                                                      fx.Index(q_batch),
                                                      head,
                                                      num_head_q,
                                                      lse_tokens,
                                                      fx.Index(q_row_off))
        lse_addr = fx.Int64(fx.ptrtoint(LSE))
        lse_base = lse_addr + fx.Int64(lse_row0) * fx.Int64(4)
        lse_bytes = (lse_addr == fx.Int64(0)).select(fx.Int64(0), fx.Int64(sq * lse_pitch) * fx.Int64(4))
        lse_rsrc = buffer_ops.create_buffer_resource_from_addr(lse_base, num_records_bytes=lse_bytes)
        buffer_ops.buffer_store(lse, lse_rsrc, q_row * lse_pitch, mask=q_ok & (lane_field(4, 2) == fx.Int32(0)))

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
        grid_y = (fx.Index(max_seqlen_q) + tr.BLOCK_M - 1) // tr.BLOCK_M
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
                                            'rocdl.flat_work_group_size': f'{tr.BLOCK_SIZE},{tr.BLOCK_SIZE}'}).launch(
            grid=(grid_y, fx.Index(num_head_q), grid_z) if tile_fastest else (fx.Index(num_head_q), grid_y, grid_z),
            block=(tr.BLOCK_SIZE, 1, 1),
            stream=stream)

    return fmha.make_fwd_launch(launch_fmha_fwd_gfx942_xwide,
                                tr,
                                causal_type,
                                host_causal_type,
                                bias,
                                dropout)
