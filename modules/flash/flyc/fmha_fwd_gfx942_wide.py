# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention forward for gfx942 at head dims 160..256, FlyDSL.

fmha_fwd_gfx942.py's math (S^T = K.Q^T, O^T += V^T.P^T) with gfx950's family-B
geometry: 4 waves x 32 rows, one wave per SIMD, so a lane has 512 VGPR+AGPR.
No second wave hides memory, so each KV tile is two barrier-separated clusters
that each mix LDS traffic into their MFMAs:
    X(j): QK(j+1), softmax tail(j)
    Y(j): PV(j), softmax head(j+1)
and the K / V^T tiles reach registers a cluster before their MFMAs where they
fit (K_EARLY, VT_EARLY), else stream VT_LEAD reads ahead. K is double-buffered
by DMA; V^T is single-buffered (the waves run in lock-step), which fits head
dim 256 in 64 KB. Causal blocks run two q tiles (PAIR_TILES) so every block
does the same work; the grid then counts GRID_BLOCK_M = 256 rows per block.
"""


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, ktrace, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
import fmha_tuning_gfx942 as tuning
from fmha_fwd_gfx942 import principal_diagonal_bits
from philox import Philox

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_fwd_gfx942_wide.lds">'
LDS_SCOPES = ('k0', 'k1', 'vt')
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
WAVE = 64
WAVE_M = 32         # query rows per wave: the N of the 32x32x8 MFMA
# Dev sweeps only: Traits field -> value, applied to every build (and its cache tag).
DEV_OVERRIDES = {}
# sched_group_barrier masks
SG_VALU = 0x2
SG_MFMA = 0x8
SG_VMEM = 0x10
SG_DS_READ = 0x100
SG_DS_WRITE = 0x200


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, num_waves, block_n, padded_head=False):
        assert head_dim % 32 == 0 and 32 <= head_dim <= 256, head_dim
        assert block_n == 32, block_n
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
        self.QK_STEPS = head_dim // 8           # MFMA k-steps of QK
        self.D_TILES = head_dim // 32           # 32-row tiles of O^T
        self.PV_STEPS = block_n // 8            # MFMA k-steps of PV per tile
        # K rows an odd number of 16B chunks apart: conflict-free reads, immediate offsets.
        self.K_ROW_BYTES = head_dim * 2
        chunks = head_dim // 8
        self.K_STRIDE = (chunks + 1 - chunks % 2) * 16
        self.K_TILE_BYTES = block_n * self.K_STRIDE
        # K DMA: each wave copies whole rows, 256B per issue; the last issue of a row
        # ends on its last byte and may overlap the one before (same bytes, same place).
        assert self.K_ROW_BYTES >= 256 and block_n % num_waves == 0
        self.K_SEGS = tuple(range(0, self.K_ROW_BYTES - 256, 256)) + (self.K_ROW_BYTES - 256,)
        self.K_ROWS_PER_WAVE = block_n // num_waves
        self.K_DMAS = self.K_ROWS_PER_WAVE * len(self.K_SEGS)
        # V^T tile: [D][BLOCK_N + 4]; the 8B pad spreads 16 lanes over all banks.
        self.VT_STRIDE = block_n + 4
        self.VT_TILE_BYTES = head_dim * self.VT_STRIDE * 2
        # V global load: one unit per thread = 4 kv rows x 8 d columns, four dwordx4
        # loads; spare threads redo a unit (same bytes to the same place).
        self.V_UNITS = (block_n // 4) * (head_dim // 8)
        assert self.V_UNITS <= self.BLOCK_SIZE, self.V_UNITS
        # Whole K / V^T tiles to registers a cluster ahead; past 224 they do not fit with O.
        self.K_EARLY = head_dim <= 224
        self.VT_EARLY = head_dim <= 224
        self.VT_LEAD = 8                        # streamed LDS reads issued ahead of the MFMAs
        self.VALU_PER_MFMA = 3                  # sched_group_barrier slot size
        self.TRACE_CLUSTERS = True              # ktrace marks per cluster (only under FLYDSL_KTRACE_ENABLE)
        # Causal blocks pair q tiles g and T-1-g; the grid counts blocks of GRID_BLOCK_M rows.
        self.PAIR_TILES = causal
        self.LDS_K_BASE = 0
        self.LDS_VT_BASE = 2 * self.K_TILE_BYTES
        self.LDS_BYTES = self.LDS_VT_BASE + self.VT_TILE_BYTES
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        for k, v in DEV_OVERRIDES.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        self.GRID_BLOCK_M = self.BLOCK_M * (2 if self.PAIR_TILES else 1)
        # The JIT cache keys on scalar closure values; the kernel references this.
        self.CACHE_TAG = (f'W{head_dim}_c{int(causal)}_{dtype_str}_w{num_waves}_n{block_n}_p{int(padded_head)}'
                          f'_k{int(self.K_EARLY)}_v{int(self.VT_EARLY)}_l{self.VT_LEAD}_a{self.VALU_PER_MFMA}'
                          f'_t{int(self.TRACE_CLUSTERS)}_q{int(self.PAIR_TILES)}')


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
    host_causal_type = (1 if causal else 0) if causal_type is None else causal_type
    if host_causal_type not in (0, 1, 2, 3):
        raise ValueError(f'causal_type must be 0-3, got {host_causal_type}')
    causal_type = 0 if host_causal_type == 0 else 3
    if bias and causal_type:
        raise ValueError('bias with a causal mask is not served (aot/_common.py flash_disabled)')
    if block_n is None:
        block_n = 32
    tr = Traits(head_dim, causal_type != 0, dtype_str, num_waves, block_n, padded_head)
    if grid_axis_order is None:
        # Tile-fastest shares K/V in L2; paired tiles balance causal blocks (B4 H32 S8k
        # d192 causal: 413 vs 386 TFLOPS head-fastest; B2 H64 S32k: 445 vs 325).
        grid_axis_order = tuning.GRID_AXIS_TILE_FASTEST
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_thr{rescale_threshold}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}'
    elem = fx.BFloat16 if dtype_str == 'bf16' else fx.Float16

    # MLIR types need a live context, so they are built at trace time.
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

    def pack_p(vals):
        """Four f32 in [0, 1] -> one operand; bf16 rounds to nearest, ties away (lessons #16)."""
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

    def cluster_barrier(name=None):
        """End of a cluster; under FLYDSL_KTRACE_ENABLE=1, marks its end and the barrier release."""
        name = name if tr.TRACE_CLUSTERS else None
        rocdl.sched_barrier(0)
        if name is not None:
            ktrace.mark(name)
        rocdl.s_barrier()
        if name is not None:
            ktrace.mark(name + '>')
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
    # The kernel ABI is flash_attn_aiw_kernel's (gfx1201), so aot/flyc_attn_fwd.py can serve both.
    def fmha_fwd_gfx942_wide_kernel(
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
        lm = lane % 32
        lh = lane // 32
        if const_expr(tile_fastest):
            g_tile = fx.Index(gpu.block_idx.x)
            head = fx.Index(gpu.block_idx.y)
        else:
            head = fx.Index(gpu.block_idx.x)
            g_tile = fx.Index(gpu.block_idx.y)
        if const_expr(tr.PAIR_TILES):
            # Block g runs q tiles g and T-1-g: causal blocks then all do the same work.
            # T odd: the middle block's second tile is T, which has no rows.
            n_q_tiles = (fx.Index(max_seqlen_q) + tr.BLOCK_M - 1) // tr.BLOCK_M
            mirror = n_q_tiles - fx.Index(1) - g_tile
            q_tiles = [mirror, (mirror == g_tile).select(n_q_tiles, g_tile)]
        else:
            q_tiles = [g_tile]
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
        o_rsrc = rsrc_for(O, q_batch, head, q_row_off, stride_o_batch, stride_o_head, stride_o_seq, sq, row_vo)
        k_rsrc = rocdl.get_buffer_rsrc(
            rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk_i32, row_qk))
        v_rsrc_raw = rocdl.get_buffer_rsrc(
            rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk_i32, row_vo))
        q_row_stride = fx.Index(fx.Int32(stride_q_seq))
        k_row_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_row_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)
        o_row_stride = fx.Index(fx.Int32(stride_o_seq))

        for q_pass in range_constexpr(len(q_tiles)):
            q_tile = q_tiles[q_pass]
            q_start = q_tile * tr.BLOCK_M
            q_wave = q_start + wave * WAVE_M
            q_row = q_wave + lm

            # Q^T packs: k-step 2t+e reads d = 16t + 8*lh + 4e + [0, 4).
            def k_col_mask(t):
                """Dword masks keeping this lane's columns 16t + 8lh + [0, 8) below hdim_qk."""
                c0 = fx.Int32(16 * t) + fx.Int32(lh * 8)
                words = []
                for w in range_constexpr(4):
                    lo = (c0 + fx.Int32(2 * w) < fx.Int32(hdim_qk)).select(fx.Int32(0xFFFF), fx.Int32(0))
                    hi = (c0 + fx.Int32(2 * w + 1) < fx.Int32(hdim_qk)).select(fx.Int32(-65536), fx.Int32(0))
                    words.append(lo | hi)
                return Vec.from_elements(words, fx.Int32)

            q_packs = []
            for t in range_constexpr(tr.QK_STEPS // 2):
                off = q_row * q_row_stride + fx.Index(16 * t) + lh * 8
                v8 = buffer_ops.buffer_load(q_rsrc, off, vec_width=8, dtype=elem)
                if const_expr(16 * t + 16 > tr.PAD_FLOOR):
                    # Slack columns may hold NaN, and 0 * NaN is NaN: zero both Q and K there.
                    v8 = Vec(v8).bitcast(fx.Int32) & k_col_mask(t)
                    v8 = Vec(v8).bitcast(elem).ir_value()
                q_packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
                q_packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())

            # K DMA: voffset is the lane's dword; row, tile and segment ride in M0, soffset, offset.
            wave_s = fx.Index(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave)))
            k_lane_off = fx.Int32(lane * 4)
            k_m0_addr = fx.Int32(lds_base + fx.Index(tr.LDS_K_BASE) + wave_s * (tr.K_ROWS_PER_WAVE * tr.K_STRIDE))
            k_m0_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, k_m0_addr))
            k_row0 = fx.Int32(wave_s * tr.K_ROWS_PER_WAVE)

            def k_dma(kv_start, buf):
                al, noal = scope_attrs(f'k{buf}')
                for i in range_constexpr(tr.K_ROWS_PER_WAVE):
                    soff = (fx.Int32(kv_start) + k_row0 + fx.Int32(i)) * k_row_stride_b
                    m0 = fx.Int64(k_m0_base + fx.Int32(buf * tr.K_TILE_BYTES + i * tr.K_STRIDE))
                    for seg in tr.K_SEGS:
                        op = rocdl.raw_ptr_buffer_load_lds(k_rsrc,
                                                           lds_ptr_at(m0),
                                                           fx.Int32(4),
                                                           k_lane_off,
                                                           soff,
                                                           fx.Int32(seg),
                                                           fx.Int32(0))
                        op = getattr(op, 'owner', op)
                        op = getattr(op, 'operation', op)
                        op.attributes['alias_scopes'] = al
                        op.attributes['noalias_scopes'] = noal

            # V: unit = 4 KV rows x 8 columns; v_perm turns it into 8 V^T rows of 4 keys.
            # Row group fastest: 16 lanes' V^T writes then cover all 32 banks.
            v_unit = tid % tr.V_UNITS
            v_g = v_unit % (tr.BLOCK_N // 4)
            v_c8 = v_unit // (tr.BLOCK_N // 4)
            v_voff = fx.Int32(v_g * 4) * v_row_stride_b + fx.Int32(v_c8 * 16)
            v_vt_base = fx.Index(tr.LDS_VT_BASE) + (v_c8 * 8 * tr.VT_STRIDE + v_g * 4) * 2

            def v_load(kv_start):
                out = []
                for r in range_constexpr(4):
                    soff = (fx.Int32(kv_start) + fx.Int32(r)) * v_row_stride_b
                    w4 = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(4, fx.Int32), v_rsrc_raw, v_voff, soff, 0))
                    out.extend(w4[i] for i in range(4))
                return out

            def v_store_lds(regs):
                sel = (fx.Int32(0x05040100), fx.Int32(0x07060302))
                al, noal = scope_attrs('vt')
                for d in range_constexpr(8):
                    w = [regs[4 * r + d // 2] for r in range(4)]
                    dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel[d % 2])), fx.Int32(rocdl.perm_b32(w[3], w[2], sel[d % 2]))]
                    llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                                 lds_view(v_vt_base + d * tr.VT_STRIDE * 2, fx.Int32, 8).llvm_ptr,
                                 alignment=8,
                                 alias_scopes=al,
                                 noalias_scopes=noal)

            def lds_load(off, ty, align, scope):
                al, noal = scope_attrs(scope)
                return llvm.LoadOp(ty,
                                   lds_view(off, elem, align).llvm_ptr,
                                   alignment=align,
                                   alias_scopes=al,
                                   noalias_scopes=noal).result

            k_rd_off = []
            for t in range_constexpr(tr.QK_STEPS // 2):
                c = fx.Index(2 * t) + lh
                k_rd_off.append(fx.Index(tr.LDS_K_BASE) + lm * tr.K_STRIDE + c * 16)
            vt_rd_off = fx.Index(tr.LDS_VT_BASE) + (lm * tr.VT_STRIDE + lh * 4) * 2

            def k_read(buf):
                return [lds_load(off + buf * tr.K_TILE_BYTES + n * 32 * tr.K_STRIDE, v8e(), 16, f'k{buf}')
                        for n in range(tr.NSUB) for off in k_rd_off]

            k_masks = {t: k_col_mask(t) for t in range(tr.QK_STEPS // 2) if 16 * t + 16 > tr.PAD_FLOOR}

            def qk(k8s):
                """S^T sub-tiles, one per 32 KV rows: [v16f32] * NSUB."""
                out = []
                nt = tr.QK_STEPS // 2
                k8s = [Vec(Vec(k8).bitcast(fx.Int32) & k_masks[i % nt]).bitcast(elem).ir_value() if i % nt in k_masks
                       else k8 for i, k8 in enumerate(k8s)]
                for n in range_constexpr(tr.NSUB):
                    s = fx.full(16, 0.0, fx.Float32)
                    for t in range_constexpr(nt):
                        k8 = Vec(k8s[n * nt + t])
                        s = mfma(k8.shuffle(k8, [0, 1, 2, 3]).ir_value(), q_packs[2 * t], s)
                        s = mfma(k8.shuffle(k8, [4, 5, 6, 7]).ir_value(), q_packs[2 * t + 1], s)
                    out.append(s)
                return out

            def v_read():
                return [[lds_load(vt_rd_off + (32 * dt * tr.VT_STRIDE + 8 * st) * 2, v4e(), 8, 'vt')
                         for st in range(tr.PV_STEPS)] for dt in range(tr.D_TILES)]

            def pv(o_accs, p_packs, vts):
                for dt in range_constexpr(tr.D_TILES):
                    for st in range_constexpr(tr.PV_STEPS):
                        o_accs[dt] = mfma(vts[dt][st], p_packs[st], o_accs[dt])
                return o_accs

            c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
            c_neg_inf = fx.Float32(float('-inf'))
            c_neg_max = fx.Float32(-3.4028234663852886e38)
            swap_addr = fx.Int32(lane ^ fx.Index(32)) * fx.Int32(4)

            def peer(x):
                xi = fx.Float32(x).bitcast(fx.Int32)
                return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, swap_addr, xi)).bitcast(fx.Float32)

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
                # Unbounded: no row index exceeds seqlen_q, no column seqlen_k.
                win_l = sq_i32
                win_r = sk_i32
            q_first = fx.Int32(q_start)
            q_last = smin(q_first + fx.Int32(tr.BLOCK_M), sq_i32) - fx.Int32(1)
            kv_lo = smax(q_first - win_l, fx.Int32(0))
            kv_hi = smin(q_last + win_r + fx.Int32(1), sk_i32)
            # Blocks past a short sequence (varlen), or whose band misses every key, do no tiles.
            alive = (q_first < sq_i32) & (kv_hi > kv_lo)
            t0 = fx.Index(alive.select(kv_lo, fx.Int32(0))) // tr.BLOCK_N
            kv_base = t0 * tr.BLOCK_N
            n_tiles = alive.select((fx.Index(kv_hi) + tr.BLOCK_N - 1) // tr.BLOCK_N - t0, fx.Index(0))

            # This lane's row keeps keys [band_lo, band_lo + band_span]: one unsigned compare
            # per key. An empty band moves band_lo out of reach of any key, so all fail.
            _row = fx.Int32(q_row)
            _hi = smin(_row + win_r, sk_i32 - fx.Int32(1))
            _lo = smax(_row - win_l, fx.Int32(0))
            _empty = _hi < _lo
            band_lo = _empty.select(fx.Int32(-(1 << 30)), _lo)
            band_span = _empty.select(fx.Int32(0), _hi - _lo)

            def mask_scores(s_vals, kv_start):
                # Lane value r is KV row 32*(r//16) + 8*((r%16)//4) + 4*lh + r%4.
                rel0 = fx.Int32(kv_start) + kv_lane - band_lo
                out = []
                for r in range_constexpr(tr.NS):
                    rel = rel0 + fx.Int32(32 * (r // 16) + 8 * ((r % 16) // 4) + r % 4)
                    out.append((fx.Uint32(rel) > fx.Uint32(band_span)).select(c_neg_inf, s_vals[r]))
                return out

            # Tile r (from t0) is clean, i.e. unmasked for every row of the block,
            # iff lead <= r < clean_end.
            def n_rel(kv):
                return fx.Index(smax(smin(kv, sk_i32), fx.Int32(0)))

            lead = (n_rel(q_last - win_l - fx.Int32(kv_base)) + tr.BLOCK_N - 1) // tr.BLOCK_N
            clean_end = n_rel(smin(q_first + win_r, sk_i32 - fx.Int32(1)) + fx.Int32(1) - fx.Int32(kv_base)) // tr.BLOCK_N
            if const_expr(bias):
                # Only the last tile can hold the dword that straddles the bias slab's end.
                clean_end = smin(clean_end, smax(n_tiles, fx.Index(1)) - fx.Index(1))
            n_loop = (n_tiles > fx.Index(1)).select(n_tiles - fx.Index(1), fx.Index(0))
            # Iteration j masks tile j+1. Masked iterations [0, j_clean) and [j_tail, n_loop)
            # bracket the clean ones; j_clean is even so the unrolled loop knows its buffers.
            j_clean = smin(smax(lead, fx.Index(1)) - fx.Index(1), n_loop)
            j_clean = smin(j_clean + j_clean % 2, n_loop)
            j_clean_end = smax(smin(smax(clean_end, fx.Index(1)) - fx.Index(1), n_loop), j_clean)

            if const_expr(bias):
                # (B, H, Sq, Sk), Sk contiguous; rows offset by Q's varlen offset, columns by K's.
                b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head) * stride_b_head
                           + fx.Int64(q_row_off) * stride_b_seq_q + fx.Int64(k_row_off))
                # Odd-pitch rows start mid-dword: align the base down, funnel-shift per lane.
                b_addr = fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2)
                b_mis = fx.Int32(b_addr & fx.Int64(3))
                # The range ends on the slab's last element (fmha_dualwave_gfx950._bias_slab_num_records_bytes).
                b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(sk_i32), fx.Int64(0))
                b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                    b_addr - fx.Int64(b_mis),
                    num_records_bytes=b_span * fx.Int64(2) + fx.Int64(b_mis)))
                b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + kv_lane) * fx.Int32(2)
                b_lane = b_byte & fx.Int32(-4)
                b_shift = (b_byte & fx.Int32(3)) * fx.Int32(8)

            def bias_load(kv_start, exact):
                """This lane's bias for one KV tile in S^T order; `exact` reads 2 bytes at a time
                so the slab's last element survives the dword range check."""
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

            def scale_and_mask(s_raw, kv_start, masked, bias_vals):
                # Scale once in fp32; the max and the exponent share that rounding.
                c_scale_v = fx.full(16, fx.Float32(c_scale), fx.Float32)
                scaled = [Vec(sr) * c_scale_v for sr in s_raw]
                s_vals = [scaled[r // 16][r % 16] for r in range(tr.NS)]
                if const_expr(bias):
                    # Upcast first: log2(e) in the input dtype would differ from the backward's.
                    s_vals = [s_vals[r] + fx.Float32(Vec(bias_vals[r // 4])[r % 4].to(fx.Float32)) * fx.Float32(LOG2E)
                              for r in range(tr.NS)]
                if const_expr(masked):
                    s_vals = mask_scores(s_vals, kv_start)
                return s_vals

            def wave_any(pred):
                return fx.Int64(rocdl.ballot(fx.Int64.ir_type, raw(pred))) != fx.Int64(0)

            def softmax_head(s_raw, m_old, kv_start, masked, bias_vals=None, sched=None):
                """Scale, mask, and update the running max; `sched` pins the region's order first."""
                s_vals = scale_and_mask(s_raw, kv_start, masked, bias_vals)
                local = s_vals[0]
                for r in range_constexpr(1, tr.NS):
                    local = fx.maxnumf(local, s_vals[r])
                if const_expr(sched is not None):
                    sched()
                m_new, corr = max_update(local, m_old)
                return s_vals, m_new, corr

            @flyc.jit
            def max_update(local, m_old):
                """Skips the cross-half exchange when no half-row max passed the threshold;
                exact at threshold 0, since then no max grew."""
                m_new = m_old
                corr = fx.Float32(1.0)
                if wave_any(local > m_old + fx.Float32(rescale_threshold)):
                    m_new = fx.maxnumf(m_old, fx.maxnumf(local, peer(local)))
                    corr = exp2(m_old - m_new)
                return m_new, corr

            @flyc.jit
            def rescale_o(o_accs, corr):
                out = list(o_accs)
                if wave_any(corr != fx.Float32(1.0)):
                    corr_v = fx.full(16, fx.Float32(corr), fx.Float32)
                    out = [Vec(o) * corr_v for o in o_accs]
                return out

            if const_expr(dropout):
                # The Triton kernels' stream, element for element (philox.py, grid_plane).
                rng = Philox.for_arch('gfx942')
                ph_seed = fmha.philox_seed_value(philox_seed_ptr)
                ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
                fmha.philox_report(philox_seed_output, philox_offset_output, ph_seed, ph_off)
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

            def softmax_tail(s_vals, m_new, corr, l_old, kv_start):
                p = [exp2(s_vals[r] - m_new) for r in range(tr.NS)]
                psum = p[0]
                for r in range_constexpr(1, tr.NS):
                    psum = psum + p[r]
                l_new = l_old * corr + psum
                if const_expr(dropout):
                    # l sums P before the mask; 1/(1-p) joins 1/l at the end.
                    keep = dropout_keep(kv_start)
                    p = [keep[r].select(p[r], fx.Float32(0.0)) for r in range(tr.NS)]
                return [pack_p(p[4 * i:4 * i + 4]) for i in range(tr.PV_STEPS)], l_new

            # LDS lifetimes (all waves in lock-step). k_early / vt_early read a tile into
            # registers a cluster before its MFMAs; otherwise the MFMAs stream it.
            #   K(t):   k_early: DMA X(t-3) -> kb[t%2], waited end X(t-2), read Y(t-2), QK X(t-1)
            #           else:    DMA Y(t-3) -> kb[t%2], waited end Y(t-2), read X(t-1) by QK
            #   V^T(t): vt_early: loaded Y(t-2), written Y(t-1) -> vt, read X(t); PV Y(t)
            #           else:     loaded X(t-1), written X(t) -> vt, read Y(t) by PV
            # Unrolled by two so K buffer indices (and alias scopes) are constants.
            k_early = tr.K_EARLY
            vt_early = tr.VT_EARLY
            n_kdma = tr.K_DMAS
            n_v = 4
            n_qk = tr.QK_STEPS * tr.NSUB
            n_pv = tr.D_TILES * tr.PV_STEPS
            n_k = tr.QK_STEPS // 2 * tr.NSUB        # K reads per tile
            n_vt = n_pv // 2                        # V^T reads per tile (read2_b64)

            def sgb(mask, n):
                if const_expr(n > 0):
                    rocdl.sched_group_barrier(mask, n, 0)

            def sched(n_mfma, n_rd, n_wr, n_vmem, lead=0):
                """Spread LDS reads (`lead` of them first), LDS writes and VMEM over the MFMAs."""
                sgb(SG_DS_READ, lead)
                n_rd -= lead
                for i in range_constexpr(n_mfma):
                    sgb(SG_MFMA, 1)
                    sgb(SG_DS_READ, (i + 1) * n_rd // n_mfma - i * n_rd // n_mfma)
                    sgb(SG_DS_WRITE, (i + 1) * n_wr // n_mfma - i * n_wr // n_mfma)
                    sgb(SG_VMEM, (i + 1) * n_vmem // n_mfma - i * n_vmem // n_mfma)
                    sgb(SG_VALU, tr.VALU_PER_MFMA)

            def anchor(vals):
                """Pins `vals` here: LLVM would otherwise sink their producers into the next cluster."""
                irs = [raw(v) for v in vals]
                ty = ir.Type.parse('!llvm.struct<(' + ', '.join(str(x.type) for x in irs) + ')>')
                cons = ','.join(['=v'] * len(irs) + [str(i) for i in range(len(irs))])
                ret = llvm.inline_asm(ty, irs, '', cons, has_side_effects=True)
                return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(len(irs))]

            # Prologue: K(0), K(1) land; X(-1) = QK(0); Y(-1) = softmax head(0).
            k_dma(kv_base, 0)
            k_dma(kv_base + tr.BLOCK_N, 1)
            # The vmcnt below counts V and bias as the newest loads: keep them after the DMAs.
            rocdl.sched_barrier(0)
            v_regs = v_load(kv_base)
            b0 = bias_load(kv_base, True)
            rocdl.s_waitcnt(vmcnt=n_v + (tr.NS if bias else 0))
            cluster_barrier()
            if const_expr(k_early):
                k_regs = k_read(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()
            if const_expr(k_early):
                k_dma(kv_base + 2 * tr.BLOCK_N, 0)
                s_raw = qk(k_regs)
            else:
                s_raw = qk(k_read(0))
                rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            if const_expr(k_early):
                k_regs = k_read(1)
            if const_expr(vt_early):
                v_store_lds(v_regs)
                v_regs = v_load(kv_base + tr.BLOCK_N)
            if const_expr(not k_early):
                k_dma(kv_base + 2 * tr.BLOCK_N, 0)
            # A finite initial max keeps fully masked rows NaN-free.
            s0, m0, corr0 = softmax_head(s_raw, c_neg_max, kv_base, True, b0)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            if const_expr(not k_early):
                k_regs = []

            n_s = tr.NS
            n_kc = len(k_regs)                      # K registers carried

            def unpack(carried):
                o0 = 3 + n_s
                return (carried[0],
                        carried[1],
                        carried[2],
                        list(carried[3:o0]),
                        list(carried[o0:o0 + tr.D_TILES]),
                        list(carried[o0 + tr.D_TILES:o0 + tr.D_TILES + n_kc]),
                        list(carried[o0 + tr.D_TILES + n_kc:]))

            def iteration(j, carried, b, masked):
                """X(j), Y(j): QK(j+1), PV(j); `b` = j % 2, known at trace time."""
                m_cur, corr, l_old, s_cur, o_accs, k_regs, v_regs = unpack(carried)
                kv_cur = kv_base + j * tr.BLOCK_N
                kv_next = kv_cur + tr.BLOCK_N
                # X(j)
                b_next = bias_load(kv_next, masked)
                if const_expr(k_early):
                    k_dma(kv_next + 2 * tr.BLOCK_N, 1 - b)
                if const_expr(not vt_early):
                    v_store_lds(v_regs)
                    v_regs = v_load(kv_next)
                vts = v_read() if vt_early else None
                s_next = qk(k_regs if k_early else k_read(1 - b))
                p_packs, l_new = softmax_tail(s_cur, m_cur, corr, l_old, kv_cur)
                p_packs = anchor(p_packs)
                sched(n_qk,
                      (n_vt if vt_early else 0) + (0 if k_early else n_k),
                      0 if vt_early else 8,
                      (n_kdma if k_early else 0) + (0 if vt_early else n_v),
                      0 if k_early else tr.VT_LEAD)
                if const_expr(k_early):
                    rocdl.s_waitcnt(vmcnt=n_v + n_kdma)
                if const_expr(vt_early):
                    # Y(j) overwrites V^T; s_barrier does not wait for these reads.
                    rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier(f'X{b}')
                # Y(j)
                o_accs = rescale_o(o_accs, corr)
                if const_expr(k_early):
                    k_regs = k_read(b)
                if const_expr(vt_early):
                    v_store_lds(v_regs)
                    v_regs = v_load(kv_next + tr.BLOCK_N)
                if const_expr(not k_early):
                    k_dma(kv_next + 2 * tr.BLOCK_N, 1 - b)
                o_accs = pv(o_accs, p_packs, vts if vt_early else v_read())
                def sched_y():
                    sched(n_pv,
                          (n_k if k_early else 0) + (0 if vt_early else n_vt),
                          8 if vt_early else 0,
                          (n_v if vt_early else 0) + (0 if k_early else n_kdma),
                          0 if vt_early else tr.VT_LEAD)

                s_vals, m_next, corr_next = softmax_head(s_next, m_cur, kv_next, masked, b_next, sched_y)
                if const_expr(not k_early):
                    rocdl.s_waitcnt(vmcnt=n_v + n_kdma)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier(f'Y{b}')
                return [m_next, corr_next, l_new] + s_vals + o_accs + k_regs + v_regs

            def finish(carried, kv_last):
                """The last tile's PV."""
                m_cur, corr, l_old, s_cur, o_accs, _, v_regs = unpack(carried)
                if const_expr(not vt_early):
                    v_store_lds(v_regs)
                vts = v_read() if vt_early else None
                p_packs, l_last = softmax_tail(s_cur, m_cur, corr, l_old, kv_last)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()
                o_accs = rescale_o(o_accs, corr)
                o_accs = pv(o_accs, p_packs, vts if vt_early else v_read())
                return [m_cur, l_last] + o_accs

            @flyc.jit
            def masked_iteration(j, carried):
                out = list(carried)
                if j % 2 == fx.Index(0):
                    out = iteration(j, carried, 0, True)
                else:
                    out = iteration(j, carried, 1, True)
                return out

            init = ([m0, corr0, fx.Float32(0.0)] + s0 + [fx.full(16, 0.0, fx.Float32) for _ in range(tr.D_TILES)]
                    + k_regs + v_regs)
            results0 = init
            for j, carried in range(0, j_clean, 1, init=init):
                results0 = yield masked_iteration(j, carried)
            n_pairs = (j_clean_end - j_clean) // 2
            results1 = results0
            for jp, carried in range(0, n_pairs, 1, init=results0):
                mid = iteration(j_clean + jp * 2, carried, 0, False)
                results1 = yield iteration(j_clean + jp * 2 + 1, mid, 1, False)
            results2 = results1
            for j, carried in range(j_clean + n_pairs * 2, n_loop, 1, init=results1):
                results2 = yield masked_iteration(j, carried)
            results = finish(results2, kv_base + n_loop * tr.BLOCK_N)

            rocdl.s_waitcnt(0)
            m_fin = results[0]
            l_fin = results[1]
            l_fin = l_fin + peer(l_fin)
            inv_l = fx.Float32(1.0) / l_fin
            inv_l = (l_fin > fx.Float32(0.0)).select(inv_l, fx.Float32(0.0))
            if const_expr(dropout):
                inv_l = inv_l * fx.Float32(dropout_scale)
            q_ok = q_row < sq
            for dt in range_constexpr(tr.D_TILES):
                o = Vec(results[2 + dt]) * fx.full(16, fx.Float32(inv_l), fx.Float32)
                for i in range_constexpr(4):
                    vals = [o[4 * i + jj] for jj in range(4)]
                    packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
                    d = fx.Index(32 * dt + 8 * i) + lh * 4
                    off = q_row * o_row_stride + d
                    ok = q_ok
                    if const_expr(32 * dt + 8 * i + 8 > tr.PAD_FLOOR):
                        ok = ok & (fx.Int32(d) < row_vo)
                    buffer_ops.buffer_store(packed.ir_value(), o_rsrc, off, mask=ok)
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
            buffer_ops.buffer_store(lse, lse_rsrc, q_row * lse_pitch, mask=q_ok & (lh == fx.Index(0)))

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
        grid_y = (fx.Index(max_seqlen_q) + tr.GRID_BLOCK_M - 1) // tr.GRID_BLOCK_M
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_fwd_gfx942_wide_kernel(Q,
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

    compiled = abi.new_compiled_cache()

    def _args(Q,
              K,
              V,
              O,  # noqa: E741
              batch_size,
              seqlen_q,
              seqlen_k=None,
              num_seqlens=0,
              scale=None,
              stream=None,
              lse=None,
              window=None,
              varlen=None,
              bias=None,
              dropout_p=None,
              philox_seed=0,
              philox_offset1=None,
              philox_offset2=0,
              philox_seed_output=None,
              philox_offset_output=None,
              principal_diagonal=None):
        """flash_attn_func_gfx1201_aiw's `_args`: every kernel argument but the stream."""
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = abi.prep_tensors([("Q", Q), ("K", K), ("V", V), ("O", O)], q_heads=("O",))
        if tr.PADDED:
            if not (0 < max(meta[2], meta[3]) <= tr.D):
                raise ValueError(f'head dims {meta[2:]} do not fit a padded {tr.D} build')
        elif meta[2] != tr.D or meta[3] != tr.D:
            raise ValueError(f'head dims {meta[2:]} on an unpadded {tr.D} build; pass padded_head=True')
        lse_p = abi.lse_args(lse, seqlen_q, varlen, meta[0])
        wl, wr = abi.resolve_window(causal_type, host_causal_type, window, seqlen_q, seqlen_k)
        vb, sq0, sq1, sk0, sk1, mq, mk = abi.varlen_args(False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        vb = principal_diagonal_bits(vb, principal_diagonal)
        bp, sb0, sb1, sb2 = bias_args(bias)
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
        so, oo = abi.dropout_outputs(dropout, philox_seed_output, philox_offset_output)
        packed = (*ptrs[:3],
                  bp,
                  ptrs[3],
                  lse_p,
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
                  so,
                  oo,
                  ip,
                  dsc,
                  *meta,
                  abi.resolve_scale(Q, scale, tr.PADDED, tr.D ** -0.5),
                  *st,
                  sb0,
                  sb1,
                  sb2)
        return packed, hold, stream

    def bias_args(t):
        """(pointer, stride_b_batch, stride_b_head, stride_b_seq_q) of a (B, H, Sq, Sk) bias."""
        if not bias:
            if t is not None:
                raise ValueError('bias= on a build without bias')
            return abi.NULL_PTR, 0, 0, 0
        if t is None:
            raise ValueError('this build has bias and requires bias=')
        if t.dim() != 4 or t.stride(3) != 1:
            raise ValueError(f'bias must be (B, H, Sq, Sk) with a contiguous last axis, got {tuple(t.shape)}')
        return abi.ptr_arg(t), t.stride(0), t.stride(1), t.stride(2)

    def _launch(*args, **kwargs):
        packed, _hold, stream = _args(*args, **kwargs)
        abi.run_compiled(compiled, launch_fmha_fwd_gfx942_wide, *packed, fx.Stream(stream))

    _launch.traits = tr
    return _launch
