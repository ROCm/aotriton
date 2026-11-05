# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention forward for gfx942, FlyDSL.

No LDS transpose read on gfx942, so the kernel computes S^T = K.Q^T and
O^T += V^T.P^T: S^T's MFMA output layout is P^T's operand layout, and V is
transposed through VGPRs on its way into LDS.
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

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_fwd_gfx942.lds">'
LDS_SCOPES = ('k0', 'k1', 'vt0', 'vt1')
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
WAVE = 64
WAVE_M = 32         # query rows per wave: the N of the 32x32x8 MFMA
DEBUG_NTILES = False


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, num_waves, block_n):
        assert head_dim % 32 == 0 and 32 <= head_dim <= 512, head_dim
        assert block_n in (32, 64), block_n
        self.D = head_dim
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
        # K rows of whole 256B DMA lines are padded 16B apart; shorter rows XOR-swizzle.
        self.K_ROW_BYTES = head_dim * 2
        self.K_CHUNKS = head_dim // 8           # 16B chunks per row
        self.K_PAD = self.K_ROW_BYTES % 256 == 0
        self.K_STRIDE = self.K_ROW_BYTES + (16 if self.K_PAD else 0)
        self.K_TILE_BYTES = block_n * self.K_STRIDE
        self.DMA_ROUND_BYTES = self.BLOCK_SIZE * 4
        assert (block_n * self.K_ROW_BYTES) % self.DMA_ROUND_BYTES == 0
        self.K_DMA_ROUNDS = block_n * self.K_ROW_BYTES // self.DMA_ROUND_BYTES
        # V^T tile: [D][BLOCK_N + 4]; the 8B pad spreads 16 lanes over all banks.
        self.VT_STRIDE = block_n + 4
        self.VT_TILE_BYTES = head_dim * self.VT_STRIDE * 2
        # V global load: one unit = 4 kv rows x 2 d columns.
        self.V_UNITS = (block_n // 4) * (head_dim // 2)
        assert self.V_UNITS % self.BLOCK_SIZE == 0, 'V tile does not split over the block'
        self.V_UNITS_PER_THREAD = self.V_UNITS // self.BLOCK_SIZE
        # Double buffers suffice for the pipeline's two-cluster safety distance.
        self.NK = 2
        self.NV = 2
        self.LDS_K_BASE = 0
        self.LDS_VT_BASE = self.NK * self.K_TILE_BYTES
        self.LDS_BYTES = self.LDS_VT_BASE + self.NV * self.VT_TILE_BYTES
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        # The JIT cache keys on scalar closure values; the kernel references this.
        self.CACHE_TAG = f'D{head_dim}_c{int(causal)}_{dtype_str}_w{num_waves}_n{block_n}'

    def k_swizzle(self, row):
        """XOR mask for the 16B chunk index of K row `row` (an fx.Index)."""
        if self.K_PAD:
            return 0
        if self.K_CHUNKS >= 8:
            return row % 8
        return (row // 2) % 4


def build_fmha_fwd_gfx942(head_dim,
                          causal=False,
                          dtype_str='bf16',
                          num_waves=8,
                          block_n=None,
                          waves_per_eu=2,
                          rescale_threshold=0.0):
    """`rescale_threshold` > 0 enables the lazy running max (FA4's, off by default;
    numerical lessons, #17): the max may lag that many log2 units."""
    if block_n is None:
        # Keep 32 MFMAs per KV tile: small heads take a longer tile.
        block_n = 64 if head_dim <= 64 else 32
    tr = Traits(head_dim, causal, dtype_str, num_waves, block_n)
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_dbg{int(DEBUG_NTILES)}_thr{rescale_threshold}'
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
        """Four f32 in [0, 1] -> one operand. bf16: round to nearest, ties away, NaN-unsafe,
        as fmha_common_gfx1201.bf16_round_nonan_pack_v8; see the numerical lessons, #16."""
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
        rocdl.sched_barrier(0)
        if name is not None:
            ktrace.mark(name)
        rocdl.s_barrier()
        if name is not None:
            ktrace.mark(name + '>')
        rocdl.sched_barrier(0)

    def barrier_if(flag_i32):
        """s_barrier on the waves whose wave-uniform `flag_i32` is nonzero."""
        llvm.inline_asm(ir.Type.parse('!llvm.void'),
                        [raw(flag_i32)],
                        's_cmp_eq_u32 $0, 0\n\ts_cbranch_scc1 1f\n\ts_barrier\n\t1:',
                        's',
                        has_side_effects=True)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
    def fmha_fwd_gfx942_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
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
    ):
        _ = cache_tag
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WAVE
        lane = tid % WAVE
        lm = lane % 32
        lh = lane // 32
        q_tile = fx.Index(gpu.block_idx.x)
        head = fx.Index(gpu.block_idx.y)
        batch = fx.Index(gpu.block_idx.z)
        head_k = head // (fx.Index(num_head_q) // fx.Index(num_head_k))
        sq = fx.Index(seqlen_q)
        sk = fx.Index(seqlen_k)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_base = fx.Index(fx.ptrtoint(lds_p))

        def lds_view(off, e, align):
            return fx.recast_iter(fx.PointerType.get(e.ir_type, fx.AddressSpace.Shared, align),
                                  lds_p + fx.Int64(off))

        def rsrc_for(ptr, sb, sh, ss, h, rows):
            base = fx.Int64(fx.ptrtoint(ptr)) + (fx.Int64(batch) * sb + fx.Int64(h) * sh) * fx.Int64(2)
            nbytes = fx.Int64(rows) * ss * fx.Int64(2)
            return buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=nbytes)

        q_rsrc = rsrc_for(Q, stride_q_batch, stride_q_head, stride_q_seq, head, sq)
        o_rsrc = rsrc_for(O, stride_o_batch, stride_o_head, stride_o_seq, head, sq)
        k_rsrc = rocdl.get_buffer_rsrc(rsrc_for(K, stride_k_batch, stride_k_head, stride_k_seq, head_k, sk))
        v_rsrc_raw = rocdl.get_buffer_rsrc(rsrc_for(V, stride_v_batch, stride_v_head, stride_v_seq, head_k, sk))
        q_row_stride = fx.Index(fx.Int32(stride_q_seq))
        k_row_stride_b = fx.Int32(stride_k_seq) * fx.Int32(2)
        v_row_stride_b = fx.Int32(stride_v_seq) * fx.Int32(2)
        o_row_stride = fx.Index(fx.Int32(stride_o_seq))

        q_start = q_tile * tr.BLOCK_M
        q_wave = q_start + wave * WAVE_M
        q_row = q_wave + lm

        # Q^T packs: k-step 2t+e reads d = 16t + 8*lh + 4e + [0, 4).
        q_packs = []
        for t in range_constexpr(tr.QK_STEPS // 2):
            off = q_row * q_row_stride + fx.Index(16 * t) + lh * 8
            v8 = buffer_ops.buffer_load(q_rsrc, off, vec_width=8, dtype=elem)
            q_packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
            q_packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())

        # K DMA. The tile offset rides in soffset, which the range check includes.
        k_voff = []
        k_m0 = []
        for i in range_constexpr(tr.K_DMA_ROUNDS):
            if const_expr(tr.K_PAD):
                line = fx.Index(i * tr.NUM_WAVES) + wave
                lines_per_row = tr.K_ROW_BYTES // 256
                r = line // lines_per_row
                c = line % lines_per_row
                k_voff.append(fx.Int32(r) * k_row_stride_b + fx.Int32(c * 256 + lane * 4))
                lds_line = r * tr.K_STRIDE + c * 256
            else:
                lds_line = fx.Index(i * tr.DMA_ROUND_BYTES) + wave * (WAVE * 4)
                p = lds_line + lane * 4
                r = p // tr.K_ROW_BYTES
                b = p % tr.K_ROW_BYTES
                c = (b // 16) ^ tr.k_swizzle(r)
                k_voff.append(fx.Int32(r) * k_row_stride_b + fx.Int32(c * 16 + b % 16))
            k_m0.append(fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                                     fx.Int32(lds_base + fx.Index(tr.LDS_K_BASE) + lds_line))))

        def k_dma(kv_start, buf):
            soff = fx.Int32(kv_start) * k_row_stride_b
            al, noal = scope_attrs(f'k{buf}')
            for i in range_constexpr(tr.K_DMA_ROUNDS):
                m0 = fx.Int64(k_m0[i] + fx.Int32(buf * tr.K_TILE_BYTES))
                op = rocdl.raw_ptr_buffer_load_lds(k_rsrc,
                                                   lds_ptr_at(m0),
                                                   fx.Int32(4),
                                                   k_voff[i],
                                                   soff,
                                                   fx.Int32(0),
                                                   fx.Int32(0))
                op = getattr(op, 'owner', op)
                op = getattr(op, 'operation', op)
                op.attributes['alias_scopes'] = al
                op.attributes['noalias_scopes'] = noal

        # V: each unit is 4 KV rows x 2 columns, transposed by v_perm.
        v_rows = []
        v_vt_off = []
        for u in range_constexpr(tr.V_UNITS_PER_THREAD):
            unit = tid + fx.Index(u * tr.BLOCK_SIZE)
            g = unit // (tr.D // 2)
            cp = unit % (tr.D // 2)
            for r in range_constexpr(4):
                v_rows.append(fx.Int32(g * 4 + fx.Index(r)) * v_row_stride_b + fx.Int32(cp * 4))
            for e in range_constexpr(2):
                v_vt_off.append(fx.Index(tr.LDS_VT_BASE) + ((cp * 2 + fx.Index(e)) * tr.VT_STRIDE + g * 4) * 2)

        def v_load(kv_start):
            soff = fx.Int32(kv_start) * v_row_stride_b
            return [fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, v_rsrc_raw, off, soff, 0)) for off in v_rows]

        def v_store_lds(regs, buf):
            sel_lo = fx.Int32(0x05040100)
            sel_hi = fx.Int32(0x07060302)
            for u in range_constexpr(tr.V_UNITS_PER_THREAD):
                w = regs[4 * u:4 * u + 4]
                d0 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_lo)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_lo))]
                d1 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_hi)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_hi))]
                for e, dw in ((0, d0), (1, d1)):
                    off = v_vt_off[2 * u + e] + buf * tr.VT_TILE_BYTES
                    al, noal = scope_attrs(f'vt{buf}')
                    llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                                 lds_view(off, fx.Int32, 8).llvm_ptr,
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
            c = (fx.Index(2 * t) + lh) ^ tr.k_swizzle(lm)
            k_rd_off.append(fx.Index(tr.LDS_K_BASE) + lm * tr.K_STRIDE + c * 16)
        vt_rd_off = fx.Index(tr.LDS_VT_BASE) + (lm * tr.VT_STRIDE + lh * 4) * 2

        def k_read(buf):
            return [lds_load(off + buf * tr.K_TILE_BYTES + n * 32 * tr.K_STRIDE, v8e(), 16, f'k{buf}')
                    for n in range(tr.NSUB) for off in k_rd_off]

        def qk(k8s):
            """S^T sub-tiles, one per 32 KV rows: [v16f32] * NSUB."""
            out = []
            nt = tr.QK_STEPS // 2
            for n in range_constexpr(tr.NSUB):
                s = fx.full(16, 0.0, fx.Float32)
                for t in range_constexpr(nt):
                    k8 = Vec(k8s[n * nt + t])
                    s = mfma(k8.shuffle(k8, [0, 1, 2, 3]).ir_value(), q_packs[2 * t], s)
                    s = mfma(k8.shuffle(k8, [4, 5, 6, 7]).ir_value(), q_packs[2 * t + 1], s)
                out.append(s)
            return out

        def v_read(buf):
            base = vt_rd_off + buf * tr.VT_TILE_BYTES
            return [[lds_load(base + (32 * dt * tr.VT_STRIDE + 8 * st) * 2, v4e(), 8, f'vt{buf}')
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
        if const_expr(tr.CAUSAL):
            delta = fx.Int32(sk) - fx.Int32(sq)
            kv_end_raw = fx.Int32(q_start + tr.BLOCK_M) + delta
            kv_end_raw = (kv_end_raw > fx.Int32(0)).select(kv_end_raw, fx.Int32(0))
            kv_end = fx.Index((kv_end_raw < fx.Int32(sk)).select(kv_end_raw, fx.Int32(sk)))
        else:
            delta = fx.Int32(0)
            kv_end = sk
        n_tiles = (kv_end + tr.BLOCK_N - 1) // tr.BLOCK_N

        def mask_limits():
            seq_lim = fx.Int32(sk) - fx.Int32(1)
            lim = seq_lim
            wave_lim = seq_lim
            if const_expr(tr.CAUSAL):
                diag = fx.Int32(q_row) + delta
                wave_diag = fx.Int32(q_wave) + delta
                lim = (diag < seq_lim).select(diag, seq_lim)
                wave_lim = (wave_diag < seq_lim).select(wave_diag, seq_lim)
            return lim, wave_lim

        def mask_scores(s_vals, kv_start):
            # Lane value r is KV row 32*(r//16) + 8*((r%16)//4) + 4*lh + r%4.
            kv0 = fx.Int32(kv_start) + kv_lane
            lim, _ = mask_limits()
            return [(kv0 + fx.Int32(32 * (r // 16) + 8 * ((r % 16) // 4) + r % 4) > lim).select(c_neg_inf, s_vals[r])
                    for r in range(tr.NS)]

        # Tiles [0, n_clean) need no mask in any wave of this block.
        lim0 = fx.Int32(sk) - fx.Int32(1)
        if const_expr(tr.CAUSAL):
            diag0 = fx.Int32(q_start) + delta
            lim0 = (diag0 < lim0).select(diag0, lim0)
        n_clean = fx.Index(((lim0 + fx.Int32(1)) > fx.Int32(0)).select(lim0 + fx.Int32(1), fx.Int32(0))) // tr.BLOCK_N
        n_loop = (n_tiles > fx.Index(1)).select(n_tiles - fx.Index(1), fx.Index(0))
        # Iteration j masks tile j+1, so the clean iterations are j < n_clean - 1.
        n_loop_clean = (n_clean > fx.Index(0)).select(n_clean - fx.Index(1), fx.Index(0))
        n_loop_clean = (n_loop_clean < n_loop).select(n_loop_clean, n_loop)

        def scale_and_mask(s_raw, kv_start, masked):
            # Scale once in fp32; the max and the exponent share that rounding.
            c_scale_v = fx.full(16, fx.Float32(c_scale), fx.Float32)
            scaled = [Vec(sr) * c_scale_v for sr in s_raw]
            s_vals = [scaled[r // 16][r % 16] for r in range(tr.NS)]
            if const_expr(masked):
                s_vals = mask_scores(s_vals, kv_start)
            return s_vals

        def wave_any(pred):
            return fx.Int64(rocdl.ballot(fx.Int64.ir_type, raw(pred))) != fx.Int64(0)

        def softmax_head(s_raw, m_old, kv_start, masked):
            """Scale, mask, and update the lazy running max; branch-free so it hides under PV."""
            s_vals = scale_and_mask(s_raw, kv_start, masked)
            local = s_vals[0]
            for r in range_constexpr(1, tr.NS):
                local = fx.maxnumf(local, s_vals[r])
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

        def softmax_tail(s_vals, m_new, corr, l_old, o_accs):
            p = [exp2(s_vals[r] - m_new) for r in range(tr.NS)]
            psum = p[0]
            for r in range_constexpr(1, tr.NS):
                psum = psum + p[r]
            l_new = l_old * corr + psum
            return [pack_p(p[4 * i:4 * i + 4]) for i in range(tr.PV_STEPS)], l_new, o_accs

        # Per tile: C0 mem, C1 QK(j+1)+exp(j), C2 mem, C3 PV(j)+max(j+1). Waves 4..7
        # trail by one barrier, so LDS producer->consumer needs two clusters:
        #   K(t):   DMA C2(t-3) -> kb[t%2], waited end C2(t-2), read C0(t-1)
        #   V^T(t): loaded C0(t-1), written C0(t) -> vt[t%2], read C2(t)
        # Unrolled by two so buffer indices (and alias scopes) are constants.
        n_kdma = tr.K_DMA_ROUNDS
        n_v = len(v_rows)
        group_b = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                               fx.Int32((wave >= fx.Index(tr.NUM_WAVES // 2)).select(1, 0))))

        def compute_begin():
            rocdl.s_setprio(1)

        def compute_end(name=None):
            rocdl.s_setprio(0)
            cluster_barrier(name)

        # Prologue: iteration -1.
        k_dma(fx.Index(0), 0)
        k_dma(fx.Index(tr.BLOCK_N), 1)
        rocdl.s_waitcnt(vmcnt=n_kdma)
        cluster_barrier()
        barrier_if(group_b)                       # open the skew
        # C0(-1)
        k_packs = k_read(0)
        v_regs = v_load(fx.Index(0))
        rocdl.s_waitcnt(lgkmcnt=0)
        cluster_barrier()
        # C1(-1)
        compute_begin()
        s_raw = qk(k_packs)
        compute_end()
        # C2(-1)
        k_dma(fx.Index(2 * tr.BLOCK_N), 0)
        rocdl.s_waitcnt(vmcnt=n_v + n_kdma)
        cluster_barrier()
        # C3(-1). A finite initial max keeps fully masked rows NaN-free.
        compute_begin()
        s0, m0, corr0 = softmax_head(s_raw, c_neg_max, fx.Index(0), True)
        compute_end()

        n_s = tr.NS

        def iteration(j, carried, b, masked):
            """Tile j; `b` = j % 2, known at trace time."""
            m_cur = carried[0]
            corr = carried[1]
            l_old = carried[2]
            s_cur = carried[3:3 + n_s]
            o_accs = list(carried[3 + n_s:3 + n_s + tr.D_TILES])
            v_regs = list(carried[3 + n_s + tr.D_TILES:])
            kv_next = (j + 1) * tr.BLOCK_N
            # C0(j)
            v_store_lds(v_regs, b)
            v_regs = v_load(kv_next)
            k_packs = k_read(1 - b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier(f'C0.{b}')
            # C1(j)
            compute_begin()
            o_accs = rescale_o(o_accs, corr)
            s_next = qk(k_packs)
            p_packs, l_new, o_accs = softmax_tail(s_cur, m_cur, corr, l_old, o_accs)
            compute_end(f'C1.{b}')
            # C2(j)
            k_dma(kv_next + 2 * tr.BLOCK_N, 1 - b)
            vts = v_read(b)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_waitcnt(vmcnt=n_v + n_kdma)
            cluster_barrier(f'C2.{b}')
            # C3(j)
            compute_begin()
            o_accs = pv(o_accs, p_packs, vts)
            s_vals, m_next, corr_next = softmax_head(s_next, m_cur, kv_next, masked)
            compute_end(f'C3.{b}')
            return [m_next, corr_next, l_new] + s_vals + o_accs + v_regs

        def finish(carried, b):
            """The last tile's PV."""
            o_last = list(carried[3 + n_s:3 + n_s + tr.D_TILES])
            # C0(n-1)
            v_store_lds(list(carried[3 + n_s + tr.D_TILES:]), b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            # C1(n-1)
            compute_begin()
            o_last = rescale_o(o_last, carried[1])
            p_packs, l_last, o_last = softmax_tail(carried[3:3 + n_s], carried[0], carried[1], carried[2], o_last)
            compute_end()
            # C2(n-1)
            vts = v_read(b)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            # C3(n-1)
            o_last = pv(o_last, p_packs, vts)
            return [carried[0], l_last] + o_last

        @flyc.jit
        def clean_tail(carried, j_odd, n_clean_loop):
            out = list(carried)
            if j_odd < n_clean_loop:
                out = iteration(j_odd, carried, 0, False)
            return out

        @flyc.jit
        def masked_iteration(j, carried):
            out = list(carried)
            if j % 2 == fx.Index(0):
                out = iteration(j, carried, 0, True)
            else:
                out = iteration(j, carried, 1, True)
            return out

        @flyc.jit
        def epilogue(carried, n_loop):
            out = [carried[0], carried[2]] + list(carried[3 + n_s:3 + n_s + tr.D_TILES])
            if n_loop % 2 == fx.Index(0):
                out = finish(carried, 0)
            else:
                out = finish(carried, 1)
            return out

        init = [m0, corr0, fx.Float32(0.0)] + s0 + [fx.full(16, 0.0, fx.Float32) for _ in range(tr.D_TILES)] + v_regs
        n_pairs = n_loop_clean // 2
        results = init
        for jp, carried in range(0, n_pairs, 1, init=init):
            mid = iteration(jp * 2, carried, 0, False)
            results = yield iteration(jp * 2 + 1, mid, 1, False)
        results = clean_tail(results, n_pairs * 2, n_loop_clean)
        results2 = results
        for j, carried in range(n_loop_clean, n_loop, 1, init=results):
            results2 = yield masked_iteration(j, carried)
        results = epilogue(results2, n_loop)
        barrier_if(fx.Int32(1) - group_b)         # close the skew

        rocdl.s_waitcnt(0)
        m_fin = results[0]
        l_fin = results[1]
        l_fin = l_fin + peer(l_fin)
        inv_l = fx.Float32(1.0) / l_fin
        inv_l = (l_fin > fx.Float32(0.0)).select(inv_l, fx.Float32(0.0))
        q_ok = q_row < sq
        for dt in range_constexpr(tr.D_TILES):
            o = Vec(results[2 + dt]) * fx.full(16, fx.Float32(inv_l), fx.Float32)
            for i in range_constexpr(4):
                vals = [o[4 * i + jj] for jj in range(4)]
                packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
                d = fx.Index(32 * dt + 8 * i) + lh * 4
                off = q_row * o_row_stride + d
                buffer_ops.buffer_store(packed.ir_value(), o_rsrc, off, mask=q_ok)
        # LSE: natural base, +inf for fully masked rows; a null L gets an empty range.
        lse = (m_fin + fx.Float32(fx.math.log2(l_fin))) * fx.Float32(LN2)
        lse = (l_fin > fx.Float32(0.0)).select(lse, fx.Float32(float('inf')))
        if const_expr(DEBUG_NTILES):
            lse = fx.Float32(fx.Int32(n_tiles))
        lse_addr = fx.Int64(fx.ptrtoint(LSE))
        lse_base = lse_addr + (fx.Int64(batch) * fx.Int64(num_head_q) + fx.Int64(head)) * fx.Int64(sq) * fx.Int64(4)
        lse_bytes = (lse_addr == fx.Int64(0)).select(fx.Int64(0), fx.Int64(sq) * fx.Int64(4))
        lse_rsrc = buffer_ops.create_buffer_resource_from_addr(lse_base, num_records_bytes=lse_bytes)
        buffer_ops.buffer_store(lse, lse_rsrc, q_row, mask=lh == fx.Index(0))

    @flyc.jit
    def launch_fmha_fwd_gfx942(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        batch: fx.Int32,
        seqlen_q: fx.Int32,
        seqlen_k: fx.Int32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
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
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = (fx.Index(seqlen_q) + tr.BLOCK_M - 1) // tr.BLOCK_M
        fmha_fwd_gfx942_kernel(Q,
                               K,
                               V,
                               O,
                               LSE,
                               seqlen_q,
                               seqlen_k,
                               num_head_q,
                               num_head_k,
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
                               value_attrs={'rocdl.waves_per_eu': waves_per_eu,
                                            'rocdl.flat_work_group_size': f'{tr.BLOCK_SIZE},{tr.BLOCK_SIZE}'}).launch(grid=(grid_x, fx.Index(num_head_q), fx.Index(batch)),
                 block=(tr.BLOCK_SIZE, 1, 1),
                 stream=stream)

    return launch_fmha_fwd_gfx942, tr
