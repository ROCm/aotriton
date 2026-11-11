# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Flash attention backward dQ (and dB) for gfx942 at head dims 160..512, FlyDSL.

4 waves, one per SIMD (512 VGPR+AGPR a lane), each owning WAVE_M query rows for
the whole KV loop. Per KV tile a wave runs three GEMMs, all in the forward's
transposed form (fmha_fwd_gfx942_wide.py):
    S^T  = K . Q^T         A = K rows (LDS),   B = Q^T (registers)
    dP^T = V . dO^T        A = V rows (LDS),   B = dO^T (registers)
    dQ^T += K^T . dS^T     A = K^T (LDS),      B = dS^T, the S^T accumulator repacked
so dS needs no shuffle and dQ^T lands in the forward's O^T layout. Two geometries,
one code path (accumulator value r is row 4G*(r//4) + 4*group + r%4, G = 64/WAVE_M):
    WAVE_M 32, 32x32x8 MFMA, 32-key tiles   head dims <= 256
    WAVE_M 16, 16x16x16 MFMA, 16-key tiles  384, 512: Q^T + dO^T + dQ^T is D/2 = 256 a lane
K is needed both ways and gfx942 has no LDS transpose read, so K and V come in
through VGPRs: K is written row-major and, a cluster later, as K^T (v_perm +
ds_write_b64, the forward's V^T path); V row-major. Single buffers (52 KB at
hd256 and hd512), two barrier-separated clusters per tile:
    A(j): write K^T(j); load K(j+1); S^T(j), dP^T(j)
    B(j): write V(j+1); load V(j+2); read K^T(j); dS(j); dQ^T += K^T(j) dS^T(j);
          write K(j+1)
Row-major rows are padded to an odd number of 16B chunks and the chunk index is
XORed with a row key measured conflict-free for the MFMA reads. Causal blocks run
q tiles g and T-1-g (PAIR_TILES).

The kernel runs at the 750 W cap (~1460 MHz at hd192); its MFMA utilization at
that clock is ~49%. Moving traffic to hide latency (two register sets, two K^T
buffers, spread writes) measured slower: it raised power and lowered the clock.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, ktrace, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
from fmha_fwd_gfx942 import principal_diagonal_bits
from philox import Philox

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dq_gfx942_wide.lds">'
LDS_SCOPES = ('k', 'v', 'kt')
LOG2E = 1.4426950408889634
WAVE = 64
# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1
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

    def __init__(self, head_dim, causal, dtype_str, padded_head=False, wave_m=None):
        # Query rows per wave, the N of the MFMA: 32 on 32x32x8, or 16 on 16x16x16,
        # which halves the loop-invariant Q^T + dO^T + dQ^T to fit hd384/512.
        wave_m = (32 if head_dim <= 256 else 16) if wave_m is None else wave_m
        assert wave_m in (16, 32), wave_m
        assert head_dim % 32 == 0 and 64 <= head_dim <= (256 if wave_m == 32 else 512), head_dim
        self.D = head_dim
        self.WAVE_M = wave_m
        self.G = WAVE // wave_m                 # lane groups: lane = group * WAVE_M + row
        self.PADDED = padded_head
        # Columns below PAD_FLOOR are real in every call this build serves.
        self.PAD_FLOOR = 0 if padded_head else head_dim
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NUM_WAVES = 4
        self.BLOCK_SIZE = self.NUM_WAVES * WAVE
        self.BLOCK_M = self.NUM_WAVES * wave_m
        # One S^T MFMA tile per KV tile; accumulator value r is key / dQ^T row
        # 4G*(r//4) + 4*group + r%4.
        self.BLOCK_N = wave_m
        self.NS = wave_m * wave_m // WAVE       # S^T values per lane
        self.QK_STEPS = head_dim // (4 * self.G)    # MFMA k-steps of S^T and dP^T
        self.D_TILES = head_dim // wave_m       # wave_m-row tiles of dQ^T
        self.PV_STEPS = self.BLOCK_N // (4 * self.G)    # MFMA k-steps of dQ^T per tile
        # Row-major K and V: an odd number of 16B chunks a row.
        chunks = head_dim // 8
        self.RM_PAD = 1 - chunks % 2            # 16B chunks of row padding
        for k, v in DEV_OVERRIDES.items():
            if k == 'RM_PAD':
                self.RM_PAD = v
        self.RM_STRIDE = (chunks + self.RM_PAD) * 16
        self.RM_TILE_BYTES = self.BLOCK_N * self.RM_STRIDE
        # K^T: [D][BLOCK_N + 4]; the 8B pad spreads 16 lanes over all banks.
        self.KT_STRIDE = self.BLOCK_N + 4
        self.KT_TILE_BYTES = head_dim * self.KT_STRIDE * 2
        # Global load unit: 4 kv rows x 8 columns, four dwordx4 a tensor; spare
        # threads redo a unit (same bytes to the same place).
        self.UNITS = (self.BLOCK_N // 4) * (head_dim // 8)
        assert self.UNITS <= self.BLOCK_SIZE, self.UNITS
        self.LDS_K_BASE = 0
        self.LDS_V_BASE = self.RM_TILE_BYTES
        self.LDS_KT_BASE = 2 * self.RM_TILE_BYTES
        self.VALU_PER_MFMA = 3                  # sched_group_barrier slot size
        self.SCHED = True                       # interleave LDS/VMEM/VALU with MFMAs
        self.LEAD_A = 4                         # LDS reads issued ahead of the S^T/dP^T MFMAs
        self.LEAD_B = 4                         # LDS reads issued ahead of the dQ^T MFMAs
        self.KT_EARLY = 1                       # read all of K^T(j) before dS(j)
        self.PAIR_TILES = causal
        # Row-major chunk XOR key (row // SW_DIV) % SW_MOD; measured conflict-free
        # (SQ_LDS_BANK_CONFLICT) for the MFMA reads with the 1-chunk row pad.
        self.SW_DIV, self.SW_MOD = (8, 2) if wave_m == 32 else (4, 4)
        self.ABLATE = 0                         # dev timing only, wrong results: bit 0 A reads, 1 dS VALU, 2 B reads, 3 barriers, 4 LDS writes, 5 K/V loads
        for k, v in DEV_OVERRIDES.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        assert self.SW_MOD <= self.G, 'the read key must stay inside the lane group bits'
        self.LDS_BYTES = self.LDS_KT_BASE + self.KT_TILE_BYTES
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        self.GRID_BLOCK_M = self.BLOCK_M * (2 if self.PAIR_TILES else 1)
        # The JIT cache keys on scalar closure values; the kernel references this.
        self.CACHE_TAG = (f'DQW{head_dim}_m{wave_m}_c{int(causal)}_{dtype_str}_p{int(padded_head)}'
                          f'_a{self.VALU_PER_MFMA}_s{int(self.SCHED)}_q{int(self.PAIR_TILES)}'
                          f'_la{self.LEAD_A}_lb{self.LEAD_B}_x{self.ABLATE}_sw{self.SW_DIV}.{self.SW_MOD}_rp{self.RM_PAD}_ke{self.KT_EARLY}')


def knobs(head_dim, causal):
    """Default build of one functional, as a JSON-able dict; `block_m` (rows per
    grid block) and `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    wave_m = 32 if head_dim <= 256 else 16
    return dict(block_m=4 * wave_m * (2 if causal else 1),
                wave_m=wave_m,
                num_waves=4,
                block_n=wave_m,
                waves_per_eu=1,
                GRID_AXIS_ORDER=GRID_AXIS_TILE_FASTEST)


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
    """The gfx1201 dQ kernel's ABI and `_launch` interface, for wide heads."""
    host_causal_type = (1 if causal else 0) if causal_type is None else causal_type
    if host_causal_type not in (0, 1, 2, 3):
        raise ValueError(f'causal_type must be 0-3, got {host_causal_type}')
    causal_type = 0 if host_causal_type == 0 else 3
    if bias and causal_type:
        raise ValueError('bias with a causal mask is not served (aot/_common.py flash_disabled)')
    tr = Traits(head_dim, causal_type != 0, dtype_str, padded_head, wave_m)
    if num_waves != 4 or block_n not in (None, tr.BLOCK_N):
        raise ValueError(f'this body runs 4 waves and {tr.BLOCK_N}-key tiles, got {num_waves}, {block_n}')
    if grid_axis_order is None:
        grid_axis_order = GRID_AXIS_TILE_FASTEST
    tile_fastest = grid_axis_order == GRID_AXIS_TILE_FASTEST
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}'
    elem = fx.BFloat16 if dtype_str == 'bf16' else fx.Float16

    # MLIR types need a live context, so they are built at trace time.
    def v4e():
        return Vec.make_type(4, elem)

    def v8e():
        return Vec.make_type(8, elem)

    def vacc():
        return Vec.make_type(tr.NS, fx.Float32)

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, tr.LDS_BYTES, 16]

    def mfma(a, b, c):
        if const_expr(tr.WAVE_M == 16):
            if const_expr(tr.DTYPE == 'bf16'):
                return rocdl.mfma_f32_16x16x16bf16_1k(vacc(), [Vec(a).bitcast(fx.Int16), Vec(b).bitcast(fx.Int16), c])
            return rocdl.mfma_f32_16x16x16f16(vacc(), [a, b, c])
        if const_expr(tr.DTYPE == 'bf16'):
            return rocdl.mfma_f32_32x32x8bf16_1k(vacc(), [Vec(a).bitcast(fx.Int16), Vec(b).bitcast(fx.Int16), c])
        return rocdl.mfma_f32_32x32x8f16(vacc(), [a, b, c])

    def exp2(x):
        return fx.Float32(rocdl.exp2(T.f32, raw(x)))

    def pack4(vals):
        """Four f32 -> one operand; bf16 rounds to nearest, ties away (lessons #16)."""
        if const_expr(tr.DTYPE == 'bf16'):
            words = []
            for h in range_constexpr(2):
                rnd = [fx.Float32(x).bitcast(fx.Int32) + fx.Int32(0x8000) for x in (vals[2 * h], vals[2 * h + 1])]
                words.append(fx.Int32(rocdl.perm_b32(rnd[1], rnd[0], fx.Int32(0x07060302))))
            return Vec.from_elements(words, fx.Int32).bitcast(elem).ir_value()
        return Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem).ir_value()

    def scope_attrs(name):
        def arr(names):
            return ir.Attribute.parse('[' + ', '.join(
                f'#llvm.alias_scope<id = "{n}", domain = {_LDS_DOMAIN}>' for n in names) + ']')
        return arr([name]), arr([n for n in LDS_SCOPES if n != name])

    def cluster_barrier(name=None):
        """End of a cluster; under FLYDSL_KTRACE_ENABLE=1, marks its end and the barrier release."""
        rocdl.sched_barrier(0)
        if name is not None:
            ktrace.mark(name)
        if const_expr(not tr.ABLATE & 8):
            rocdl.s_barrier()
        if name is not None:
            ktrace.mark(name + '>')
        rocdl.sched_barrier(0)

    def sgb(mask, n):
        if const_expr(n > 0):
            rocdl.sched_group_barrier(mask, n, 0)

    def sched(n_mfma, n_rd, n_wr, n_vmem, n_valu, lead=0):
        """Spread LDS reads (`lead` of them first), LDS writes, VMEM and VALU over the MFMAs."""
        if const_expr(not tr.SCHED):
            return
        sgb(SG_DS_READ, lead)
        n_rd -= lead
        for i in range_constexpr(n_mfma):
            sgb(SG_MFMA, 1)
            sgb(SG_DS_READ, (i + 1) * n_rd // n_mfma - i * n_rd // n_mfma)
            sgb(SG_DS_WRITE, (i + 1) * n_wr // n_mfma - i * n_wr // n_mfma)
            sgb(SG_VMEM, (i + 1) * n_vmem // n_mfma - i * n_vmem // n_mfma)
            sgb(SG_VALU, (i + 1) * n_valu // n_mfma - i * n_valu // n_mfma)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
    # The kernel ABI is bwd_dq_kernel's (gfx1201), so aot/flyc_bwd_dq.py can serve both.
    def fmha_bwd_dq_gfx942_wide_kernel(
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
        lm = lane % tr.WAVE_M
        lh = lane // tr.WAVE_M                  # lane group
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

        def lds_view(off, e, align):
            return fx.recast_iter(fx.PointerType.get(e.ir_type, fx.AddressSpace.Shared, align),
                                  lds_p + fx.Int64(off))

        def lds_load(off, ty, align, scope):
            al, noal = scope_attrs(scope)
            return llvm.LoadOp(ty,
                               lds_view(off, elem, align).llvm_ptr,
                               alignment=align,
                               alias_scopes=al,
                               noalias_scopes=noal).result

        def lds_store(val, off, align, scope):
            al, noal = scope_attrs(scope)
            llvm.StoreOp(raw(val),
                         lds_view(off, fx.Int32, align).llvm_ptr,
                         alignment=align,
                         alias_scopes=al,
                         noalias_scopes=noal)

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

        # LSE and Delta share the logsumexp layout (fmha.lse_row_addressing).
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)
        lse_row0, lse_pitch = fmha.lse_row_addressing(varlen_bits,
                                                      fx.Index(q_batch),
                                                      head,
                                                      num_head_q,
                                                      lse_tokens,
                                                      fx.Index(q_row_off))

        def row_rsrc(ptr):
            base = fx.Int64(fx.ptrtoint(ptr)) + fx.Int64(lse_row0) * fx.Int64(4)
            return buffer_ops.create_buffer_resource_from_addr(base,
                                                               num_records_bytes=fx.Int64(sq * lse_pitch) * fx.Int64(4))

        lse_rsrc = row_rsrc(LSE)
        delta_rsrc = row_rsrc(Delta)

        # Load unit: 4 KV rows x 8 columns, row group fastest.
        u = tid % tr.UNITS
        u_g = u % (tr.BLOCK_N // 4)
        u_c8 = u // (tr.BLOCK_N // 4)
        k_voff = fx.Int32(u_g * 4) * k_row_stride_b + fx.Int32(u_c8 * 16)
        v_voff = fx.Int32(u_g * 4) * v_row_stride_b + fx.Int32(u_c8 * 16)
        # Row-major write: rows 4g..4g+3 share swizzle key (4g / 8) % 4.
        rm_wr = fx.Index(u_g * 4 * tr.RM_STRIDE) + (u_c8 ^ (((u_g * 4) // tr.SW_DIV) % tr.SW_MOD)) * 16
        kt_wr = fx.Index(tr.LDS_KT_BASE) + (u_c8 * 8 * tr.KT_STRIDE + u_g * 4) * 2

        def unit_mask(hdim):
            """Dword masks keeping this unit's columns below `hdim` (padded heads)."""
            c0 = fx.Int32(u_c8 * 8)
            words = []
            for w in range_constexpr(4):
                lo = (c0 + fx.Int32(2 * w) < fx.Int32(hdim)).select(fx.Int32(0xFFFF), fx.Int32(0))
                hi = (c0 + fx.Int32(2 * w + 1) < fx.Int32(hdim)).select(fx.Int32(-65536), fx.Int32(0))
                words.append(lo | hi)
            return words

        if const_expr(tr.PADDED):
            k_umask = unit_mask(hdim_qk)
            v_umask = unit_mask(hdim_vo)

        def unit_load(rsrc, voff, row_stride_b, kv_start):
            if const_expr(tr.ABLATE & 32):
                return [fx.Int32(kv_start) + fx.Int32(i) for i in range(16)]
            out = []
            for r in range_constexpr(4):
                soff = (fx.Int32(kv_start) + fx.Int32(r)) * row_stride_b
                w4 = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(4, fx.Int32), rsrc, voff, soff, 0))
                out.extend(w4[i] for i in range(4))
            return out

        def unit_clean(regs, umask):
            if const_expr(not tr.PADDED):
                return regs
            return [fx.Int32(regs[i]) & umask[i % 4] for i in range(16)]

        def rm_store(regs, base, scope):
            for r in range_constexpr(0 if tr.ABLATE & 16 else 4):
                lds_store(Vec.from_elements(regs[4 * r:4 * r + 4], fx.Int32), rm_wr + base + r * tr.RM_STRIDE, 16, scope)

        def kt_store(regs):
            """v_perm turns the unit into 8 K^T rows of 4 keys."""
            sel = (fx.Int32(0x05040100), fx.Int32(0x07060302))
            for d in range_constexpr(0 if tr.ABLATE & 16 else 8):
                w = [regs[4 * r + d // 2] for r in range(4)]
                dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel[d % 2])), fx.Int32(rocdl.perm_b32(w[3], w[2], sel[d % 2]))]
                lds_store(Vec.from_elements(dw, fx.Int32), kt_wr + d * tr.KT_STRIDE * 2, 8, 'kt')

        # MFMA A-operand reads of row-major K / V: row lm, chunk 2t + lh, swizzled.
        sw = (lm // tr.SW_DIV) % tr.SW_MOD
        rm_rd = lm * tr.RM_STRIDE + (lh ^ sw) * 16
        kt_rd = fx.Index(tr.LDS_KT_BASE) + (lm * tr.KT_STRIDE + lh * 4) * 2

        def rm_read(base, scope):
            return [lds_load(rm_rd + base + t * tr.G * 16, v8e(), 16, scope) for t in range(tr.QK_STEPS // 2)]

        def kt_read():
            return [[lds_load(kt_rd + (tr.WAVE_M * dt * tr.KT_STRIDE + 4 * tr.G * st) * 2, v4e(), 8, 'kt')
                     for st in range(tr.PV_STEPS)] for dt in range(tr.D_TILES)]

        def gemm_t(a8s, b_packs):
            """S^T-shaped product of a row-major tile and a register operand."""
            s = fx.full(tr.NS, 0.0, fx.Float32)
            for t in range_constexpr(tr.QK_STEPS // 2):
                a8 = Vec(a8s[t])
                s = mfma(a8.shuffle(a8, [0, 1, 2, 3]).ir_value(), b_packs[2 * t], s)
                s = mfma(a8.shuffle(a8, [4, 5, 6, 7]).ir_value(), b_packs[2 * t + 1], s)
            return s

        def gemm_dq(dq_accs, ds_packs, kts):
            for st in range_constexpr(tr.PV_STEPS):
                for dt in range_constexpr(tr.D_TILES):
                    dq_accs[dt] = mfma(kts[dt][st], ds_packs[st], dq_accs[dt])
            return dq_accs

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

        if const_expr(dropout):
            # The Triton kernels' stream, element for element (philox.py, grid_plane).
            rng = Philox.for_arch('gfx942')
            ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            ph_plane = z_i32 * fx.Int32(num_head_q) + fx.Int32(head)
            ph_base, ph_stride = rng.grid_plane(ph_off, ph_plane, max_seqlen_q, max_seqlen_k)

        if const_expr(bias):
            # (B, H, Sq, Sk), Sk contiguous; rows offset by Q's varlen offset, columns by K's.
            # Both offsets sit in the base, so the range covers the slab (as gfx1201).
            b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head) * stride_b_head
                       + fx.Int64(q_row_off) * stride_b_seq_q + fx.Int64(k_row_off))
            # Odd-pitch rows start mid-dword: align the base down, funnel-shift per lane.
            b_addr = fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2)
            b_mis = fx.Int32(b_addr & fx.Int64(3))
            b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                b_addr - fx.Int64(b_mis),
                num_records_bytes=b_span * fx.Int64(2) + fx.Int64(b_mis)))
            # dB: its own strides; an all-zero triple means the caller wants no dB.
            db_elems = (fx.Int64(q_batch) * stride_db_batch + fx.Int64(head) * stride_db_head
                        + fx.Int64(q_row_off) * stride_db_seq_q + fx.Int64(k_row_off))
            db_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_db_seq_q + fx.Int64(sk_i32), fx.Int64(0))
            db_addr = fx.Int64(fx.ptrtoint(DB)) + db_elems * fx.Int64(2)
            db_rsrc = buffer_ops.create_buffer_resource_from_addr(db_addr, num_records_bytes=db_span * fx.Int64(2))
            # Clean tiles store 8B groups when every group starts 8B-aligned.
            db_fast = (((db_addr & fx.Int64(7)) == fx.Int64(0))
                       & ((fx.Int64(stride_db_seq_q) & fx.Int64(3)) == fx.Int64(0)))
            store_db = ((fx.Int64(stride_db_batch) != fx.Int64(0))
                        | (fx.Int64(stride_db_head) != fx.Int64(0))
                        | (fx.Int64(stride_db_seq_q) != fx.Int64(0)))

        for q_pass in range_constexpr(len(q_tiles)):
            q_tile = q_tiles[q_pass]
            q_start = q_tile * tr.BLOCK_M
            q_row = q_start + wave * tr.WAVE_M + lm
            q_ok = q_row < sq

            def col_mask(t, hdim):
                """Dword masks keeping this lane's columns 8Gt + 8lh + [0, 8) below hdim."""
                c0 = fx.Int32(8 * tr.G * t) + fx.Int32(lh * 8)
                words = []
                for w in range_constexpr(4):
                    lo = (c0 + fx.Int32(2 * w) < fx.Int32(hdim)).select(fx.Int32(0xFFFF), fx.Int32(0))
                    hi = (c0 + fx.Int32(2 * w + 1) < fx.Int32(hdim)).select(fx.Int32(-65536), fx.Int32(0))
                    words.append(lo | hi)
                return Vec.from_elements(words, fx.Int32)

            def preload(rsrc, row_stride, hdim):
                """B^T packs: k-step 2t+e reads d = 8Gt + 8*lh + 4e + [0, 4)."""
                packs = []
                for t in range_constexpr(tr.QK_STEPS // 2):
                    off = q_row * row_stride + fx.Index(8 * tr.G * t) + lh * 8
                    v8 = buffer_ops.buffer_load(rsrc, off, vec_width=8, dtype=elem)
                    if const_expr(8 * tr.G * t + 8 * tr.G > tr.PAD_FLOOR):
                        # Slack columns may hold NaN, and 0 * NaN is NaN.
                        v8 = Vec(v8).bitcast(fx.Int32) & col_mask(t, hdim)
                        v8 = Vec(v8).bitcast(elem).ir_value()
                    packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
                    packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())
                return packs

            q_packs = preload(q_rsrc, q_row_stride, hdim_qk)
            do_packs = preload(do_rsrc, do_row_stride, hdim_vo)
            lse = fx.Float32(buffer_ops.buffer_load(lse_rsrc, q_row * lse_pitch, vec_width=1, dtype=fx.Float32))
            delta = fx.Float32(buffer_ops.buffer_load(delta_rsrc, q_row * lse_pitch, vec_width=1, dtype=fx.Float32))
            # Natural-base LSE; +inf (dead rows, rows past sq) makes P exactly 0.
            lse2 = q_ok.select(lse * fx.Float32(LOG2E), fx.Float32(float('inf')))
            neg_lse2 = fx.Float32(0.0) - lse2

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
                # Lane value r is KV row 4G*(r//4) + 4*lh + r%4.
                rel0 = fx.Int32(kv_start) + kv_lane - band_lo
                out = []
                for r in range_constexpr(tr.NS):
                    rel = rel0 + fx.Int32(4 * tr.G * (r // 4) + r % 4)
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
            r_lead = smin(lead, n_tiles)
            r_clean = smax(smin(clean_end, n_tiles), r_lead)

            if const_expr(bias):
                b_byte = b_mis + (fx.Int32(q_row) * fx.Int32(stride_b_seq_q) + kv_lane) * fx.Int32(2)
                b_lane = b_byte & fx.Int32(-4)
                b_shift = (b_byte & fx.Int32(3)) * fx.Int32(8)
                db_lane = fx.Int32(q_row) * fx.Int32(stride_db_seq_q) + kv_lane
                db_ok = q_ok & store_db

            def bias_load(kv_start, exact):
                """This lane's bias for one KV tile in S^T order; `exact` reads 2 bytes at a time
                so the slab's last element survives the dword range check."""
                if const_expr(not bias):
                    return None
                soff = fx.Int32(kv_start) * fx.Int32(2)
                out = []
                for g in range_constexpr(tr.NS // 4):
                    if const_expr(exact):
                        off = b_byte + fx.Int32(4 * tr.G * g * 2)
                        hs = [rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, off + fx.Int32(2 * e), soff, 0) for e in range(4)]
                        out.append(Vec.from_elements([fx.Int16(h) for h in hs], fx.Int16).bitcast(elem).ir_value())
                        continue
                    off = b_lane + fx.Int32(4 * tr.G * g * 2)
                    dw = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(3, fx.Int32), b_rsrc, off, soff, 0))
                    words = [fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.fshr.i32',
                                                          [raw(dw[w + 1]), raw(dw[w]), raw(b_shift)], [], []))
                             for w in range(2)]
                    out.append(Vec.from_elements(words, fx.Int32).bitcast(elem).ir_value())
                return out

            def dropout_keep(kv_start):
                """Keep flags in S^T order: each 4-key group is one Philox call."""
                keep = []
                for g in range_constexpr(tr.NS // 4):
                    col = fx.Int32(kv_start) + kv_lane + fx.Int32(4 * tr.G * g)
                    vals = rng.u32(ph_seed, rng.grid_offset(ph_base, ph_stride, fx.Int32(q_row), col))
                    keep.extend(fx.Int32(v) > fx.Int32(idropout_p) for v in vals)
                return keep

            def db_store_scalar(packs, base, kv_start):
                """One element at a time: a dB row may start mid-dword and end mid-group."""
                for r in range_constexpr(tr.NS):
                    rel = 4 * tr.G * (r // 4) + r % 4
                    col = fx.Int32(kv_start) + kv_lane + fx.Int32(rel)
                    buffer_ops.buffer_store(Vec(packs[r // 4])[r % 4],
                                            db_rsrc,
                                            base + fx.Int32(rel),
                                            mask=db_ok & (col < sk_i32))

            def ds_tile(s_acc, dp_acc, kv_start, masked, bias_vals):
                """dS^T packs of one tile; stores dB = dS on bias builds."""
                if const_expr(bias):
                    s_vals = [Vec(s_acc)[r] * c_scale for r in range(tr.NS)]
                    # Upcast first: log2(e) in the input dtype would differ from the forward's.
                    s_vals = [s_vals[r] + fx.Float32(Vec(bias_vals[r // 4])[r % 4].to(fx.Float32)) * fx.Float32(LOG2E)
                              - lse2 for r in range(tr.NS)]
                else:
                    # One fma: the exponent of S*scale - lse2, rounded once.
                    s_vals = [fx.Float32(fmath.fma(Vec(s_acc)[r], c_scale, neg_lse2)) for r in range(tr.NS)]
                if const_expr(masked):
                    # After the scale, as lessons #7 requires; -inf - lse2 is -inf.
                    s_vals = mask_scores(s_vals, kv_start)
                p = [exp2(s_vals[r]) for r in range(tr.NS)]
                dp = [fx.Float32(Vec(dp_acc)[r]) for r in range(tr.NS)]
                if const_expr(dropout):
                    # The mask and 1/(1-p) hit dP; dS uses the undropped P (lessons #9).
                    keep = dropout_keep(kv_start)
                    dp = [keep[r].select(dp[r] * fx.Float32(dropout_scale), fx.Float32(0.0)) for r in range(tr.NS)]
                ds = [p[r] * (dp[r] - delta) for r in range(tr.NS)]
                packs = [pack4(ds[4 * i:4 * i + 4]) for i in range(tr.PV_STEPS)]
                if const_expr(bias):
                    # dB = dS as rounded for the MFMA. Group i is 4 contiguous keys.
                    base = db_lane + fx.Int32(kv_start)
                    if const_expr(masked):
                        db_store_scalar(packs, base, kv_start)
                    else:
                        if db_fast:
                            for i in range_constexpr(tr.PV_STEPS):
                                buffer_ops.buffer_store(packs[i], db_rsrc, base + fx.Int32(4 * tr.G * i), mask=db_ok)
                        else:
                            db_store_scalar(packs, base, kv_start)
                return packs

            dt_n = tr.D_TILES

            def iteration(j, carried, masked):
                """A(j), B(j) of KV tile j (counted from kv_base). Carries K(j) and V(j+1) in
                registers: K(j+1) loads in A(j) after K^T(j) is out, V(j+2) in B(j) after V(j+1)
                is out. One set each, so the loop carries no register rotation."""
                dq_accs = list(carried[:dt_n])
                k_regs = list(carried[dt_n:dt_n + 16])
                v_regs = list(carried[dt_n + 16:])
                kv_cur = kv_base + j * tr.BLOCK_N
                # A(j): K^T(j), then K(j+1) into the same registers; bias(j) first so its
                # wait does not drain them.
                kt_store(k_regs)
                b_vals = bias_load(kv_cur, masked)
                rocdl.sched_barrier(0)
                k_regs = unit_load(k_rsrc, k_voff, k_row_stride_b, kv_cur + tr.BLOCK_N)
                if const_expr(tr.ABLATE & 1):
                    fake = [Vec(q_packs[2 * t]).shuffle(Vec(do_packs[2 * t]), list(range(8))).ir_value() for t in range(tr.QK_STEPS // 2)]
                    s_acc = gemm_t(fake, q_packs)
                    dp_acc = gemm_t(fake, do_packs)
                else:
                    s_acc = gemm_t(rm_read(tr.LDS_K_BASE, 'k'), q_packs)
                    dp_acc = gemm_t(rm_read(tr.LDS_V_BASE, 'v'), do_packs)
                sched(tr.QK_STEPS * 2, tr.QK_STEPS, 0, 0, 0, lead=tr.LEAD_A)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier('A')
                # B(j): V(j+1) row-major, then V(j+2) into its registers; dS(j);
                # dQ^T += K^T(j) dS^T(j); K(j+1) row-major last.
                rm_store(unit_clean(v_regs, v_umask if tr.PADDED else None), tr.LDS_V_BASE, 'v')
                rocdl.sched_barrier(0)
                v_regs = unit_load(v_rsrc, v_voff, v_row_stride_b, kv_cur + 2 * tr.BLOCK_N)
                if const_expr(tr.ABLATE & 4):
                    kts = [[q_packs[(dt * 4 + st) % len(q_packs)] for st in range(tr.PV_STEPS)] for dt in range(tr.D_TILES)]
                else:
                    kts = kt_read()
                if const_expr(tr.KT_EARLY):
                    # The whole K^T tile lands under the dS VALU.
                    rocdl.sched_barrier(0)
                if const_expr(tr.ABLATE & 2):
                    ds_packs = [pack4([Vec(s_acc)[4 * i + jj] + Vec(dp_acc)[4 * i + jj] for jj in range(4)]) for i in range(tr.PV_STEPS)]
                else:
                    ds_packs = ds_tile(s_acc, dp_acc, kv_cur, masked, b_vals)
                dq_accs = gemm_dq(dq_accs, ds_packs, kts)
                if const_expr(not tr.KT_EARLY):
                    sched(dt_n * tr.PV_STEPS, dt_n * tr.PV_STEPS // 2, 0, 0, 0, lead=tr.LEAD_B)
                rocdl.sched_barrier(0)
                k_regs = unit_clean(k_regs, k_umask if tr.PADDED else None)
                rm_store(k_regs, tr.LDS_K_BASE, 'k')
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier('B')
                return dq_accs + k_regs + v_regs

            # Prologue: K, V (tile 0) row-major; K^T(0) is written by A(0); V(1) in flight.
            k_regs = unit_load(k_rsrc, k_voff, k_row_stride_b, kv_base)
            v_regs = unit_load(v_rsrc, v_voff, v_row_stride_b, kv_base)
            k_regs = unit_clean(k_regs, k_umask if tr.PADDED else None)
            rm_store(k_regs, tr.LDS_K_BASE, 'k')
            rm_store(unit_clean(v_regs, v_umask if tr.PADDED else None), tr.LDS_V_BASE, 'v')
            rocdl.sched_barrier(0)
            v_regs = unit_load(v_rsrc, v_voff, v_row_stride_b, kv_base + tr.BLOCK_N)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()

            init = [fx.full(tr.NS, 0.0, fx.Float32) for _ in range(dt_n)] + k_regs + v_regs
            res0 = init
            for j, carried in range(fx.Index(0), r_lead, 1, init=init):
                res0 = yield iteration(j, carried, True)
            res1 = res0
            for j, carried in range(r_lead, r_clean, 1, init=res0):
                res1 = yield iteration(j, carried, False)
            res2 = res1
            for j, carried in range(r_clean, n_tiles, 1, init=res1):
                res2 = yield iteration(j, carried, True)

            # dQ = sm_scale * dQ^T, rounded to nearest even.
            sc = fx.full(tr.NS, fx.Float32(sm_scale), fx.Float32)
            for dt in range_constexpr(dt_n):
                o = Vec(res2[dt]) * sc
                for i in range_constexpr(tr.NS // 4):
                    vals = [o[4 * i + jj] for jj in range(4)]
                    packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
                    d0 = tr.WAVE_M * dt + 4 * tr.G * i
                    d = fx.Index(d0) + lh * 4
                    ok = q_ok
                    if const_expr(d0 + 4 * tr.G > tr.PAD_FLOOR):
                        ok = ok & (fx.Int32(d) < row_qk)
                    buffer_ops.buffer_store(packed.ir_value(), dq_rsrc, q_row * dq_row_stride + d, mask=ok)

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
        grid_y = (fx.Index(max_seqlen_q) + tr.GRID_BLOCK_M - 1) // tr.GRID_BLOCK_M
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_bwd_dq_gfx942_wide_kernel(Q,
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
        ptrs, meta, st = abi.prep_tensors([("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)], q_heads=("DO", "DQ"))
        if tr.PADDED:
            if not (0 < max(meta[2], meta[3]) <= tr.D):
                raise ValueError(f'head dims {meta[2:]} do not fit a padded {tr.D} build')
        elif meta[2] != tr.D or meta[3] != tr.D:
            raise ValueError(f'head dims {meta[2:]} on an unpadded {tr.D} build; pass padded_head=True')
        lp = abi.row_tensor_arg(lse, "logsumexp", meta[0], seqlen_q, varlen)
        dp = abi.row_tensor_arg(delta, "delta", meta[0], seqlen_q, varlen)
        wl, wr = abi.resolve_window(causal_type, host_causal_type, window, seqlen_q, seqlen_k)
        vb, sq0, sq1, sk0, sk1, mq, mk = abi.varlen_args(False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        vb = principal_diagonal_bits(vb, principal_diagonal)
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
        bp, dbp, bs, dbs = abi.bias_args(1 if bias_on else 0, True, bias, dbias, Q)
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

    bias_on = bias

    def _launch(*args, **kwargs):
        packed, _hold, stream = _args(*args, **kwargs)
        abi.run_compiled(compiled, launch_fmha_bwd_dq_gfx942_wide, *packed, fx.Stream(stream))

    _launch.traits = tr
    _launch.block_m = tr.GRID_BLOCK_M
    _launch.varlen_bits = abi.varlen_bits
    _launch.varlen_compact = abi.varlen_compact
    _launch.varlen_padded = abi.varlen_padded
    return _launch
