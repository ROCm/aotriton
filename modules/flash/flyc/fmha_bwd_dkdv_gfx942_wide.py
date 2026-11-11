# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dK/dV backward for gfx942 at head dims 160..256, FlyDSL.

K and V stay in registers; Q and dO stream past through LDS in 16-row tiles.
v_mfma_f32_16x16x16, four waves (one a SIMD), each owning 16 * KV_SUBS keys:

    S[q][kv]    = Q . K^T      A = Q  (LDS, row-major)   B = K (registers)
    dP[q][kv]   = dO . V^T     A = dO (LDS, row-major)   B = V (registers)
    dV^T[d][kv] += dO^T . P    A = dO^T (LDS, d-major)   B = P (S's output layout)
    dK^T[d][kv] += Q^T . dS    A = Q^T  (LDS, d-major)   B = dS

gfx942 has no LDS transpose read, so each Q/dO tile lives in LDS twice: RM
(row-major, filled by buffer_load...lds DMA, double-buffered) and T (q-major
per d, filled through VGPRs with v_perm, single-buffered). Each tile is two
barrier-separated clusters:
    X(j): S(j+1), dP(j+1) from RM; P(j), dS(j) on the VALU; T <- tile j
    Y(j): dV, dK += tile j from T; DMA RM(j+3); load tile j+1 for T
At head dim 256 and KV_SUBS 2 the accumulators take 256 registers and K, V 128.
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
from philox import Philox

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942_wide.lds">'
LDS_SCOPES = ('rm0', 'rm1', 't0', 't1', 's0', 's1')
LOG2E = 1.4426950408889634
WAVE = 64
NUM_WAVES = 4
BLOCK_M = 16            # q rows per tile: the N of the 16x16x16 MFMA
# T layout: row (dt, g) holds d = 16dt + [0, 16), q = 4g + [0, 4), 8B per d; rows are
# 144B apart (16B pad: T writes hit 8 bank quads), or 128B with 16B chunks XORed by g.
T_G_PAD = 144
SIDE_BYTES = 512        # LSE then Delta of one tile, 256B per DMA issue
OOB_VOFF = 0x7FFFFFF0   # a voffset past any range: the load returns 0
# sched_group_barrier masks
SG_VALU = 0x2
SG_MFMA = 0x8
SG_VMEM = 0x10
SG_DS_READ = 0x100
SG_DS_WRITE = 0x200
# sched_barrier mask: VALU, SALU, VMEM and DS writes may cross; DS reads and MFMAs may not.
PIN_MASK = 0x002 | 0x004 | 0x010 | 0x020 | 0x040 | 0x200 | 0x400
# Dev sweeps only: Traits field -> value, applied to every build (and its cache tag).
DEV_OVERRIDES = {}


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, kv_subs, padded_head=False):
        assert head_dim % 32 == 0 and 64 <= head_dim <= 512, head_dim
        assert kv_subs in (1, 2), kv_subs
        self.D = head_dim
        self.PADDED = padded_head
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NSUB = kv_subs                     # 16-key MFMA columns per wave
        self.ROWS = 16 * kv_subs                # keys per wave
        self.BLOCK_KV = NUM_WAVES * self.ROWS
        self.BLOCK_SIZE = NUM_WAVES * WAVE
        self.NT = head_dim // 32                # 16B RM reads per tensor and tile, two k-steps each
        self.NDT = head_dim // 16               # 16-row d tiles of dK^T / dV^T
        # RM rows an odd number of 16B chunks apart (conflict-free b128 reads), or, when
        # LDS is short (SWZ), packed with chunk c of row r at c ^ (r % 8); T likewise.
        chunks = head_dim // 8
        rm_pad = (chunks + 1 - chunks % 2) * 16
        self.ROW_BYTES = head_dim * 2
        # RM DMA: 256B per issue; the last issue of a row ends on its last byte.
        self.SEGS = tuple(range(0, self.ROW_BYTES - 256, 256)) + (self.ROW_BYTES - 256,)
        self.N_DMA = 2 * (BLOCK_M // NUM_WAVES) * len(self.SEGS)   # per wave and tile
        # T units: 4 q rows x 8 d of one tensor; waves 0-1 take Q, 2-3 dO.
        self.UNITS = head_dim // 2
        self.U_PER_THREAD = -(-self.UNITS // (2 * WAVE))
        # Where T comes from: 'lds' transposes the RM tile already in LDS into a
        # double-buffered T; 'global' loads the tile again into VGPRs for a single T.
        # LSE/Delta are DMA'd beside T (SIDE_LDS) or loaded into VGPRs a tile ahead.
        rm_buf = 2 * BLOCK_M * rm_pad
        t_pad = 2 * self.NDT * 4 * T_G_PAD
        packed = 2 * BLOCK_M * self.ROW_BYTES  # an RM or T buffer without padding
        if 2 * rm_buf + 2 * (t_pad + SIDE_BYTES) <= 65536:
            self.T_SRC, self.SWZ, self.SIDE_LDS, self.RM_SLOTS = 'lds', False, True, 2
        elif 4 * packed <= 65536:
            self.T_SRC, self.SWZ, self.SIDE_LDS, self.RM_SLOTS = 'lds', True, False, 2
        else:
            self.T_SRC, self.SIDE_LDS = 'global', False
            self.RM_SLOTS = 2 if 2 * rm_buf + t_pad <= 65536 else 1
            self.SWZ = rm_buf + t_pad > 65536
        # 'global': how many tiles ahead T's loads run.
        self.G_AHEAD = 1
        self.PREFETCH = 2                       # LDS operand reads in flight ahead of their MFMAs
        self.KV_VGPR = True                     # pin K, V to VGPRs
        self.TB_POS = 99                        # T-build write's place among X's spread work (99: last)
        # Walk q tiles last to first: a causal head's blocks then all start on its last
        # tiles and share them in L2, instead of each starting at its own diagonal.
        self.Q_DOWN = causal
        self.VALU_PER_MFMA = 2                  # sched_group_barrier slot size
        self.SCHED = False
        self.TRACE_SUB = False                  # dev: ktrace marks inside X
        self.EXP = ''                           # dev experiments (wrong results): 'not', 'nol'
        for k, v in DEV_OVERRIDES.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        self.RM_STRIDE = self.ROW_BYTES if self.SWZ else rm_pad
        self.RM_TENSOR = BLOCK_M * self.RM_STRIDE
        self.RM_BUF = 2 * self.RM_TENSOR        # Q then dO
        self.T_GS = 128 if self.SWZ else T_G_PAD
        self.T_DTS = 4 * self.T_GS
        self.T_TENSOR = self.NDT * self.T_DTS
        if self.SWZ:
            assert head_dim % 128 == 0, head_dim
        self.LDS_RM = 0
        if self.T_SRC == 'lds':
            assert self.RM_SLOTS == 2
            self.LDS_T = 2 * self.RM_BUF
            self.T_SLOT = 2 * self.T_TENSOR + (SIDE_BYTES if self.SIDE_LDS else 0)
            self.LDS_BYTES = self.LDS_T + 2 * self.T_SLOT
            # VMEM a Y cluster issues after the RM DMA it must see land a tile later.
            self.N_VM_Y = self.N_DMA + (2 if self.SIDE_LDS else 0)
        else:
            self.LDS_T = self.RM_SLOTS * self.RM_BUF
            self.T_SLOT = 0
            self.LDS_BYTES = self.LDS_T + 2 * self.T_TENSOR
            n_ld = 4 * self.U_PER_THREAD + 8 if self.G_AHEAD == 2 else 0
            # RM double: Y(j)'s DMA is waited a tile later; single: by its own end.
            self.N_VM_Y = (self.N_DMA if self.RM_SLOTS == 2 else 0) + n_ld
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        self.CACHE_TAG = (f'BW{head_dim}_c{int(causal)}_{dtype_str}_s{kv_subs}_p{int(padded_head)}_{self.T_SRC}'
                          f'_f{self.PREFETCH}_k{int(self.KV_VGPR)}_tb{self.TB_POS}_d{int(self.Q_DOWN)}_r{self.RM_SLOTS}_g{self.G_AHEAD}_z{int(self.SWZ)}{int(self.SIDE_LDS)}_a{self.VALU_PER_MFMA}_x{int(self.SCHED)}_t{int(self.TRACE_SUB)}_e{self.EXP}')


# csrc/flyc_common.h's kFlycGridAxis*, as fmha_tuning_gfx942.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1


def default_grid_axis_order(causal):
    """Tile-fastest shares a head's Q/dO stream in L2; causal blocks differ in
    length and want head-fastest (hd192 B2 H32 S8k: 431 vs 336 TFLOPS)."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def default_kv_subs(head_dim):
    """Keys per wave / 16: 32 keys halve the LDS and global traffic per FLOP."""
    return 2


def build_fmha_bwd_dkdv_gfx942_wide(head_dim,
                                    causal=False,
                                    causal_type=None,
                                    dtype_str='bf16',
                                    bias=False,
                                    dropout=False,
                                    kv_subs=None,
                                    waves_per_eu=1,
                                    grid_axis_order=None,
                                    padded_head=False):
    """The gfx1201 dK/dV kernel's ABI and `_launch` interface, for wide heads."""
    host_causal_type = (1 if causal else 0) if causal_type is None else causal_type
    if host_causal_type not in (0, 1, 2, 3):
        raise ValueError(f'causal_type must be 0-3, got {host_causal_type}')
    causal_type = 0 if host_causal_type == 0 else 3
    if bias and causal_type:
        raise ValueError('bias with a causal mask is not served (aot/_common.py flash_disabled)')
    if kv_subs is None:
        kv_subs = default_kv_subs(head_dim)
    tr = Traits(head_dim, causal_type != 0, dtype_str, kv_subs, padded_head)
    if grid_axis_order is None:
        grid_axis_order = default_grid_axis_order(tr.CAUSAL)
    tile_fastest = grid_axis_order == GRID_AXIS_TILE_FASTEST
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}'
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
        """Four f32 -> one operand; bf16 rounds to nearest, ties away (lessons #16)."""
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

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
    # The kernel ABI is bwd_dkdv_kernel's (gfx1201), so aot/flyc_bwd_dkdv.py can serve both.
    def fmha_bwd_dkdv_gfx942_wide_kernel(
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

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)

        def span_bytes(rows, ss, row_elems):
            span = (fx.Int64(rows) - fx.Int64(1)) * ss + fx.Int64(row_elems)
            return (fx.Int32(rows) > fx.Int32(0)).select(span, fx.Int64(0)) * fx.Int64(2)

        def base_addr(ptr, batch, h, row_off, sb, sh, ss):
            elems = fx.Int64(batch) * sb + fx.Int64(h) * sh + fx.Int64(row_off) * ss
            return fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(2)

        def rsrc_for(ptr, batch, h, row_off, sb, sh, ss, rows, row_elems):
            # One sequence of one head, ending on its last row's last owned column.
            return buffer_ops.create_buffer_resource_from_addr(base_addr(ptr, batch, h, row_off, sb, sh, ss),
                                                               num_records_bytes=span_bytes(rows, ss, row_elems))

        def col_mask(c0, hdim):
            """Dword masks keeping columns c0 + [0, 8) below hdim."""
            words = []
            for w in range_constexpr(4):
                lo = (c0 + fx.Int32(2 * w) < hdim).select(fx.Int32(0xFFFF), fx.Int32(0))
                hi = (c0 + fx.Int32(2 * w + 1) < hdim).select(fx.Int32(-65536), fx.Int32(0))
                words.append(lo | hi)
            return Vec.from_elements(words, fx.Int32)

        def masked8(v8, c0, hdim):
            return Vec(Vec(v8).bitcast(fx.Int32) & col_mask(c0, hdim)).bitcast(elem).ir_value()

        def pin_vgpr(vals):
            """Opaque VGPR copies: LLVM can neither look through them nor move them to AGPRs."""
            if const_expr(not vals):
                return []
            irs = [raw(v) for v in vals]
            ty = ir.Type.parse('!llvm.struct<(' + ', '.join(str(x.type) for x in irs) + ')>')
            cons = ','.join(['=v'] * len(irs) + [str(i) for i in range(len(irs))])
            ret = llvm.inline_asm(ty, irs, '', cons, has_side_effects=True)
            return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(len(irs))]

        # ---- K and V: resident B operands, key kvw + 16*sub + lc, d = 32t + 8gq + [0, 8) ----
        k0 = fx.Int32(kv_tile) * fx.Int32(tr.BLOCK_KV)
        kvw = k0 + wave_s * fx.Int32(tr.ROWS)
        k_rsrc = rsrc_for(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq, sk_i32, row_qk)
        v_rsrc = rsrc_for(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq, sk_i32, row_vo)

        def resident(rsrc, ss, hdim):
            out = []
            for t in range_constexpr(tr.NT):
                row = []
                for sub in range_constexpr(tr.NSUB):
                    r = fx.Index(kvw + fx.Int32(16 * sub)) + lc
                    c0 = fx.Index(32 * t) + gq * 8
                    v8 = buffer_ops.buffer_load(rsrc, r * fx.Index(fx.Int32(ss)) + c0, vec_width=8, dtype=elem)
                    if const_expr(tr.PADDED):
                        # Slack columns may hold NaN, and 0 * NaN is NaN.
                        v8 = masked8(v8, fx.Int32(c0), hdim)
                    row.append(Vec(v8))
                out.append(row)
            return out

        k_res = resident(k_rsrc, stride_k_seq, fx.Int32(hdim_qk))
        v_res = resident(v_rsrc, stride_v_seq, fx.Int32(hdim_vo))
        if const_expr(tr.KV_VGPR):
            # Keep K and V out of the AGPRs, which the accumulators fill.
            flat = pin_vgpr([x for row in k_res + v_res for x in row])
            k_res = [[Vec(flat[t * tr.NSUB + sub]) for sub in range(tr.NSUB)] for t in range(tr.NT)]
            n_k = tr.NT * tr.NSUB
            v_res = [[Vec(flat[n_k + t * tr.NSUB + sub]) for sub in range(tr.NSUB)] for t in range(tr.NT)]

        def smin(x, y):
            return (x < y).select(x, y)

        def smax(x, y):
            return (x > y).select(x, y)

        # ---- The visited q range of this key block, and its clean (unmasked) tiles ----
        k_hi = smin(k0 + fx.Int32(tr.BLOCK_KV), sk_i32) - fx.Int32(1)
        if const_expr(tr.CAUSAL):
            win_l, win_r = fmha.resolve_window(window_left, window_right, sq_i32, sk_i32, varlen_bits)
            q_lo = smax(k0 - win_r, fx.Int32(0))
            q_hi = smin(k_hi + win_l, sq_i32 - fx.Int32(1))
        else:
            q_lo = fx.Int32(0)
            q_hi = sq_i32 - fx.Int32(1)
        alive = (k0 < sk_i32) & (q_lo <= q_hi)
        t0 = alive.select(q_lo, fx.Int32(0)) // fx.Int32(BLOCK_M)
        n_tiles = alive.select(q_hi // fx.Int32(BLOCK_M) - t0 + fx.Int32(1), fx.Int32(0))
        n_loop = smax(n_tiles - fx.Int32(1), fx.Int32(0))
        t_last = t0 + n_loop
        n_pairs = n_loop // fx.Int32(2)

        # This lane's keys see q rows [band_lo, band_lo + band_span]: one unsigned compare per element.
        # A block with no tile still runs tile 0 (prologue, finish) under an empty band.
        band_lo = []
        if const_expr(tr.CAUSAL):
            span = win_l + win_r
            empty = (span < fx.Int32(0)) | (n_tiles == fx.Int32(0))
            band_span = empty.select(fx.Int32(0), span)
            for sub in range_constexpr(tr.NSUB):
                kv = kvw + fx.Int32(16 * sub) + fx.Int32(lc)
                band_lo.append(empty.select(fx.Int32(-(1 << 30)), kv - win_r))
        else:
            empty = n_tiles == fx.Int32(0)
            band_span = empty.select(fx.Int32(0), fx.Int32(1 << 30))
            band_lo = [empty.select(fx.Int32(-(1 << 30)), fx.Int32(0))] * tr.NSUB

        # ---- Per-lane LDS offsets ----
        # RM read t: row lc, chunk 4t + gq (XORed by lc % 8 under SWZ: two bases by t's parity).
        if const_expr(tr.SWZ):
            rm_rd = [fx.Index(tr.LDS_RM) + lc * tr.RM_STRIDE + ((fx.Index(4 * e) + gq) ^ (lc % 8)) * 16
                     for e in range(2)]
            t_rd = fx.Index(tr.LDS_T) + gq * tr.T_GS + (((lc // 2) ^ gq) * 16) + (lc % 2) * 8
        else:
            rm_rd = [fx.Index(tr.LDS_RM) + lc * tr.RM_STRIDE + gq * 16] * 2
            t_rd = fx.Index(tr.LDS_T) + gq * tr.T_GS + lc * 8
        lds_tile = tr.T_SRC == 'lds'

        def rm_read(slot, tensor, t):
            off = slot * tr.RM_BUF + tensor * tr.RM_TENSOR + (128 * (t // 2) if tr.SWZ else 64 * t)
            return lds_load(rm_rd[t % 2] + off, v8e(), 16, f'rm{slot}')

        def t_read(slot, tensor, dt):
            return lds_load(t_rd + (slot * tr.T_SLOT + tensor * tr.T_TENSOR + dt * tr.T_DTS), v4e(), 8, f't{slot}')

        # T units: waves 0-1 Q, 2-3 dO; unit i = (g = i % 4, c8 = i // 4), extra threads redo one.
        t_tensor = wave_s // fx.Int32(2)
        is_do = t_tensor == fx.Int32(1)
        units = [(tid % (2 * WAVE) + fx.Index(2 * WAVE * u)) % tr.UNITS for u in range(tr.U_PER_THREAD)]
        u_g = [x % 4 for x in units]
        u_c8 = [x // 4 for x in units]
        t_wr = [fx.Index(tr.LDS_T) + fx.Index(t_tensor) * tr.T_TENSOR + (c8 // 2) * tr.T_DTS + g * tr.T_GS
                + (c8 % 2) * 64 for g, c8 in zip(u_g, u_c8)]
        # The unit's four rows in RM, for the 'lds' T source.
        t_src = [fx.Index(tr.LDS_RM) + fx.Index(t_tensor) * tr.RM_TENSOR + (u_g[0] * 4 + rr) * tr.RM_STRIDE
                 + ((u_c8[0] ^ ((u_g[0] * 4 + rr) % 8)) if tr.SWZ else u_c8[0]) * 16 for rr in range(4)]
        side_rd = fx.Index(tr.LDS_T + 2 * tr.T_TENSOR) + gq * 16
        # DMA voffset of row 4w + i: its dword, its 16B chunk XORed by the row under SWZ.
        if const_expr(tr.SWZ):
            dma_voffs = [(fx.Int32(lane // 4) ^ ((wave_s * fx.Int32(4) + fx.Int32(i)) & fx.Int32(7))) * fx.Int32(16)
                         + fx.Int32(lane % 4) * fx.Int32(4) for i in range(BLOCK_M // NUM_WAVES)]

        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        sel_lo = fx.Int32(0x05040100)
        sel_hi = fx.Int32(0x07060302)
        m0_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_base + fx.Index(tr.LDS_RM))))
        dma_voff = fx.Int32(lane * 4)
        # This wave's first RM row; the per-row constants join it at each DMA.
        dma_m0 = m0_base + wave_s * fx.Int32((BLOCK_M // NUM_WAVES) * tr.RM_STRIDE)

        def m0_at(base, const=0):
            """base + const at the use: hoisted, the DMAs' M0 values would pin a dozen SGPRs."""
            return fx.Int32(llvm.inline_asm(T.i32,
                                            [raw(fx.Int32(base))],
                                            f's_add_u32 $0, $1, {const}',
                                            '=s,s',
                                            has_side_effects=True))
        zero4 = fx.full(4, 0.0, fx.Float32)
        if const_expr(dropout):
            rng = Philox.for_arch('gfx942')
            ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        def sgb(mask, n):
            if const_expr(n > 0):
                rocdl.sched_group_barrier(mask, n, 0)

        def sched(n_mfma, n_rd, n_wr, n_vmem):
            """Spread LDS reads, LDS writes and VMEM over the MFMAs."""
            if const_expr(not tr.SCHED):
                return
            for i in range_constexpr(n_mfma):
                sgb(SG_MFMA, 1)
                sgb(SG_DS_READ, (i + 1) * n_rd // n_mfma - i * n_rd // n_mfma)
                sgb(SG_DS_WRITE, (i + 1) * n_wr // n_mfma - i * n_wr // n_mfma)
                sgb(SG_VMEM, (i + 1) * n_vmem // n_mfma - i * n_vmem // n_mfma)
                sgb(SG_VALU, tr.VALU_PER_MFMA)

        def pinned():
            """Fix the order of LDS reads and MFMAs; everything else may cross."""
            rocdl.sched_barrier(PIN_MASK)

        def stream(n, read, use, hooks=()):
            """Software-pipelined LDS operands: read t + PREFETCH, then use t.
            `hooks` (other work) run spread over the steps, in order."""
            at = {}
            for i, h in enumerate(hooks):
                at.setdefault(min(n - 1, (2 * i + 1) * n // (2 * len(hooks))), []).append(h)
            ahead = [read(t) for t in range(min(tr.PREFETCH, n))]
            for t in range_constexpr(n):
                if const_expr(t + tr.PREFETCH < n):
                    ahead.append(read(t + tr.PREFETCH))
                pinned()
                use(t, ahead[t])
                for h in at.get(t, []):
                    h()
                pinned()

        def anchor(vals):
            """Opaque copies: loop-carried load results would otherwise cost a vmcnt(0) at the loop head."""
            if const_expr(not vals):
                return []
            irs = [raw(v) for v in vals]
            ty = ir.Type.parse('!llvm.struct<(' + ', '.join(str(x.type) for x in irs) + ')>')
            cons = ','.join(['=v'] * len(irs) + [str(i) for i in range(len(irs))])
            ret = llvm.inline_asm(ty, irs, '', cons, has_side_effects=True)
            return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(len(irs))]

        def set_scopes(op, name):
            al, noal = scope_attrs(name)
            op = getattr(op, 'owner', op)
            op = getattr(op, 'operation', op)
            op.attributes['alias_scopes'] = al
            op.attributes['noalias_scopes'] = noal

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
            if const_expr(not lds_tile):
                # The T unit's tensor, picked per wave.
                u_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                    is_do.select(do_base, q_base), num_records_bytes=is_do.select(do_span, q_span)))
                u_rs_b = is_do.select(do_rs_b, q_rs_b)
                u_voff = [fx.Int32(g * 4) * u_rs_b + fx.Int32(c8 * 16) for g, c8 in zip(u_g, u_c8)]

            lse_row0, lse_pitch = fmha.lse_row_addressing(varlen_bits,
                                                          fx.Index(q_batch),
                                                          head_q,
                                                          num_head_q,
                                                          lse_tokens,
                                                          fx.Index(q_row_off))
            pitch_b = fx.Int32(lse_pitch) * fx.Int32(4)
            side_bytes = fx.Int64(fx.Index(sq_i32) * lse_pitch) * fx.Int64(4)
            lse_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(LSE)) + fx.Int64(lse_row0) * fx.Int64(4), num_records_bytes=side_bytes))
            delta_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(Delta)) + fx.Int64(lse_row0) * fx.Int64(4), num_records_bytes=side_bytes))
            # Side DMA: lane L < 16 copies row q0 + L; the rest read 0. VGPR side: rows 4gq + r.
            side_dma_voff = (lane < fx.Index(BLOCK_M)).select(fx.Int32(lane) * pitch_b, fx.Int32(OOB_VOFF))
            side_voff = fx.Int32(gq * 4) * pitch_b

            if const_expr(bias):
                b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head_q) * stride_b_head
                           + fx.Int64(q_row_off) * stride_b_seq_q)
                b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(k_row_off + sk_i32),
                              fx.Int64(0))
                b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2), num_records_bytes=b_span * fx.Int64(2)))
                b_rs_b = fx.Int32(stride_b_seq_q) * fx.Int32(2)
                b_voff = [fx.Int32(gq * 4) * b_rs_b + (k_row_off + kvw + fx.Int32(16 * sub + lc)) * fx.Int32(2)
                          for sub in range(tr.NSUB)]

            if const_expr(dropout):
                ph_plane = z_i32 * fx.Int32(num_head_q) + fx.Int32(head_q)
                ph_base, ph_stride = rng.grid_plane(ph_off, ph_plane, max_seqlen_q, max_seqlen_k)

            def q0_of(j):
                if const_expr(tr.Q_DOWN):
                    return (t_last - fx.Int32(j)) * fx.Int32(BLOCK_M)
                return (t0 + fx.Int32(j)) * fx.Int32(BLOCK_M)

            def dma_rows(j, slot):
                """RM(j) -> slot as one callable per row: wave w copies rows 4w..4w+3 of Q and of dO."""
                q0 = q0_of(j)
                out = []
                for tensor in range_constexpr(2):
                    for i in range_constexpr(BLOCK_M // NUM_WAVES):
                        def row_dma(tensor=tensor, i=i):
                            rsrc = do_rsrc if tensor else q_rsrc
                            rs_b = do_rs_b if tensor else q_rs_b
                            row = wave_s * fx.Int32(BLOCK_M // NUM_WAVES) + fx.Int32(i)
                            soff = (q0 + row) * rs_b
                            m0 = fx.Int64(m0_at(dma_m0, slot * tr.RM_BUF + tensor * tr.RM_TENSOR + i * tr.RM_STRIDE))
                            for seg in tr.SEGS:
                                set_scopes(rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                                         lds_ptr_at(m0),
                                                                         fx.Int32(4),
                                                                         dma_voffs[i] if tr.SWZ else dma_voff,
                                                                         soff,
                                                                         fx.Int32(seg),
                                                                         fx.Int32(0)),
                                           f'rm{slot}')
                        out.append(row_dma)
                return out

            def dma(j, slot):
                for f in dma_rows(j, slot):
                    f()

            def side_dma(j, slot):
                """LSE(j) and Delta(j) -> T[slot]'s side area, 256B each."""
                soff = q0_of(j) * pitch_b
                for i, rsrc in enumerate((lse_rsrc, delta_rsrc)):
                    m0 = fx.Int64(m0_at(m0_base, tr.LDS_T + slot * tr.T_SLOT + 2 * tr.T_TENSOR + 256 * i))
                    set_scopes(rocdl.raw_ptr_buffer_load_lds(rsrc,
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
                lse = Vec(lds_load(side_rd + slot * tr.T_SLOT, v4i, 16, f's{slot}')).bitcast(fx.Float32)
                dlt = Vec(lds_load(side_rd + (slot * tr.T_SLOT + 256), v4i, 16, f's{slot}')).bitcast(fx.Float32)
                return [lse[r] for r in range(4)], [dlt[r] for r in range(4)]

            def unit_load(j):
                """This thread's T units of tile j from global: 4 rows x 4 dwords each."""
                q0 = q0_of(j)
                out = []
                for u in range_constexpr(tr.U_PER_THREAD):
                    for rr in range_constexpr(4):
                        soff = (q0 + fx.Int32(rr)) * u_rs_b
                        w4 = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(4, fx.Int32), u_rsrc, u_voff[u], soff, 0))
                        out.extend(w4[i] for i in range(4))
                return out

            def t_write(regs, slot):
                al, noal = scope_attrs(f't{slot}')
                for u in range_constexpr(len(regs) // 16):
                    for k in range_constexpr(4):
                        dw = []
                        for e in (2 * k, 2 * k + 1):
                            w = [regs[16 * u + 4 * r + e // 2] for r in range(4)]
                            sel = sel_hi if e % 2 else sel_lo
                            dw += [fx.Int32(rocdl.perm_b32(w[1], w[0], sel)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel))]
                        if const_expr(tr.SWZ):
                            at = t_wr[u] + slot * tr.T_SLOT + (fx.Index(k) ^ u_g[u]) * 16
                        else:
                            at = t_wr[u] + (slot * tr.T_SLOT + 16 * k)
                        llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                                     lds_view(at, fx.Int32, 16).llvm_ptr,
                                     alignment=16,
                                     alias_scopes=al,
                                     noalias_scopes=noal)

            def t_build_read(slot):
                """This thread's unit of RM[slot], for T[slot]."""
                out = []
                for rr in range_constexpr(4):
                    w4 = Vec(lds_load(t_src[rr] + slot * tr.RM_BUF, Vec.make_type(4, fx.Int32), 16, f'rm{slot}'))
                    out.extend(w4[i] for i in range(4))
                return out

            def side_load(j):
                """LSE and Delta of tile j at q = q0 + 4gq + r, from global ('global' T source)."""
                soff = q0_of(j) * pitch_b
                out = []
                for rsrc in (lse_rsrc, delta_rsrc):
                    for r in range_constexpr(4):
                        out.append(fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type,
                                                                      rsrc,
                                                                      side_voff + fx.Int32(r) * pitch_b,
                                                                      soff,
                                                                      0)))
                return out

            def bias_load(j):
                """Bias of tile j at q = q0 + 4gq + r, key kvw + 16 sub + lc: one i16 each."""
                if const_expr(not bias):
                    return []
                bsoff = q0_of(j) * b_rs_b
                out = []
                for sub in range_constexpr(tr.NSUB):
                    for r in range_constexpr(4):
                        h = rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, b_voff[sub] + fx.Int32(r) * b_rs_b, bsoff, 0)
                        out.append(fx.Int32(fx.Int16(h)))
                return out

            def sdp(slot, hooks=()):
                """S and dP of the tile in RM[slot]: [v4f32] * NSUB each, q = 4gq + r, key lc."""
                s = [zero4] * tr.NSUB
                dp = [zero4] * tr.NSUB

                def read(t):
                    return Vec(rm_read(slot, 0, t)), Vec(rm_read(slot, 1, t))

                def use(t, ops):
                    q8, o8 = ops
                    if const_expr(tr.PADDED):
                        c0 = fx.Int32(32 * t) + fx.Int32(gq * 8)
                        q8 = Vec(masked8(q8, c0, fx.Int32(hdim_qk)))
                        o8 = Vec(masked8(o8, c0, fx.Int32(hdim_vo)))
                    for e in range_constexpr(2):
                        sl = [4 * e, 4 * e + 1, 4 * e + 2, 4 * e + 3]
                        qa = q8.shuffle(q8, sl).ir_value()
                        oa = o8.shuffle(o8, sl).ir_value()
                        for sub in range_constexpr(tr.NSUB):
                            kb = k_res[t][sub]
                            vb = v_res[t][sub]
                            s[sub] = mfma(qa, kb.shuffle(kb, sl).ir_value(), s[sub])
                            dp[sub] = mfma(oa, vb.shuffle(vb, sl).ir_value(), dp[sub])

                stream(tr.NT, read, use, hooks)
                return s, dp

            def softmax(s_raw, dp_raw, lse, delta, bvals, j, masked, sub, out):
                """P (dropped, for dV) and dS packs of tile j, key column `sub`, into out[sub]."""
                lse2 = [fx.Float32(x) * fx.Float32(LOG2E) for x in lse]
                q0 = q0_of(j)
                if const_expr(True):
                    sv = Vec(s_raw[sub]) * fx.full(4, fx.Float32(c_scale), fx.Float32)
                    dpv = Vec(dp_raw[sub])
                    if const_expr(masked):
                        rel0 = q0 + fx.Int32(gq * 4) - band_lo[sub]
                    if const_expr(dropout):
                        kv = kvw + fx.Int32(16 * sub) + fx.Int32(lc)
                        slot = fx.Int32(lc % rng.randoms_per_offset)
                    ps = []
                    dss = []
                    for r in range_constexpr(4):
                        x = sv[r]
                        if const_expr(bias):
                            bv = Vec.from_elements([fx.Int16(bvals[4 * sub + r])], fx.Int16).bitcast(elem)
                            # Upcast first: log2(e) in the input dtype would differ from the forward's.
                            x = x + fx.Float32(Vec(bv)[0].to(fx.Float32)) * fx.Float32(LOG2E)
                        p = exp2(x - lse2[r])
                        if const_expr(masked):
                            live = fx.Uint32(rel0 + fx.Int32(r)) <= fx.Uint32(band_span)
                            p = live.select(p, fx.Float32(0.0))
                        dpr = fx.Float32(dpv[r])
                        p_drop = p
                        if const_expr(dropout):
                            vals = rng.u32(ph_seed, rng.grid_offset(ph_base, ph_stride, q0 + fx.Int32(gq * 4 + r), kv))
                            val = vals[0]
                            for i in range_constexpr(1, len(vals)):
                                val = (slot == fx.Int32(i)).select(vals[i], val)
                            keep = fx.Int32(val) > fx.Int32(idropout_p)
                            p_drop = keep.select(p * fx.Float32(dropout_scale), fx.Float32(0.0))
                            dpr = keep.select(dpr * fx.Float32(dropout_scale), fx.Float32(0.0))
                        ps.append(p_drop)
                        dss.append(p * (dpr - fx.Float32(delta[r])))
                    out[sub] = (pack4(ps), pack4(dss))

            def dvdk(dk_accs, dv_accs, p_pk, ds_pk, slot, hooks=()):
                dk_accs = list(dk_accs)
                dv_accs = list(dv_accs)

                def read(u):
                    return [(t_read(slot, 1, dt), t_read(slot, 0, dt)) for dt in range(2 * u, min(2 * u + 2, tr.NDT))]

                def use(u, ops):
                    for i, (a_do, a_q) in enumerate(ops):
                        dt = 2 * u + i
                        for sub in range_constexpr(tr.NSUB):
                            k = dt * tr.NSUB + sub
                            dv_accs[k] = mfma(a_do, p_pk[sub], dv_accs[k])
                            dk_accs[k] = mfma(a_q, ds_pk[sub], dk_accs[k])

                stream((tr.NDT + 1) // 2, read, use, hooks)
                return dk_accs, dv_accs

            t_slot_of = (0, 1) if lds_tile else (0, 0)
            n_acc = tr.NDT * tr.NSUB
            n_gs = 16 * tr.U_PER_THREAD + 8          # 'global': one tile's T units and LSE/Delta
            n_extra_g = n_gs * tr.G_AHEAD
            if const_expr(lds_tile):
                n_extra_g = 0 if tr.SIDE_LDS else 8
            # RM slot of tile t, and the tile Y(j) DMAs: three ahead double-buffered, two single.
            rm_slot = (0, 1) if tr.RM_SLOTS == 2 else (0, 0)
            dma_ahead = 3 if tr.RM_SLOTS == 2 else 2

            def g_split(regs):
                """T units, LSE, Delta of one 'global' register set."""
                n_u = 16 * tr.U_PER_THREAD
                side = regs[n_u:n_gs]
                return (regs[:n_u],
                        [fx.Int32(x).bitcast(fx.Float32) for x in side[:4]],
                        [fx.Int32(x).bitcast(fx.Float32) for x in side[4:]])

            def unpack(c):
                o = 2 * n_acc
                return (list(c[:n_acc]),
                        list(c[n_acc:o]),
                        list(c[o:o + tr.NSUB]),
                        list(c[o + tr.NSUB:o + 2 * tr.NSUB]),
                        list(c[o + 2 * tr.NSUB:o + 2 * tr.NSUB + n_extra_g]),
                        list(c[o + 2 * tr.NSUB + n_extra_g:]))

            n_sdp = 4 * tr.NSUB * tr.NT
            n_dvdk = 2 * tr.NSUB * tr.NDT

            def iteration(j, carried, b, masked, last=False):
                """X(j), Y(j); b = j % 2, known at trace time. `last`: no tile j+1."""
                dk_accs, dv_accs, s_cur, dp_cur, g_regs, bvals = unpack(carried)
                # X(j): S(j+1), dP(j+1); P(j), dS(j); T(j+1) ('lds') or T(j) ('global').
                sub_marks = tr.TRACE_SUB and not last
                g_nxt = []
                if const_expr(lds_tile and tr.SIDE_LDS):
                    lse, delta = side_read(b)
                elif const_expr(lds_tile):
                    lse = [fx.Int32(x).bitcast(fx.Float32) for x in g_regs[:4]]
                    delta = [fx.Int32(x).bitcast(fx.Float32) for x in g_regs[4:8]]
                    if const_expr(not last):
                        g_nxt = side_load(j + 1)
                else:
                    # Y(j-1) is done with T. G_AHEAD 1 loads tile j+1 here; 2 loaded it in Y(j-1).
                    u_regs, lse, delta = g_split(g_regs[:n_gs])
                    if const_expr('not' not in tr.EXP):
                        t_write(u_regs, 0)
                    if const_expr(not last and 'nol' in tr.EXP):
                        g_nxt = g_regs
                    elif const_expr(not last and tr.G_AHEAD == 1):
                        g_nxt = unit_load(j + 1) + side_load(j + 1)
                b_nxt = []
                if const_expr(not last):
                    b_nxt = bias_load(j + 1)
                rocdl.sched_barrier(0)
                if const_expr(sub_marks):
                    ktrace.mark(f'a{b}')
                    rocdl.sched_barrier(0)
                pd = [None] * tr.NSUB
                work = []
                for sub in range_constexpr(tr.NSUB):
                    if const_expr('nosm' in tr.EXP):
                        pd[sub] = (Vec(s_cur[sub]).to(elem).ir_value(), Vec(dp_cur[sub]).to(elem).ir_value())
                    else:
                        work.append(lambda sub=sub: softmax(s_cur, dp_cur, lse, delta, bvals, j, masked, sub, pd))
                if const_expr(not last and lds_tile and 'notb' not in tr.EXP):
                    units = t_build_read(1 - b)
                    work.insert(min(tr.TB_POS, len(work)), lambda: t_write(units, 1 - b))
                if const_expr(not last):
                    s_nxt, dp_nxt = sdp(rm_slot[1 - b], work)
                else:
                    for w in work:
                        w()
                if const_expr(sub_marks):
                    rocdl.sched_barrier(0)
                    ktrace.mark(f'b{b}')
                    rocdl.sched_barrier(0)
                p_pk = [x[0] for x in pd]
                ds_pk = [x[1] for x in pd]
                if const_expr(not last):
                    sched(n_sdp, 2 * tr.NT, 0, 0)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier(None if last else f'X{b}')
                # Y(j): dV, dK of tile j; DMA RM(j+3) (and LSE/Delta(j+2)) spread over its MFMAs.
                vm_work = []
                pend = {}
                if const_expr(not last):
                    vm_work = dma_rows(j + dma_ahead, rm_slot[1 - b])
                    if const_expr(lds_tile and tr.SIDE_LDS):
                        vm_work.append(lambda: side_dma(j + 2, b))
                    elif const_expr(tr.G_AHEAD == 2 and 'nol' not in tr.EXP):
                        # After the DMA: the in-order vmcnt then gives these a tile and a half.
                        def far_loads():
                            rocdl.sched_barrier(0)
                            pend['g'] = unit_load(j + 2) + side_load(j + 2)
                        vm_work.append(far_loads)
                dk_accs, dv_accs = dvdk(dk_accs, dv_accs, p_pk, ds_pk, t_slot_of[b], vm_work)
                if const_expr(not last):
                    sched(n_dvdk, 2 * tr.NDT, 0, tr.N_DMA)
                    rocdl.s_waitcnt(vmcnt=tr.N_VM_Y)
                    if const_expr(not lds_tile and tr.G_AHEAD == 2 and 'nol' not in tr.EXP):
                        g_nxt = anchor(g_regs[n_gs:]) + pend['g']
                    else:
                        g_nxt = anchor(g_nxt)
                    b_nxt = anchor(b_nxt)
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier(None if last else f'Y{b}')
                if const_expr(last):
                    return dk_accs + dv_accs
                return dk_accs + dv_accs + s_nxt + dp_nxt + g_nxt + b_nxt

            @flyc.jit
            def odd_tail(j, carried):
                out = list(carried)
                if j < fx.Index(n_loop):
                    out = iteration(j, carried, 0, tr.CAUSAL)
                return out

            @flyc.jit
            def finish(j, carried):
                out = list(carried[:2 * n_acc])
                if j % fx.Index(2) == fx.Index(0):
                    out = iteration(j, carried, 0, True, last=True)
                else:
                    out = iteration(j, carried, 1, True, last=True)
                return out

            # Prologue: RM(0), RM(1) DMA; X(-1) = S(0), dP(0) (and T(0)); Y(-1) = DMA RM(2).
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            dma(0, 0)
            if const_expr(lds_tile and tr.SIDE_LDS):
                side_dma(0, 0)
                g0 = []
            elif const_expr(lds_tile):
                g0 = side_load(0)
            else:
                g0 = unit_load(0) + side_load(0)
            b0 = bias_load(0)
            rocdl.sched_barrier(0)
            if const_expr(tr.RM_SLOTS == 2):
                dma(1, 1)
                rocdl.s_waitcnt(vmcnt=tr.N_DMA)
            else:
                rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            s0, dp0 = sdp(0)
            if const_expr(lds_tile):
                t_write(t_build_read(0), 0)
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()
            dma(dma_ahead - 1, rm_slot[(dma_ahead - 1) % 2])
            if const_expr(lds_tile and tr.SIDE_LDS):
                side_dma(1, 1)
            elif const_expr(tr.G_AHEAD == 2):
                rocdl.sched_barrier(0)
                g0 = g0 + unit_load(1) + side_load(1)
            rocdl.s_waitcnt(vmcnt=tr.N_VM_Y)
            cluster_barrier()

            # One body for every tile: masking costs a few VALU an element, a second
            # (masked) body costs registers. Pairs from j = 0, then at most one even j.
            init = list(accs) + s0 + dp0 + g0 + b0
            res1 = init
            for jp, carried in range(0, fx.Index(n_pairs), 1, init=init):
                mid = iteration(jp * 2, carried, 0, tr.CAUSAL or 'mask' in tr.EXP)
                res1 = yield iteration(jp * 2 + 1, mid, 1, tr.CAUSAL or 'mask' in tr.EXP)
            res2 = odd_tail(fx.Index(n_pairs) * 2, res1)
            return finish(fx.Index(n_loop), res2)

        n_acc_all = 2 * tr.NDT * tr.NSUB
        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range(n_acc_all)]
        results = acc0
        for g, carried in range(0, fx.Index(group), 1, init=acc0):
            results = yield head_pass(fx.Int32(g), carried)

        # ---- Epilogue: lane holds dK^T / dV^T [d = 16dt + 4gq + r][key kvw + 16sub + lc] ----
        rocdl.s_waitcnt(vmcnt=0)
        dk_rsrc = rsrc_for(DK, k_batch, head_k, k_row_off, stride_dk_batch, stride_dk_head, stride_dk_seq, sk_i32,
                           row_qk)
        dv_rsrc = rsrc_for(DV, k_batch, head_k, k_row_off, stride_dv_batch, stride_dv_head, stride_dv_seq, sk_i32,
                           row_vo)
        scale_v = fx.full(4, fx.Float32(sm_scale), fx.Float32)
        n_acc = tr.NDT * tr.NSUB
        for which in range_constexpr(2):
            rsrc = dv_rsrc if which else dk_rsrc
            ss = fx.Index(fx.Int32(stride_dv_seq if which else stride_dk_seq))
            row_lim = row_vo if which else row_qk
            for dt in range_constexpr(tr.NDT):
                for sub in range_constexpr(tr.NSUB):
                    acc = Vec(results[which * n_acc + dt * tr.NSUB + sub])
                    if const_expr(not which):
                        # dS is against the scaled scores: sm_scale lands once, here.
                        acc = acc * scale_v
                    packed = Vec.from_elements([fx.Float32(acc[r]).to(elem) for r in range(4)], elem)
                    kv = kvw + fx.Int32(16 * sub) + fx.Int32(lc)
                    d = fx.Int32(16 * dt) + fx.Int32(gq * 4)
                    ok = (kv < sk_i32) & (d < row_lim)
                    buffer_ops.buffer_store(packed.ir_value(), rsrc, fx.Index(kv) * ss + fx.Index(d), mask=ok)

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
        grid_x = (fx.Index(max_seqlen_k) + tr.BLOCK_KV - 1) // tr.BLOCK_KV
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_bwd_dkdv_gfx942_wide_kernel(Q,
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
                                         value_attrs={'rocdl.waves_per_eu': waves_per_eu,
                                                      'rocdl.flat_work_group_size':
                                                      f'{tr.BLOCK_SIZE},{tr.BLOCK_SIZE}'}).launch(
            grid=(grid_x, fx.Index(num_head_k), grid_z) if tile_fastest else (fx.Index(num_head_k), grid_x, grid_z),
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
        """fmha_bwd_dkdv_gfx1201_kernel's `_launch`."""
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
        ps, po1, po2, ip, dsc, _hold = abi.dropout_args(dropout,
                                                        dropout_p,
                                                        philox_seed,
                                                        philox_offset1,
                                                        philox_offset2,
                                                        Q.device,
                                                        stream)
        bp, _, bs, _ = abi.bias_args(1 if bias_on else 0, False, bias, None, Q)
        abi.run_compiled(compiled,
                         launch_fmha_bwd_dkdv_gfx942_wide,
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

    bias_on = bias
    _launch.traits = tr
    _launch.varlen_bits = abi.varlen_bits
    _launch.varlen_compact = abi.varlen_compact
    _launch.varlen_padded = abi.varlen_padded
    _launch.varlen_strided = abi.varlen_strided
    return _launch
