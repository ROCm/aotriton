# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dK/dV backward for gfx942 at wide head dims (256..448), FlyDSL: role-split teams.

Two waves form a team that owns 16 * KV_SUBS keys. The K-wave holds K and dK^T, the
V-wave V and dV^T (gfx1201's split_head_dim), so a wave keeps 0.75 * D registers of
state at KV_SUBS = 2 where the 'wide' body needs 1.5 * D, and reads only its own
tensor: half the LDS reads per MFMA of a 16-key wave.

    K-wave: S    = Q . K^T      A = Q  (RM)        B = K (registers)
            dK^T += Q^T . dS    A = Q^T (derived)  B = dS
    V-wave: dP   = dO . V^T     A = dO (RM)        B = V
            dV^T += dO^T . P    A = dO^T (derived) B = P

Q and dO live in LDS row-major only (DMA, double-buffered); the transposed A operand
is four ds_read_u16 of one column ("derived", gfx1201's transposed_source). Each
16-row q tile is three barrier-separated phases:
    X: S or dP from RM; both raw tiles to the team exchange; DMA RM(j+1)
    Z: P and dS of half the team's key columns (sub % 2 == role) to the exchange
    Y: dK^T or dV^T += tile j from the derived reads of RM(j)
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
from philox import Philox

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942_xwide.lds">'
LDS_SCOPES = ('rm0', 'rm1', 'rm2', 'xs', 'xp', 's0', 's1', 's2')
LOG2E = 1.4426950408889634
WAVE = 64
NUM_WAVES = 4
BLOCK_M = 16            # q rows per tile: the N of the 16x16x16 MFMA
XS_SUB = 2 * WAVE * 16  # exchange of one key column: raw S and dP, 16B per lane each
XP_SUB = 2 * WAVE * 8   # P and dS packs of one key column, 8B per lane each
SIDE_SLOT = 512         # LSE then Delta of one tile, 256B per DMA issue
OOB_VOFF = 0x7FFFFFF0   # a voffset past any range: the load returns 0
# sched_barrier mask: VALU, SALU, VMEM and DS writes may cross; DS reads and MFMAs may not.
PIN_MASK = 0x002 | 0x004 | 0x010 | 0x020 | 0x040 | 0x200 | 0x400
# Dev sweeps only: Traits field -> value, applied to every build (and its cache tag).
DEV_OVERRIDES = {}


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, kv_subs, padded_head=False):
        assert head_dim % 32 == 0 and 64 <= head_dim <= 512, head_dim
        assert kv_subs in (2, 4), kv_subs
        self.D = head_dim
        self.PADDED = padded_head
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NSUB = kv_subs                     # 16-key MFMA columns per team
        self.TEAM_KEYS = 16 * kv_subs
        self.BLOCK_KV = (NUM_WAVES // 2) * self.TEAM_KEYS
        self.BLOCK_SIZE = NUM_WAVES * WAVE
        self.NT = head_dim // 32                # 16B RM reads per tile, two k-steps each
        self.NDT = head_dim // 16               # 16-row d tiles of dK^T / dV^T
        x_bytes = (NUM_WAVES // 2) * kv_subs * (XS_SUB + XP_SUB) + 3 * SIDE_SLOT
        # A tile's D in PARTS column parts, each its own RM slot: two whole-tile slots,
        # or (at 512, where those and the exchange outgrow LDS) a ring of three half-tiles.
        self.PARTS = 1 if 2 * 2 * BLOCK_M * self._stride(head_dim) + x_bytes <= 65536 else 2
        self.PREFETCH = 2                       # RM reads in flight ahead of their MFMAs
        self.Y_AHEAD = 2                        # derived read pairs in flight ahead of their MFMAs
        # Pin the resident K or V to VGPRs ('v') or AGPRs ('a', MFMA reads B from either);
        # padded builds leave LLVM free (the wide body's regalloc aborts). 512 dense: 'a' 344
        # TFLOPS (36 spills), 'v' 323 (138); 384 ties.
        self.KV_PIN = '' if padded_head else ('a' if head_dim > 384 else 'v')
        self.FRESH_BASES = True                 # per-tile opaque LDS slot bases
        self.OPS_VGPR = False                   # keep RM operands out of AGPRs
        # Walk q tiles last to first: a causal head's blocks then all start on its last
        # tiles and share them in L2, instead of each starting at its own diagonal.
        self.Q_DOWN = causal
        for k, v in DEV_OVERRIDES.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        self.DP = head_dim // self.PARTS        # columns per part
        assert self.DP % 32 == 0, (head_dim, self.PARTS)
        self.NT_P = self.NT // self.PARTS
        self.NDT_P = self.NDT // self.PARTS
        self.RM_SLOTS = 2 if self.PARTS == 1 else 3
        self.PERIOD = 2 if self.PARTS == 1 else 3   # tiles until the slot pattern repeats
        self.RM_STRIDE = self._stride(self.DP)
        self.PART_BYTES = self.DP * 2
        # RM DMA: 256B per issue; the last issue of a row ends on its last byte.
        self.SEGS = tuple(range(0, self.PART_BYTES - 256, 256)) + (self.PART_BYTES - 256,)
        self.N_DMA = 2 * (BLOCK_M // NUM_WAVES) * len(self.SEGS)   # per wave and part
        self.RM_TENSOR = BLOCK_M * self.RM_STRIDE
        self.RM_BUF = 2 * self.RM_TENSOR        # Q then dO
        self.LDS_RM = 0
        self.LDS_XS = self.RM_SLOTS * self.RM_BUF
        self.XS_TEAM = kv_subs * XS_SUB
        self.LDS_XP = self.LDS_XS + (NUM_WAVES // 2) * self.XS_TEAM
        self.XP_TEAM = kv_subs * XP_SUB
        self.LDS_SIDE = self.LDS_XP + (NUM_WAVES // 2) * self.XP_TEAM
        self.LDS_BYTES = self.LDS_SIDE + self.PERIOD * SIDE_SLOT
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        self.CACHE_TAG = (f'BX{head_dim}_c{int(causal)}_{dtype_str}_s{kv_subs}_p{int(padded_head)}_h{self.PARTS}'
                          f'_f{self.PREFETCH}.{self.Y_AHEAD}_k{self.KV_PIN}_d{int(self.Q_DOWN)}_b{int(self.FRESH_BASES)}{int(self.OPS_VGPR)}')

    @staticmethod
    def _stride(cols):
        """RM rows an odd number of 16B chunks apart: conflict-free b128 reads, and the
        derived reads' four row groups (4 rows apart) land on distinct bank halves."""
        chunks = cols // 8
        return (chunks + 1 - chunks % 2) * 16


def lds_bytes(head_dim, kv_subs):
    """LDS of a build, for the tuning module's legality check."""
    return Traits(head_dim, False, 'bf16', kv_subs).LDS_BYTES


# csrc/flyc_common.h's kFlycGridAxis*, as fmha_tuning_gfx942.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1


def default_grid_axis_order(causal):
    """Tile-fastest shares a head's Q/dO stream in L2; causal blocks differ in
    length and want head-fastest."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def default_kv_subs(head_dim):
    """16-key columns per team, from the tuning module (flydsl-free)."""
    import fmha_tuning_bwd_dkdv_gfx942 as tuning
    return tuning.default_kv_subs(head_dim) if tuning.body(head_dim) == 'xwide' else 2


def build_fmha_bwd_dkdv_gfx942_xwide(head_dim,
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
    block_kv = tr.BLOCK_KV
    soft_subs = tr.NSUB // 2                    # key columns each role turns into P and dS

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

    def cluster_barrier():
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[tr.BLOCK_SIZE, 1, 1])
    # The kernel ABI is bwd_dkdv_kernel's (gfx1201), so aot/flyc_bwd_dkdv.py can serve both.
    def fmha_bwd_dkdv_gfx942_xwide_kernel(
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
        role = wave_s & fx.Int32(1)             # 0: K, S, dK; 1: V, dP, dV
        team = wave_s >> fx.Int32(1)
        is_v = role == fx.Int32(1)

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

        def lds_store(off, val, align, scope):
            al, noal = scope_attrs(scope)
            llvm.StoreOp(raw(val),
                         lds_view(off, fx.Int32, align).llvm_ptr,
                         alignment=align,
                         alias_scopes=al,
                         noalias_scopes=noal)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)
        hdim_own = is_v.select(fx.Int32(hdim_vo), fx.Int32(hdim_qk))

        def span_bytes(rows, ss, row_elems):
            span = (fx.Int64(rows) - fx.Int64(1)) * ss + fx.Int64(row_elems)
            return (fx.Int32(rows) > fx.Int32(0)).select(span, fx.Int64(0)) * fx.Int64(2)

        def base_addr(ptr, batch, h, row_off, sb, sh, ss):
            elems = fx.Int64(batch) * sb + fx.Int64(h) * sh + fx.Int64(row_off) * ss
            return fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(2)

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

        def opaque(vals, rc='v'):
            """Opaque copies in register class rc: LLVM can neither look through them nor move them."""
            if const_expr(not vals):
                return []
            irs = [raw(v) for v in vals]
            ty = ir.Type.parse('!llvm.struct<(' + ', '.join(str(x.type) for x in irs) + ')>')
            cons = ','.join([f'={rc}'] * len(irs) + [str(i) for i in range(len(irs))])
            ret = llvm.inline_asm(ty, irs, '', cons, has_side_effects=True)
            return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(len(irs))]

        # ---- The resident B operand: K (K-wave) or V (V-wave), key kvt + 16 sub + lc ----
        k0 = fx.Int32(kv_tile) * fx.Int32(tr.BLOCK_KV)
        kvt = k0 + team * fx.Int32(tr.TEAM_KEYS)
        k_base = base_addr(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq)
        v_base = base_addr(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq)
        r_ss = is_v.select(fx.Int64(stride_v_seq), fx.Int64(stride_k_seq))
        r_rsrc = buffer_ops.create_buffer_resource_from_addr(is_v.select(v_base, k_base),
                                                             num_records_bytes=span_bytes(sk_i32,
                                                                                          r_ss,
                                                                                          is_v.select(row_vo, row_qk)))
        res = []
        for t in range_constexpr(tr.NT):
            row = []
            for sub in range_constexpr(tr.NSUB):
                r = fx.Index(kvt + fx.Int32(16 * sub)) + lc
                c0 = fx.Index(32 * t) + gq * 8
                v8 = buffer_ops.buffer_load(r_rsrc, r * fx.Index(fx.Int32(r_ss)) + c0, vec_width=8, dtype=elem)
                if const_expr(tr.PADDED):
                    # Slack columns may hold NaN, and 0 * NaN is NaN.
                    v8 = masked8(v8, fx.Int32(c0), hdim_own)
                row.append(v8)
            res.append(row)
        if const_expr(tr.KV_PIN):
            flat = opaque([x for row in res for x in row], tr.KV_PIN)
            res = [[flat[t * tr.NSUB + sub] for sub in range(tr.NSUB)] for t in range(tr.NT)]
        res = [[Vec(x) for x in row] for row in res]

        def smin(x, y):
            return (x < y).select(x, y)

        def smax(x, y):
            return (x > y).select(x, y)

        # ---- The visited q range of this key block ----
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
        t_last = t0 + n_tiles - fx.Int32(1)
        n_groups = n_tiles // fx.Int32(tr.PERIOD)

        # The key columns this wave turns into P and dS: sub = 2i + role.
        soft_kv = [kvt + fx.Int32(16) * (fx.Int32(2 * i) + role) + fx.Int32(lc) for i in range(soft_subs)]
        # Key kv sees q rows [kv - win_r, kv - win_r + band_span]: one unsigned compare per element
        # of rel = q - kv + win_r = q0 + band_s[i] + (4gq - lc + r), band_s wave-uniform.
        if const_expr(tr.CAUSAL):
            empty = win_l + win_r < fx.Int32(0)
            band_span = empty.select(fx.Int32(0), win_l + win_r)
            band_s = [empty.select(fx.Int32(1 << 30), win_r - kvt - fx.Int32(16) * (fx.Int32(2 * i) + role))
                      for i in range(soft_subs)]

        # ---- Per-lane LDS offsets ----
        # RM read t of the own tensor: row lc, chunk 4t + gq.
        rm_rd = fx.Index(tr.LDS_RM) + fx.Index(role) * tr.RM_TENSOR + lc * tr.RM_STRIDE + gq * 16
        # Derived reads of d block u: dword 16u + lc (d = 32u + 2lc, +1) of rows 4gq + [0, 4).
        dv_rd = fx.Index(tr.LDS_RM) + fx.Index(role) * tr.RM_TENSOR + gq * (4 * tr.RM_STRIDE) + lc * 4
        xs_at = fx.Index(tr.LDS_XS) + fx.Index(team) * tr.XS_TEAM + lane * 16
        xp_at = fx.Index(tr.LDS_XP) + fx.Index(team) * tr.XP_TEAM + lane * 8

        def slot_base(base, slot):
            """base + slot offset, opaque: hoisted, every slot's bases would stay live in VGPRs."""
            if const_expr(not tr.FRESH_BASES):
                return base + slot * tr.RM_BUF
            v = fx.Int32(base + slot * tr.RM_BUF)
            return fx.Index(fx.Int32(llvm.inline_asm(T.i32, [raw(v)], '', '=v,0', has_side_effects=True)))

        def rm_read(base, slot, t):
            return lds_load(base + 64 * t, v8e(), 16, f'rm{slot}')

        def derived_read(base, slot, u):
            """A operands of d = 32u + 2m and 32u + 2m + 1 (m = lc): four dwords, four v_perm.
            (gfx942's SRAMECC zeroes the other half of a d16 load, so u16 reads pack no cheaper.)"""
            al, noal = scope_attrs(f'rm{slot}')
            w = []
            for i in range_constexpr(4):
                off = base + (i * tr.RM_STRIDE + 64 * u)
                w.append(fx.Int32(llvm.LoadOp(T.i32,
                                              lds_view(off, fx.Int32, 4).llvm_ptr,
                                              alignment=4,
                                              alias_scopes=al,
                                              noalias_scopes=noal).result))
            out = []
            for sel in (sel_lo, sel_hi):
                dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel))]
                out.append(Vec.from_elements(dw, fx.Int32).bitcast(elem).ir_value())
            return out

        sel_lo = fx.Int32(0x05040100)
        sel_hi = fx.Int32(0x07060302)
        side_rd = fx.Index(tr.LDS_SIDE) + gq * 16
        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        m0_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_base + fx.Index(tr.LDS_RM))))
        dma_voff = fx.Int32(lane * 4)
        dma_m0 = m0_base + wave_s * fx.Int32((BLOCK_M // NUM_WAVES) * tr.RM_STRIDE)

        def m0_at(base, const=0):
            """base + const at the use: hoisted, the DMAs' M0 values would pin a dozen SGPRs."""
            return fx.Int32(llvm.inline_asm(T.i32,
                                            [raw(fx.Int32(base))],
                                            f's_add_u32 $0, $1, {const}',
                                            '=s,s,~{scc}',
                                            has_side_effects=True))
        zero4 = fx.full(4, 0.0, fx.Float32)
        if const_expr(dropout):
            rng = Philox.for_arch('gfx942')
            assert rng.randoms_per_offset == 4
            ph_seed = fmha.philox_seed_value(philox_seed_ptr)
            ph_off = fmha.philox_offset_base(philox_offset1, philox_offset2)
            # Lane (lc, gq) draws q row 4gq + lc % 4 for keys 4 (lc // 4) + [0, 4); its quad
            # holds a 4x4 block, and a DPP broadcast deals each lane its key's bits.
            quad_shift = fx.Int32(lc % 4)
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        def pinned():
            """Fix the order of LDS reads and MFMAs; everything else may cross."""
            rocdl.sched_barrier(PIN_MASK)

        def stream(n, ahead_n, read, use, hooks=()):
            """Software-pipelined LDS operands: read t + ahead_n, then use t.
            `hooks` (other work) run spread over the steps, in order."""
            at = {}
            for i, h in enumerate(hooks):
                at.setdefault(min(n - 1, (2 * i + 1) * n // (2 * len(hooks))), []).append(h)
            ahead = [read(t) for t in range(min(ahead_n, n))]
            for t in range_constexpr(n):
                if const_expr(t + ahead_n < n):
                    ahead.append(read(t + ahead_n))
                pinned()
                use(t, ahead[t])
                for h in at.get(t, []):
                    h()
                pinned()

        def set_scopes(op, name):
            al, noal = scope_attrs(name)
            op = getattr(op, 'owner', op)
            op = getattr(op, 'operation', op)
            op.attributes['alias_scopes'] = al
            op.attributes['noalias_scopes'] = noal

        def store_out(accs):
            """Accumulator (2u + e, sub) holds dK^T or dV^T [d = 32u + 8gq + 2r + e][key kvt + 16sub + lc]."""
            dk_base = base_addr(DK, k_batch, head_k, k_row_off, stride_dk_batch, stride_dk_head, stride_dk_seq)
            dv_base = base_addr(DV, k_batch, head_k, k_row_off, stride_dv_batch, stride_dv_head, stride_dv_seq)
            o_ss = is_v.select(fx.Int64(stride_dv_seq), fx.Int64(stride_dk_seq))
            row_lim = is_v.select(row_vo, row_qk)
            rsrc = buffer_ops.create_buffer_resource_from_addr(is_v.select(dv_base, dk_base),
                                                               num_records_bytes=span_bytes(sk_i32, o_ss, row_lim))
            ss = fx.Index(fx.Int32(o_ss))
            # dS is against the scaled scores: sm_scale lands once, here.
            scale = is_v.select(fx.Float32(1.0), fx.Float32(sm_scale))
            sv = fx.full(4, scale, fx.Float32)
            for u in range_constexpr(tr.NDT // 2):
                for sub in range_constexpr(tr.NSUB):
                    ev = Vec(accs[(2 * u) * tr.NSUB + sub]) * sv
                    od = Vec(accs[(2 * u + 1) * tr.NSUB + sub]) * sv
                    vals = [x for r in range(4) for x in (ev[r], od[r])]
                    packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
                    kv = kvt + fx.Int32(16 * sub) + fx.Int32(lc)
                    d = fx.Int32(32 * u) + fx.Int32(gq * 8)
                    for hh in range_constexpr(2):
                        ok = (kv < sk_i32) & (d + fx.Int32(4 * hh) < row_lim)
                        half = packed.shuffle(packed, [4 * hh + x for x in range(4)])
                        buffer_ops.buffer_store(half.ir_value(),
                                                rsrc,
                                                fx.Index(kv) * ss + fx.Index(d + fx.Int32(4 * hh)),
                                                mask=ok)

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
            side_dma_voff = (lane < fx.Index(BLOCK_M)).select(fx.Int32(lane) * pitch_b, fx.Int32(OOB_VOFF))

            if const_expr(bias):
                b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head_q) * stride_b_head
                           + fx.Int64(q_row_off) * stride_b_seq_q)
                b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(k_row_off + sk_i32),
                              fx.Int64(0))
                b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2), num_records_bytes=b_span * fx.Int64(2)))
                b_rs_b = fx.Int32(stride_b_seq_q) * fx.Int32(2)
                b_voff = [fx.Int32(gq * 4) * b_rs_b + (k_row_off + kv) * fx.Int32(2) for kv in soft_kv]

            if const_expr(dropout):
                ph_plane = z_i32 * fx.Int32(num_head_q) + fx.Int32(head_q)
                ph_base, ph_stride = rng.grid_plane(ph_off, ph_plane, max_seqlen_q, max_seqlen_k)

            def q0_of(j):
                if const_expr(tr.Q_DOWN):
                    return (t_last - j) * fx.Int32(BLOCK_M)
                return (t0 + j) * fx.Int32(BLOCK_M)

            def dma_rows(j, part, slot):
                """Part `part` of RM(j) -> slot as one callable per row: wave w copies rows
                4w..4w+3 of Q and of dO."""
                q0 = q0_of(j)
                out = []
                for tensor in range_constexpr(2):
                    for i in range_constexpr(BLOCK_M // NUM_WAVES):
                        def row_dma(tensor=tensor, i=i):
                            rsrc = do_rsrc if tensor else q_rsrc
                            rs_b = do_rs_b if tensor else q_rs_b
                            row = wave_s * fx.Int32(BLOCK_M // NUM_WAVES) + fx.Int32(i)
                            # The immediate offset moves the LDS address too; the part's column rides in soffset.
                            soff = (q0 + row) * rs_b + fx.Int32(part * tr.PART_BYTES)
                            m0 = fx.Int64(m0_at(dma_m0, slot * tr.RM_BUF + tensor * tr.RM_TENSOR + i * tr.RM_STRIDE))
                            for seg in tr.SEGS:
                                set_scopes(rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                                         lds_ptr_at(m0),
                                                                         fx.Int32(4),
                                                                         dma_voff,
                                                                         soff,
                                                                         fx.Int32(seg),
                                                                         fx.Int32(0)),
                                           f'rm{slot}')
                        out.append(row_dma)
                return out

            def side_dma(j, sslot):
                """LSE(j) and Delta(j) -> side slot, 256B each; lane L < 16 copies row q0 + L."""
                soff = q0_of(j) * pitch_b
                for i, rsrc in enumerate((lse_rsrc, delta_rsrc)):
                    m0 = fx.Int64(m0_at(m0_base, tr.LDS_SIDE - tr.LDS_RM + sslot * SIDE_SLOT + 256 * i))
                    set_scopes(rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                             lds_ptr_at(m0),
                                                             fx.Int32(4),
                                                             side_dma_voff,
                                                             soff,
                                                             fx.Int32(0),
                                                             fx.Int32(0)),
                               f's{sslot}')

            def side_read(sslot):
                """LSE and Delta of q = q0 + 4gq + r."""
                v4i = Vec.make_type(4, fx.Int32)
                lse = Vec(lds_load(side_rd + sslot * SIDE_SLOT, v4i, 16, f's{sslot}')).bitcast(fx.Float32)
                dlt = Vec(lds_load(side_rd + (sslot * SIDE_SLOT + 256), v4i, 16, f's{sslot}')).bitcast(fx.Float32)
                return [lse[r] for r in range(4)], [dlt[r] for r in range(4)]

            def bias_load(j):
                """Bias of tile j at q = q0 + 4gq + r for the softmax columns: one i16 each."""
                if const_expr(not bias):
                    return []
                bsoff = q0_of(j) * b_rs_b
                out = []
                for i in range_constexpr(soft_subs):
                    for r in range_constexpr(4):
                        h = rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, b_voff[i] + fx.Int32(r) * b_rs_b, bsoff, 0)
                        out.append(fx.Int32(fx.Int16(h)))
                return out

            def keep_bits(q0, i):
                """Dropout keep of (q0 + 4gq + r, soft_kv[i]) for r < 4, as four i1."""
                kv4 = soft_kv[i] - fx.Int32(lc % 4)
                vals = rng.u32(ph_seed, rng.grid_offset(ph_base, ph_stride, q0 + fx.Int32(gq * 4 + lc % 4), kv4))
                bits = fx.Int32(0)
                for s in range_constexpr(4):
                    bits = bits | (fx.Int32(vals[s]) > fx.Int32(idropout_p)).select(fx.Int32(1 << s), fx.Int32(0))
                word = fx.Int32(0)
                for r in range_constexpr(4):
                    got = fx.Int32(rocdl.update_dpp(T.i32, raw(fx.Int32(0)), raw(bits), 0x55 * r, 0xF, 0xF, True))
                    word = word | (got << fx.Int32(4 * r))
                word = word >> quad_shift
                return [(word & fx.Int32(1 << (4 * r))) != fx.Int32(0) for r in range(4)]

            def softmax(i, s_raw, dp_raw, lse, delta, bvals, q0, masked):
                """P (dropped, for dV) and dS packs of key column soft_kv[i]."""
                sv = Vec(s_raw) * fx.full(4, fx.Float32(c_scale), fx.Float32)
                dpv = Vec(dp_raw)
                if const_expr(masked):
                    # The lane term afresh per tile: hoisted, it holds a VGPR through the loop.
                    lt = fx.Int32(lane // 16) * fx.Int32(4) - fx.Int32(lane % 16)
                    lt = fx.Int32(llvm.inline_asm(T.i32, [raw(lt)], '', '=v,0', has_side_effects=True))
                    rel0 = q0 + band_s[i] + lt
                if const_expr(dropout):
                    keep = keep_bits(q0, i)
                ps = []
                dss = []
                for r in range_constexpr(4):
                    x = sv[r]
                    if const_expr(bias):
                        bv = Vec.from_elements([fx.Int16(bvals[4 * i + r])], fx.Int16).bitcast(elem)
                        # Upcast first: log2(e) in the input dtype would differ from the forward's.
                        x = x + fx.Float32(Vec(bv)[0].to(fx.Float32)) * fx.Float32(LOG2E)
                    p = exp2(x - fx.Float32(lse[r]) * fx.Float32(LOG2E))
                    if const_expr(masked):
                        live = fx.Uint32(rel0 + fx.Int32(r)) <= fx.Uint32(band_span)
                        p = live.select(p, fx.Float32(0.0))
                    dpr = fx.Float32(dpv[r])
                    p_drop = p
                    if const_expr(dropout):
                        p_drop = keep[r].select(p * fx.Float32(dropout_scale), fx.Float32(0.0))
                        dpr = keep[r].select(dpr * fx.Float32(dropout_scale), fx.Float32(0.0))
                    ps.append(p_drop)
                    dss.append(p * (dpr - fx.Float32(delta[r])))
                return pack4(ps), pack4(dss)

            def tile(j, carried, ph, masked):
                """X, Z, Y of tile j; ph = j % PERIOD (trace time) picks the RM slots."""
                accs = list(carried[:n_acc])
                bvals = carried[n_acc:]
                q0 = q0_of(j)
                j1 = j + fx.Int32(1)
                nxt = {}

                def slot(p, h):
                    return (tr.PARTS * p + h) % tr.RM_SLOTS

                def side_next():
                    side_dma(j1, (ph + 1) % tr.PERIOD)
                    nxt['bias'] = bias_load(j1)
                n_side = 2 + (4 * soft_subs if bias else 0)

                # X: S or dP of tile j, part by part. Part 0 also DMAs part 0 of RM(j+1)
                # and loads LSE/Delta/bias(j+1); between parts, part 1 of RM(j) must land.
                own = [zero4] * tr.NSUB
                for h in range_constexpr(tr.PARTS):
                    sl_x = slot(ph, h)
                    hooks = []
                    if const_expr(h == 0):
                        hooks = dma_rows(j1, 0, slot(ph + 1, 0)) + [side_next]

                    x_base = slot_base(rm_rd, sl_x)

                    def x_read(t, sl_x=sl_x, x_base=x_base):
                        x8 = rm_read(x_base, sl_x, t)
                        if const_expr(tr.OPS_VGPR):
                            # Into VGPRs: in AGPRs the operands crowd out the accumulators.
                            x8 = llvm.inline_asm(x8.type, [x8], '', '=v,0', has_side_effects=False)
                        return Vec(x8)

                    def x_use(t, x8, h=h):
                        tg = h * tr.NT_P + t
                        if const_expr(tr.PADDED):
                            x8 = Vec(masked8(x8, fx.Int32(32 * tg) + fx.Int32(gq * 8), hdim_own))
                        for e in range_constexpr(2):
                            sl = [4 * e, 4 * e + 1, 4 * e + 2, 4 * e + 3]
                            xa = x8.shuffle(x8, sl).ir_value()
                            for sub in range_constexpr(tr.NSUB):
                                rb = res[tg][sub]
                                own[sub] = mfma(xa, rb.shuffle(rb, sl).ir_value(), own[sub])

                    stream(tr.NT_P, tr.PREFETCH, x_read, x_use, hooks)
                    if const_expr(h + 1 < tr.PARTS):
                        # Everything this part issued is newer than part 1 of RM(j) (Y(j-1)).
                        rocdl.s_waitcnt(vmcnt=tr.N_DMA + n_side)
                        cluster_barrier()
                # Raw S (K-wave) or dP (V-wave) of every column to the exchange: [sub][role].
                for sub in range_constexpr(tr.NSUB):
                    lds_store(xs_at + fx.Index(role) * (WAVE * 16) + sub * XS_SUB, own[sub], 16, 'xs')
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()

                # Z: P and dS of columns 2i + role; both packs to the exchange: [sub][P, dS].
                lse, delta = side_read(ph)
                for i in range_constexpr(soft_subs):
                    sub_off = (fx.Index(2 * i) + fx.Index(role)) * XS_SUB
                    s_raw = lds_load(xs_at + sub_off, v4f(), 16, 'xs')
                    dp_raw = lds_load(xs_at + sub_off + WAVE * 16, v4f(), 16, 'xs')
                    p_pk, ds_pk = softmax(i, s_raw, dp_raw, lse, delta, bvals, q0, masked)
                    xo = xp_at + (fx.Index(2 * i) + fx.Index(role)) * XP_SUB
                    lds_store(xo, Vec(p_pk).bitcast(fx.Int32), 8, 'xp')
                    lds_store(xo + WAVE * 8, Vec(ds_pk).bitcast(fx.Int32), 8, 'xp')
                rocdl.s_waitcnt(lgkmcnt=0)
                cluster_barrier()

                # Y: dK^T (K-wave, B = dS) or dV^T (V-wave, B = P) += tile j, part by part.
                # Once part 0's slot is free, part 1 of RM(j+1) DMAs into it.
                pk_off = (fx.Index(1) - fx.Index(role)) * (WAVE * 8)
                pks = [lds_load(xp_at + (sub * XP_SUB) + pk_off, v4e(), 8, 'xp') for sub in range(tr.NSUB)]
                for h in range_constexpr(tr.PARTS):
                    sl_y = slot(ph, h)
                    hooks = []
                    if const_expr(h == 1):
                        hooks = dma_rows(j1, 1, slot(ph + 1, 1))

                    def y_use(u, ab, h=h):
                        for e in range_constexpr(2):
                            for sub in range_constexpr(tr.NSUB):
                                k = (h * tr.NDT_P + 2 * u + e) * tr.NSUB + sub
                                accs[k] = mfma(ab[e], pks[sub], accs[k])

                    y_base = slot_base(dv_rd, sl_y)
                    stream(tr.NDT_P // 2,
                           tr.Y_AHEAD,
                           lambda u, sl_y=sl_y, y_base=y_base: derived_read(y_base, sl_y, u),
                           y_use,
                           hooks)
                    if const_expr(h + 1 == tr.PARTS):
                        # Part 0 of RM(j+1) (and LSE/Delta/bias) landed; part 1 may still fly.
                        rocdl.s_waitcnt(vmcnt=tr.N_DMA if tr.PARTS == 2 else 0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    cluster_barrier()
                return accs + nxt['bias']

            n_acc = tr.NDT * tr.NSUB

            @flyc.jit
            def tail_a(j, carried):
                out = list(carried)
                if j < fx.Index(n_tiles):
                    out = tile(fx.Int32(j), carried, 0, tr.CAUSAL)
                return out

            @flyc.jit
            def tail_b(j, carried):
                out = list(carried)
                if j < fx.Index(n_tiles):
                    out = tile(fx.Int32(j), carried, 1, tr.CAUSAL)
                return out

            # Prologue: RM(0) DMA (both parts), LSE/Delta/bias(0).
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            for h in range_constexpr(tr.PARTS):
                for f in dma_rows(fx.Int32(0), h, h):
                    f()
            side_dma(fx.Int32(0), 0)
            b0 = bias_load(fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()

            init = list(accs) + b0
            res1 = init
            for jp, carried in range(0, fx.Index(n_groups), 1, init=init):
                jg = fx.Int32(jp) * fx.Int32(tr.PERIOD)
                cur = carried
                for p in range_constexpr(tr.PERIOD - 1):
                    cur = tile(jg + fx.Int32(p), cur, p, tr.CAUSAL)
                res1 = yield tile(jg + fx.Int32(tr.PERIOD - 1), cur, tr.PERIOD - 1, tr.CAUSAL)
            j_rest = fx.Index(n_groups) * tr.PERIOD
            out = tail_a(j_rest, res1)
            if const_expr(tr.PERIOD == 3):
                out = tail_b(j_rest + fx.Index(1), out)
            return out[:n_acc]

        n_acc = tr.NDT * tr.NSUB
        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range(n_acc)]
        results = acc0
        for g, carried in range(0, fx.Index(group), 1, init=acc0):
            results = yield head_pass(fx.Int32(g), carried)
        rocdl.s_waitcnt(vmcnt=0)
        store_out(results)

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx942_xwide(
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
        # Scalars, not traits fields: the JIT cache keys on scalar closure values.
        _ = cache_tag
        grid_x = (fx.Index(max_seqlen_k) + block_kv - 1) // block_kv
        grid_z = (fx.Int32(num_seqlens) != fx.Int32(0)).select(fx.Index(num_seqlens), fx.Index(batch_size))
        fmha_bwd_dkdv_gfx942_xwide_kernel(Q,
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
                         launch_fmha_bwd_dkdv_gfx942_xwide,
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
