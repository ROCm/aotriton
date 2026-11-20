# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dK/dV backward for gfx942 at head dims 384 and 512, FlyDSL: role-split teams
that also shard the contraction (gfx1201's contraction_shards = 2).

One team of four waves owns 16 * KV_SUBS keys. Wave (shard c, role r) holds K
(r = 0) or V (r = 1), and dK^T or dV^T, only for the d columns [c, c + 1) * D/2,
so a wave keeps 0.1875 * D * KV_SUBS registers of state (the xwide body: 0.375).
S and dP are sums of the two shards' partials, reduced through LDS:

    K-wave c: S_c  = Q[:, c] . K[:, c]^T       dK^T[c] += Q^T[c] . dS
    V-wave c: dP_c = dO[:, c] . V[:, c]^T      dV^T[c] += dO^T[c] . P

A wave reads only its own tensor's half (Q for K-waves, dO for V-waves), so it
loads it itself, whole rows a 16B load, into an LDS area no other wave reads: X's
row-major operands and Y's transposed ones (four dword reads, four v_perm) come
back from it without a barrier. Per 16-row q tile j, two barrier-separated phases:
    A(j): S, dP of tile j summed from the exchange; P(j), dS(j) of KV_SUBS
          (column, row group) items a wave (at 3, a column a wave) to the exchange;
          partial S or dP of tile j+1 from the own area; tile j+2 loads
    B(j): partials of j+1 to the exchange; dK^T or dV^T += tile j from the derived
          operands read in B(j-1); tile j+1's derived reads; tile j+2 to the own area
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, ktrace, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops

import fmha_utils as fmha
from . import tuning_dkdv as tuning
from .utils import (
    LOG2E,
    WAVE,
    Lds,
    alias_scopes,
    apply_mask,
    base_addr,
    cluster_barrier,
    col_mask,
    deal_keep_bits,
    exp2,
    mfma_16x16x16,
    pack_operand,
    philox_plane,
    philox_stream,
    pin_values,
    pinned,
    smax,
    smin,
    span_bytes,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942_xxwide.lds">'
LDS_SCOPES = ('own', 'xs', 'xp')
NUM_WAVES = 4
BLOCK_M = 16            # q rows per tile: the N of the 16x16x16 MFMA
SHARDS = 2              # contraction shards: d halves
XS_SUB = 2 * SHARDS * WAVE * 16     # partial S and dP of one key column, 16B per lane each
XP_SUB = 2 * WAVE * 8   # P and dS packs of one key column, 8B per lane each
# Dev sweeps only: Traits field -> value, applied to every build (and its cache tag).
DEV_OVERRIDES = {}


class Traits:
    """Compile-time shape of one kernel."""

    def __init__(self, head_dim, causal, dtype_str, kv_subs, padded_head=False):
        assert head_dim % 64 == 0 and 128 <= head_dim <= 512, head_dim
        assert kv_subs in (2, 3, 4), kv_subs
        self.D = head_dim
        self.PADDED = padded_head
        self.CAUSAL = causal
        self.DTYPE = dtype_str
        self.NSUB = kv_subs                     # 16-key MFMA columns of the team
        self.BLOCK_KV = 16 * kv_subs
        self.BLOCK_SIZE = NUM_WAVES * WAVE
        self.NT = head_dim // 32                # 32-column d blocks: one 16B operand read, two k-steps
        # Softmax items (column, row r) per wave and tile: KV_SUBS of them, or at 3 a column
        # each, the fourth wave on a spare column.
        self.NI = 4 if kv_subs == 3 else kv_subs
        self.NZ = 4 if kv_subs == 3 else kv_subs   # exchange columns
        self.NB = self.NT // SHARDS             # own blocks
        self.Z_AT = 99                          # the X step of A(j) that Z(j) follows (99: last)
        self.W_AT = 0                           # the X step of A(j) the tile loads start after
        # Pin the resident K or V to VGPRs ('v') or AGPRs ('a'); '' leaves LLVM free. 512 at
        # KV_SUBS 3: 'v' spills, 'a' and '' do not ('' 424 vs 418 TFLOPS).
        self.KV_PIN = ''
        # Walk q tiles last to first: a causal head's blocks then all start on its last
        # tiles and share them in L2, instead of each starting at its own diagonal.
        self.Q_DOWN = causal
        self.TRACE = False                      # dev: ktrace marks per phase
        self.EXP = ''                           # dev experiments (wrong results): 'nosm', 'nobar', 'nox', 'noy', 'noder'
        for k, v in DEV_OVERRIDES.items():
            assert hasattr(self, k), k
            setattr(self, k, v)
        # Each wave's own area: its tensor's 16 rows, its half of the columns.
        self.OWN_STRIDE = self._stride(self.NB * 32)
        self.OWN_BYTES = BLOCK_M * self.OWN_STRIDE
        self.CPR = head_dim // 16               # 16B chunks of a row's own half
        self.N_LD = BLOCK_M * self.CPR // WAVE  # 16B loads a lane per tile
        self.LDS_OWN = 0
        self.LDS_XS = NUM_WAVES * self.OWN_BYTES
        self.LDS_XP = self.LDS_XS + self.NZ * XS_SUB
        self.LDS_BYTES = self.LDS_XP + self.NZ * XP_SUB
        assert self.LDS_BYTES <= 65536, self.LDS_BYTES
        self.CACHE_TAG = (f'BXX{head_dim}_c{int(causal)}_{dtype_str}_s{kv_subs}_p{int(padded_head)}'
                          f'_z{self.Z_AT}.{self.W_AT}_k{self.KV_PIN}_d{int(self.Q_DOWN)}_e{self.EXP}_t{int(self.TRACE)}')

    @staticmethod
    def _stride(cols):
        """Rows an odd number of 16B chunks apart: conflict-free b128 reads, and the
        derived reads' four row groups (4 rows apart) land on distinct bank halves."""
        chunks = cols // 8
        return (chunks + 1 - chunks % 2) * 16


def lds_bytes(head_dim, kv_subs):
    """LDS of a build, for the tuning module's legality check."""
    return Traits(head_dim, False, 'bf16', kv_subs).LDS_BYTES


def build_fmha_bwd_dkdv_gfx942_xxwide(head_dim,
                                     causal=False,
                                     causal_type=None,
                                     dtype_str='bf16',
                                     bias=False,
                                     dropout=False,
                                     kv_subs=None,
                                     waves_per_eu=1,
                                     grid_axis_order=None,
                                     padded_head=False):
    """The gfx1201 dK/dV kernel's ABI and `_launch` interface, for the widest heads."""
    host_causal_type, causal_type = fmha.builder_causal_types(causal, causal_type, bias)
    if kv_subs is None:
        kv_subs = tuning.body_kv_subs('xxwide', head_dim)
    tr = Traits(head_dim, causal_type != 0, dtype_str, kv_subs, padded_head)
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(tr.CAUSAL)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    cache_tag = f'{tr.CACHE_TAG}_wpe{waves_per_eu}_g{grid_axis_order}_b{int(bias)}_d{int(dropout)}'
    elem = fx.BFloat16 if dtype_str == 'bf16' else fx.Float16
    block_kv = tr.BLOCK_KV

    def v4e():
        return Vec.make_type(4, elem)

    def v8e():
        return Vec.make_type(8, elem)

    @fx.struct
    class Smem:
        buf: fx.Array[fx.Int8, tr.LDS_BYTES, 16]

    def mfma(a, b, c):
        return mfma_16x16x16(a, b, c, elem)

    def pack4(vals):
        return pack_operand(vals, elem)

    def pack_items(vals):
        """NI f32 -> the bits of NI elements: i32 or <2 x i32>; bf16 rounds as pack4."""
        if const_expr(len(vals) == 4):
            return Vec(pack4(vals)).bitcast(fx.Int32)
        if const_expr(tr.DTYPE == 'bf16'):
            rnd = [fx.Float32(x).bitcast(fx.Int32) + fx.Int32(0x8000) for x in vals]
            return fx.Int32(rocdl.perm_b32(rnd[1], rnd[0], fx.Int32(0x07060302)))
        return Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem).bitcast(fx.Int32)[0]

    def scope_attrs(name):
        return alias_scopes(_LDS_DOMAIN, LDS_SCOPES, name)

    def trace(name):
        """Under FLYDSL_KTRACE_ENABLE=1 (or a ktrace capture), a timestamp."""
        rocdl.sched_barrier(0)
        ktrace.mark(name)
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
        shard = wave_s >> fx.Int32(1)           # own columns: [shard, shard + 1) * D / 2
        is_v = role == fx.Int32(1)
        # Softmax items (column z_sub, rows 4gq + z_r0 + [0, NI)): wave w takes items w * NI + [0, NI).
        z_sub = (wave_s * fx.Int32(tr.NI)) >> fx.Int32(2)
        z_r0 = (wave_s * fx.Int32(tr.NI)) & fx.Int32(3)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_base = fx.Index(fx.ptrtoint(lds_p))

        lds_io = Lds(lds_p, elem, _LDS_DOMAIN, LDS_SCOPES)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)
        hdim_own = is_v.select(fx.Int32(hdim_vo), fx.Int32(hdim_qk))

        def masked8(v8, c0, hdim):
            return apply_mask(v8, col_mask(c0, hdim), elem)

        # ---- The resident B operand: K (K-wave) or V (V-wave) of the own half, key kvt + 16 sub + lc ----
        k0 = fx.Int32(kv_tile) * fx.Int32(tr.BLOCK_KV)
        kvt = k0
        k_base = base_addr(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq)
        v_base = base_addr(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq)
        r_ss = is_v.select(fx.Int64(stride_v_seq), fx.Int64(stride_k_seq))
        r_rsrc = buffer_ops.create_buffer_resource_from_addr(is_v.select(v_base, k_base),
                                                             num_records_bytes=span_bytes(sk_i32,
                                                                                          r_ss,
                                                                                          is_v.select(row_vo, row_qk)))
        res = []
        for t in range_constexpr(tr.NB):
            row = []
            for sub in range_constexpr(tr.NSUB):
                r = fx.Index(kvt + fx.Int32(16 * sub)) + lc
                c0 = fx.Index(32 * t) + fx.Index(shard) * (tr.D // 2) + gq * 8
                v8 = buffer_ops.buffer_load(r_rsrc, r * fx.Index(fx.Int32(r_ss)) + c0, vec_width=8, dtype=elem)
                if const_expr(tr.PADDED):
                    # Slack columns may hold NaN, and 0 * NaN is NaN.
                    v8 = masked8(v8, fx.Int32(c0), hdim_own)
                row.append(v8)
            res.append(row)
        if const_expr(tr.KV_PIN):
            flat = pin_values([x for row in res for x in row], tr.KV_PIN)
            res = [[flat[t * tr.NSUB + sub] for sub in range(tr.NSUB)] for t in range(tr.NB)]
        res = [[Vec(x) for x in row] for row in res]

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
        n_pairs = n_tiles // fx.Int32(2)

        # The key this lane turns into P and dS.
        soft_kv = kvt + fx.Int32(16) * z_sub + fx.Int32(lc)
        # Key kv sees q rows [kv - win_r, kv - win_r + band_span]: one unsigned compare per element
        # of rel = q - kv + win_r = q0 + band_s + (4gq - lc + r), band_s wave-uniform.
        if const_expr(tr.CAUSAL):
            empty = win_l + win_r < fx.Int32(0)
            band_span = empty.select(fx.Int32(0), win_l + win_r)
            band_s = empty.select(fx.Int32(1 << 30), win_r - kvt - fx.Int32(16) * z_sub + z_r0)

        # ---- Per-lane LDS offsets ----
        # Own area: X's operand of block b is row lc, chunk 4b + gq; the derived reads of
        # block b are dword 16b + lc (d = D/2 shard + 32b + 2lc, +1) of rows 4gq + [0, 4).
        own_base = fx.Index(tr.LDS_OWN) + fx.Index(wave) * tr.OWN_BYTES
        own_x = own_base + lc * tr.OWN_STRIDE + gq * 16
        own_rd = own_base + gq * (4 * tr.OWN_STRIDE) + lc * 4
        # Exchange: partial S / dP at [sub][2 shard + role][lane]; P, dS at [sub][P, dS][lane].
        xs_at = fx.Index(tr.LDS_XS) + lane * 16
        xp_at = fx.Index(tr.LDS_XP) + lane * 8

        def derived_load(b):
            """The four dwords of own block b at d = 2m and 2m + 1 (m = lc), rows 4gq + [0, 4).
            (gfx942's SRAMECC zeroes the other half of a d16 load, so u16 reads pack no cheaper.)"""
            al, noal = scope_attrs('own')
            w = []
            for i in range_constexpr(4):
                off = own_rd + (i * tr.OWN_STRIDE + 64 * b)
                w.append(fx.Int32(llvm.LoadOp(T.i32,
                                              lds_io.view(off, fx.Int32, 4).llvm_ptr,
                                              alignment=4,
                                              alias_scopes=al,
                                              noalias_scopes=noal).result))
            return w

        def derived_ops(w):
            """derived_load's dwords -> the A operands of d = 2m and 2m + 1: four v_perm."""
            out = []
            for sel in (sel_lo, sel_hi):
                dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel))]
                out.append(Vec.from_elements(dw, fx.Int32).bitcast(elem).ir_value())
            return out

        sel_lo = fx.Int32(0x05040100)
        sel_hi = fx.Int32(0x07060302)
        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        zero4 = fx.full(4, 0.0, fx.Float32)
        if const_expr(dropout):
            rng, ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            assert rng.randoms_per_offset == 4
            # Lane (lc, gq) draws q row 4gq + lc % 4 for keys 4 (lc // 4) + [0, 4); its quad
            # holds a 4x4 block, and a DPP broadcast deals each lane its key's bits.
            quad_shift = fx.Int32(lc % 4) + z_r0 * fx.Int32(4)
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        def store_out(accs):
            """Accumulator (2b + e, sub) holds dK^T or dV^T [d = D/2 shard + 32b + 8gq + 2r + e][key kvt + 16sub + lc]."""
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
            for u in range_constexpr(tr.NB):
                for sub in range_constexpr(tr.NSUB):
                    ev = Vec(accs[(2 * u) * tr.NSUB + sub]) * sv
                    od = Vec(accs[(2 * u + 1) * tr.NSUB + sub]) * sv
                    vals = [x for r in range(4) for x in (ev[r], od[r])]
                    packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
                    kv = kvt + fx.Int32(16 * sub) + fx.Int32(lc)
                    d = fx.Int32(32 * u) + shard * fx.Int32(tr.D // 2) + fx.Int32(gq * 8)
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
            # The own tensor: Q (K-wave) or dO (V-wave).
            o_rs_b = is_v.select(fx.Int32(stride_do_seq), fx.Int32(stride_q_seq)) * fx.Int32(2)
            o_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(is_v.select(do_base, q_base),
                                                                                       num_records_bytes=is_v.select(
                                                                                           do_span,
                                                                                           q_span)))
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
            side_voff = (fx.Int32(gq * 4) + z_r0) * pitch_b

            if const_expr(bias):
                b_elems = (fx.Int64(q_batch) * stride_b_batch + fx.Int64(head_q) * stride_b_head
                           + fx.Int64(q_row_off) * stride_b_seq_q)
                b_span = smax(fx.Int64(sq_i32 - fx.Int32(1)) * stride_b_seq_q + fx.Int64(k_row_off + sk_i32),
                              fx.Int64(0))
                b_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(fx.ptrtoint(B)) + b_elems * fx.Int64(2), num_records_bytes=b_span * fx.Int64(2)))
                b_rs_b = fx.Int32(stride_b_seq_q) * fx.Int32(2)
                b_voff = (fx.Int32(gq * 4) + z_r0) * b_rs_b + (k_row_off + soft_kv) * fx.Int32(2)

            if const_expr(dropout):
                ph_base, ph_stride = philox_plane(rng, ph_off, z_i32, num_head_q, head_q, max_seqlen_q, max_seqlen_k)

            def q0_of(j):
                if const_expr(tr.Q_DOWN):
                    return (t_last - j) * fx.Int32(BLOCK_M)
                return (t0 + j) * fx.Int32(BLOCK_M)

            # Load i of a lane: 16B chunk c = 64i + lane of the wave's half-rows, whole rows a load.
            r_row = [fx.Int32((fx.Index(WAVE * i) + lane) // tr.CPR) for i in range(tr.N_LD)]
            r_col = [fx.Int32((fx.Index(WAVE * i) + lane) % tr.CPR) for i in range(tr.N_LD)]
            r_voff = [r_row[i] * o_rs_b + r_col[i] * fx.Int32(16) + shard * fx.Int32(tr.D) for i in range(tr.N_LD)]

            def x_loads(j, out):
                """Own tile j into out (N_LD v8), one callable a load."""
                def ld(i):
                    v = rocdl.raw_ptr_buffer_load(v8e(), o_rsrc, r_voff[i], q0_of(j) * o_rs_b, 0)
                    if const_expr(tr.PADDED):
                        # Slack columns may hold NaN, and 0 * NaN is NaN.
                        v = masked8(v, r_col[i] * fx.Int32(8) + shard * fx.Int32(tr.D // 2), hdim_own)
                    out[i] = v
                out.extend([None] * tr.N_LD)
                return [lambda i=i: ld(i) for i in range(tr.N_LD)]

            def own_writes(xop):
                """The own tile to its LDS area, one callable a load."""
                def wr(i):
                    at = own_base + fx.Index(r_row[i]) * tr.OWN_STRIDE + fx.Index(r_col[i]) * 16
                    lds_io.store(at, Vec(xop[i]).bitcast(fx.Int32), 16, 'own')
                return [lambda i=i: wr(i) for i in range(tr.N_LD)]

            def x_operands():
                """X's A operands of own block b: row lc, chunk 4b + gq."""
                return [lds_io.load(own_x + 64 * b, v8e(), 16, 'own') for b in range(tr.NB)]

            def side_load(j):
                """LSE and Delta of q = q0 + 4gq + z_r0 + i, i < NI, as f32."""
                soff = q0_of(j) * pitch_b
                out = []
                for rsrc in (lse_rsrc, delta_rsrc):
                    for i in range_constexpr(tr.NI):
                        out.append(fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type,
                                                                      rsrc,
                                                                      side_voff + fx.Int32(i) * pitch_b,
                                                                      soff,
                                                                      0)))
                return out

            def bias_load(j):
                """Bias of tile j at the softmax items: one i16 each."""
                if const_expr(not bias):
                    return []
                bsoff = q0_of(j) * b_rs_b
                out = []
                for i in range_constexpr(tr.NI):
                    h = rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, b_voff + fx.Int32(i) * b_rs_b, bsoff, 0)
                    out.append(fx.Int32(fx.Int16(h)))
                return out

            def keep_bits(q0):
                """Dropout keep of (q0 + 4gq + z_r0 + i, soft_kv) for i < NI, as i1."""
                kv4 = soft_kv - fx.Int32(lc % 4)
                row = q0 + fx.Int32(gq * 4 + lc % 4)
                return deal_keep_bits(rng, ph_seed, ph_base, ph_stride, row, kv4, idropout_p, quad_shift, tr.NI)

            def softmax(s_raw, dp_raw, lse, delta, bvals, q0, masked):
                """P (dropped, for dV) and dS of the softmax items, f32."""
                if const_expr(masked):
                    # The lane term afresh per tile: hoisted, it holds a VGPR through the loop.
                    lt = fx.Int32(lane // 16) * fx.Int32(4) - fx.Int32(lane % 16)
                    lt = fx.Int32(pin_values([lt])[0])
                    rel0 = q0 + band_s + lt
                if const_expr(dropout):
                    keep = keep_bits(q0)
                ps = []
                dss = []
                for r in range_constexpr(tr.NI):
                    x = fx.Float32(s_raw[r]) * fx.Float32(c_scale)
                    if const_expr(bias):
                        bv = Vec.from_elements([fx.Int16(bvals[r])], fx.Int16).bitcast(elem)
                        # Upcast first: log2(e) in the input dtype would differ from the forward's.
                        x = x + fx.Float32(Vec(bv)[0].to(fx.Float32)) * fx.Float32(LOG2E)
                    p = exp2(x - fx.Float32(lse[r]) * fx.Float32(LOG2E))
                    if const_expr(masked):
                        live = fx.Uint32(rel0 + fx.Int32(r)) <= fx.Uint32(band_span)
                        p = live.select(p, fx.Float32(0.0))
                    dpr = fx.Float32(dp_raw[r])
                    p_drop = p
                    if const_expr(dropout):
                        p_drop = keep[r].select(p * fx.Float32(dropout_scale), fx.Float32(0.0))
                        dpr = keep[r].select(dpr * fx.Float32(dropout_scale), fx.Float32(0.0))
                    ps.append(p_drop)
                    dss.append(p * (dpr - fx.Float32(delta[r])))
                return ps, dss

            def x_steps(xop, own):
                """Partial S or dP of a tile from its own blocks xop into own, one step a block."""
                def x_use(b):
                    if const_expr('nox' in tr.EXP):
                        return
                    x8 = Vec(xop[b])
                    for e in range_constexpr(2):
                        sl = [4 * e, 4 * e + 1, 4 * e + 2, 4 * e + 3]
                        xa = x8.shuffle(x8, sl).ir_value()
                        for sub in range_constexpr(tr.NSUB):
                            rb = res[b][sub]
                            own[sub] = mfma(xa, rb.shuffle(rb, sl).ir_value(), own[sub])
                return [lambda b=b: x_use(b) for b in range(tr.NB)]

            def derived_hooks(out):
                """Derived reads of every own block into out, one callable a block."""
                def rd(b):
                    if const_expr('noder' in tr.EXP):
                        out[b] = [fx.Int32(lane) + fx.Int32(i) for i in range(4)]
                        return
                    out[b] = derived_load(b)
                out.extend([None] * tr.NB)
                return [lambda b=b: rd(b) for b in range(tr.NB)]

            def write_partials(own):
                for sub in range_constexpr(tr.NSUB):
                    lds_io.store(xs_at + fx.Index(wave_s) * (WAVE * 16) + sub * XS_SUB, own[sub], 16, 'xs')

            def run(steps, hooks):
                """steps (callables) with hooks[i] after step i; LDS reads and MFMAs keep their order."""
                for t in range_constexpr(len(steps)):
                    pinned()
                    steps[t]()
                    for h in hooks.get(t, []):
                        h()
                    pinned()

            def spread(hooks, n, work, first=0):
                """work over steps first..n-1, evenly."""
                for i, f in enumerate(work):
                    hooks.setdefault(min(n - 1, first + i * (n - first) // max(1, len(work))), []).append(f)
                return hooks

            def phase_end(name=None):
                """LDS writes done, then the barrier; the tile loads stay in flight."""
                if const_expr(name is not None):
                    trace(name)
                rocdl.s_waitcnt(lgkmcnt=0)
                if const_expr('nobar' in tr.EXP):
                    return
                cluster_barrier()
                if const_expr(name is not None):
                    trace(name + 'b')

            n_bias = tr.NI if bias else 0

            def unpack(c):
                o = n_acc
                accs = list(c[:o])
                dvd = [list(c[o + 4 * b:o + 4 * b + 4]) for b in range(tr.NB)]
                o += 4 * tr.NB
                side = list(c[o:o + 2 * tr.NI])
                o += 2 * tr.NI
                bv = list(c[o:o + n_bias])
                o += n_bias
                return accs, dvd, side, bv

            def refill(xop):
                """Tile xop to the own area, after the derived reads of the tile before (the
                same wave's LDS ops run in order)."""
                if const_expr('nowr' not in tr.EXP):
                    for f in own_writes(xop):
                        f()

            def iteration(j, carried, masked):
                """A(j): Z(j), X(j+1); B(j): Y(j), derived reads of tile j+1, tile j+2 to the own area.
                Carried: the accumulators, tile j's derived dwords, LSE/Delta and bias items."""
                accs, dvd, side, bvals = unpack(carried)
                q0 = q0_of(j)
                lse = [fx.Int32(x).bitcast(fx.Float32) for x in side[:tr.NI]]
                delta = [fx.Int32(x).bitcast(fx.Float32) for x in side[tr.NI:]]

                # ---- A(j): side(j+1), bias(j+1) and tile j+2 load.
                side_next = side_load(j + fx.Int32(1))
                bias_next = bias_load(j + fx.Int32(1))
                rocdl.sched_barrier(0)
                ni_t = Vec.make_type(tr.NI, fx.Float32)
                z_at = xs_at + fx.Index(z_sub) * XS_SUB + fx.Index(z_r0) * 4
                xs = [lds_io.load(z_at + w * (WAVE * 16), ni_t, 4 * tr.NI, 'xs') for w in range(2 * SHARDS)]

                def z_work():
                    parts = [[Vec(x)[i] for i in range(tr.NI)] for x in xs]
                    s_sum = [fx.Float32(parts[0][i]) + fx.Float32(parts[2][i]) for i in range(tr.NI)]
                    dp_sum = [fx.Float32(parts[1][i]) + fx.Float32(parts[3][i]) for i in range(tr.NI)]
                    if const_expr('nosm' in tr.EXP):
                        ps, dss = s_sum, dp_sum
                    else:
                        ps, dss = softmax(s_sum, dp_sum, lse, delta, bvals, q0, masked)
                    xo = xp_at + fx.Index(z_sub) * XP_SUB + fx.Index(z_r0) * 2
                    for k, vals in enumerate((ps, dss)):
                        lds_io.store(xo + k * (WAVE * 8), pack_items(vals), 2 * tr.NI, 'xp')

                own = [zero4] * tr.NSUB
                steps = x_steps(x_operands(), own)
                xop = []
                hooks = spread({}, len(steps), x_loads(j + fx.Int32(2), xop), tr.W_AT)
                hooks.setdefault(min(tr.Z_AT, len(steps) - 1), []).append(z_work)
                if const_expr(tr.TRACE):
                    trace('As')
                run(steps, hooks)
                phase_end('A' if tr.TRACE else None)

                # ---- B(j): partials of tile j+1 out; Y(j) from registers; derived reads of tile j+1;
                # then tile j+2 replaces it in the own area and X's operands of it come back.
                write_partials(own)
                pk_off = (fx.Index(1) - fx.Index(role)) * (WAVE * 8)
                pks = [lds_io.load(xp_at + (sub * XP_SUB) + pk_off, v4e(), 8, 'xp') for sub in range(tr.NSUB)]

                def y_use(b):
                    if const_expr('noy' in tr.EXP):
                        return
                    ops = derived_ops(dvd[b])
                    for e in range_constexpr(2):
                        for sub in range_constexpr(tr.NSUB):
                            k = (2 * b + e) * tr.NSUB + sub
                            accs[k] = mfma(ops[e], pks[sub], accs[k])

                steps = [lambda b=b: y_use(b) for b in range(tr.NB)]
                dvd_next = []
                hooks = spread({}, len(steps), derived_hooks(dvd_next))
                hooks.setdefault(len(steps) - 1, []).append(lambda: refill(xop))
                run(steps, hooks)
                phase_end('B' if tr.TRACE else None)
                return accs + [x for b in dvd_next for x in b] + side_next + bias_next

            n_acc = 2 * tr.NB * tr.NSUB

            @flyc.jit
            def tail(j, carried):
                out = list(carried)
                if j < fx.Index(n_tiles):
                    out = iteration(fx.Int32(j), carried, tr.CAUSAL)
                return out

            # Prologue: tile 0 to the own area: X(0), derived reads; partials of tile 0 out;
            # tile 1 to the own area.
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            xop0 = []
            for f in x_loads(fx.Int32(0), xop0):
                f()
            xop1 = []
            for f in x_loads(fx.Int32(1), xop1):
                f()
            side0 = side_load(fx.Int32(0))
            b0 = bias_load(fx.Int32(0))
            own0 = [zero4] * tr.NSUB
            refill(xop0)
            for f in x_steps(x_operands(), own0):
                f()
            dvd0 = []
            for f in derived_hooks(dvd0):
                f()
            refill(xop1)
            write_partials(own0)
            phase_end()

            init = list(accs) + [x for b in dvd0 for x in b] + side0 + b0
            res1 = init
            for jp, carried in range(0, fx.Index(n_pairs), 1, init=init):
                mid = iteration(fx.Int32(jp) * fx.Int32(2), carried, tr.CAUSAL)
                res1 = yield iteration(fx.Int32(jp) * fx.Int32(2) + fx.Int32(1), mid, tr.CAUSAL)
            out = tail(fx.Index(n_pairs) * 2, res1)
            return out[:n_acc]

        n_acc = 2 * tr.NB * tr.NSUB
        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range(n_acc)]
        results = acc0
        for g, carried in range(0, fx.Index(group), 1, init=acc0):
            results = yield head_pass(fx.Int32(g), carried)
        rocdl.s_waitcnt(vmcnt=0)
        store_out(results)

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx942_xxwide(
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
                                    value_attrs={'rocdl.waves_per_eu': waves_per_eu,
                                                 'rocdl.flat_work_group_size':
                                                 f'{tr.BLOCK_SIZE},{tr.BLOCK_SIZE}'}).launch(
            grid=(grid_x, fx.Index(num_head_k), grid_z) if tile_fastest else (fx.Index(num_head_k), grid_x, grid_z),
            block=(tr.BLOCK_SIZE, 1, 1),
            stream=stream)

    _launch = fmha.make_dkdv_launch(launch_fmha_bwd_dkdv_gfx942_xxwide,
                                    tr,
                                    causal_type,
                                    host_causal_type,
                                    bias,
                                    dropout)
    _launch.varlen_bits = fmha.varlen_bits
    _launch.varlen_compact = fmha.varlen_compact
    _launch.varlen_padded = fmha.varlen_padded
    _launch.varlen_strided = fmha.varlen_strided
    return _launch
