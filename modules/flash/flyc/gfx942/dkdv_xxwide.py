# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The xxwide dK/dV body (head dims 384, 512) on `flex_gemm`, `softmax` and `tensor_io`.

    S  = Q.K^T                  P = exp2(sm_scale*log2e*S + log2e*bias - lse2)
    dP = dO.V^T                 dS = P * (dP - Delta)
    dV^T += dO^T.P              dK^T += Q^T.dS;  dK = sm_scale * dK

One team of four waves owns 16 * kv_subs
keys; wave (shard c, role r) holds K (r 0) or V (r 1), and dK^T or dV^T, for the
d columns [c, c + 1) * D/2 only, so S and dP are sums of the two shards' partials,
reduced through LDS:
    K-wave c: S_c  = Q[:, c].K[:, c]^T   (`trait_ab`)   dK^T[c] += Q^T[c].dS   (`trait_tb`)
    V-wave c: dP_c = dO[:, c].V[:, c]^T                 dV^T[c] += dO^T[c].P
A wave loads its own tensor's half itself, whole rows a 16 B load, into an LDS
area no other wave reads (that private staging stays in this body); X reads it
row-major, Y derives its transpose (`derived_from`, the dword loads issued a
phase before their v_perm). Per 16-row q tile j, two barrier-separated phases:
    A(j): S, dP of tile j summed from the exchange; P(j), dS(j) of the wave's
          items to the exchange; partial S or dP of tile j+1; tile j+2 loads
    B(j): partials of j+1 to the exchange; dK^T or dV^T += tile j from the
          derived loads of B(j-1); tile j+1's derived loads; tile j+2 to the own area
The policy's band mask sets the score to -inf before exp2 (`mask_at` 'score':
+1.4% causal over 'prob', the same bits). Same kernel ABI and launcher
interface as the narrow body in `dkdv.py`.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops

import fmha_utils as fmha
from . import flex_gemm
from . import host
from . import softmax
from . import tensor_io
from . import utils
from . import tuning_dkdv as tuning
from .flex_gemm import (
    gemm_derived_load,
    gemm_derived_ops,
    gemm_mma_joint_at,
    gemm_read_at,
    gemm_read_plan,
)
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
BM = 16                 # q rows per tile: the N of the 16x16x16 MFMA
SHARDS = 2              # contraction shards: d halves
XS_SUB = 2 * SHARDS * WAVE * 16     # partial S and dP of one key column, 16 B per lane each
XP_SUB = 2 * WAVE * 8   # P and dS packs of one key column, 8 B per lane each


@dataclass(frozen=True)
class DkdvXxwideTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    kv_subs: int                # 16-key MFMA columns of the team
    waves_per_eu: int
    grid_axis_order: int
    bias: bool
    dropout: bool
    dropout_scale_at: str       # softmax.P_DROP_PER_ELEM or P_DROP_AT_STORE
    mask_at: str                # softmax.MASK_PROB or MASK_SCORE
    z_at: int = 99              # the X step of A(j) that Z(j) follows (99: last)
    w_at: int = 0               # the X step of A(j) the tile loads start after

    # make_dkdv_launch's names
    @property
    def D(self):  # noqa: N802
        return self.head_dim

    @property
    def PADDED(self):  # noqa: N802
        return self.padded

    @property
    def block_size(self):
        return NUM_WAVES * WAVE

    @property
    def block_kv(self):
        return 16 * self.kv_subs

    @property
    def pad_floor(self):
        return 0 if self.padded else self.head_dim

    @property
    def items(self):
        """Softmax items (column, row r) per wave and tile: kv_subs of them, or at 3 a
        column each, the fourth wave on a spare column. Also the exchange's columns."""
        return 4 if self.kv_subs == 3 else self.kv_subs

    @property
    def q_down(self):
        """Walk q tiles last to first: a causal head's blocks then all start on its last
        tiles and share them in L2, instead of each starting at its own diagonal."""
        return self.causal

    def gemms(self):
        """(X, Y): X = partial S or dP on the wave's own area (its tensor's 16 rows,
        its half of the columns; area `wave`), Y = dK^T or dV^T derived from it."""
        xg = flex_gemm.trait_ab(self.dtype,
                                BM,
                                self.head_dim // SHARDS,
                                self.pad_floor,
                                self.block_size,
                                0,
                                'own',
                                dma='rows',
                                mfma='16x16x16',
                                buffers=1,
                                b_tiles=self.kv_subs)
        yg = flex_gemm.trait_tb(self.dtype,
                                BM,
                                self.head_dim // SHARDS,
                                self.block_size,
                                0,
                                'own',
                                mfma='16x16x16',
                                derived_from=xg.a,
                                b_tiles=self.kv_subs)
        return xg, yg


def build_fmha_bwd_dkdv_gfx942_xxwide(head_dim,
                                           causal=False,
                                           causal_type=None,
                                           dtype_str='bf16',
                                           bias=False,
                                           dropout=False,
                                           kv_subs=None,
                                           waves_per_eu=1,
                                           grid_axis_order=None,
                                           padded_head=False,
                                           dropout_scale_at=None,
                                           mask_at=None):
    """build_fmha_bwd_dkdv_gfx942's interface and kernel ABI, for head dims 384 and 512."""
    assert head_dim % 64 == 0 and 128 <= head_dim <= 512, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if kv_subs is None:
        kv_subs = tuning.body_kv_subs('xxwide', head_dim)
    assert kv_subs in (2, 3, 4), kv_subs
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    tr = DkdvXxwideTraits(head_dim=head_dim,
                          padded=padded_head,
                          causal=causal_type != 0,
                          dtype=dtype_str,
                          kv_subs=kv_subs,
                          waves_per_eu=waves_per_eu,
                          grid_axis_order=grid_axis_order,
                          bias=bias,
                          dropout=dropout,
                          dropout_scale_at=dropout_scale_at or tuning.default_dropout_scale_at('xxwide'),
                          mask_at=mask_at or tuning.default_mask_at('xxwide'))
    xg, yg = tr.gemms()
    NSUB = tr.kv_subs
    NI = tr.items
    NB = xg.reads                   # own 32-column blocks: one 16 B read, two k-steps
    n_acc = yg.n_acc
    own_bytes = xg.a.tile_bytes
    own_stride = xg.a.stride
    cpr = head_dim // 16            # 16 B chunks of a row's own half
    n_ld = BM * cpr // WAVE         # 16 B loads a lane per tile
    lds_xs = NUM_WAVES * own_bytes
    lds_xp = lds_xs + NI * XS_SUB
    lds_bytes = lds_xp + NI * XP_SUB
    assert lds_bytes <= 65536, lds_bytes
    tag = 'dkdv_xxwide_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)

    def pack_items(vals):
        """NI f32 -> the bits of NI elements: i32 or <2 x i32>; bf16 rounds as pack_operand."""
        if const_expr(len(vals) == 4):
            return Vec(pack_operand(vals, elem)).bitcast(fx.Int32)
        if const_expr(dtype_str == 'bf16'):
            rnd = [fx.Float32(x).bitcast(fx.Int32) + fx.Int32(0x8000) for x in vals]
            return fx.Int32(rocdl.perm_b32(rnd[1], rnd[0], fx.Int32(0x07060302)))
        return Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem).bitcast(fx.Int32)[0]

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
        z_sub = (wave_s * fx.Int32(NI)) >> fx.Int32(2)
        z_r0 = (wave_s * fx.Int32(NI)) & fx.Int32(3)

        lds = fx.SharedAllocator().allocate(Smem).peek()
        lds_p = lds.buf.ptr
        lds_io = Lds(lds_p, elem, _LDS_DOMAIN, LDS_SCOPES)

        # The 8xD contract: a row owns ceil8(hdim) columns (numerical lessons, #11).
        row_qk = (fx.Int32(hdim_qk) + fx.Int32(7)) & fx.Int32(-8)
        row_vo = (fx.Int32(hdim_vo) + fx.Int32(7)) & fx.Int32(-8)
        hdim_own = is_v.select(fx.Int32(hdim_vo), fx.Int32(hdim_qk))

        def masked8(v8, c0, hdim):
            return apply_mask(v8, col_mask(c0, hdim), elem)

        # ---- the resident B operand: K (K-wave) or V (V-wave) of the own half, key
        # kvt + 16 sub + lc; as B packs [block][read]
        k0 = fx.Int32(kv_tile) * fx.Int32(tr.block_kv)
        kvt = k0
        k_base = base_addr(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq)
        v_base = base_addr(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq)
        r_ss = is_v.select(fx.Int64(stride_v_seq), fx.Int64(stride_k_seq))
        r_rsrc = buffer_ops.create_buffer_resource_from_addr(is_v.select(v_base, k_base),
                                                             num_records_bytes=span_bytes(sk_i32,
                                                                                          r_ss,
                                                                                          is_v.select(row_vo, row_qk)))
        res = []
        for t in range_constexpr(NB):
            row = []
            for sub in range_constexpr(NSUB):
                r = fx.Index(kvt + fx.Int32(16 * sub)) + lc
                c0 = fx.Index(32 * t) + fx.Index(shard) * (head_dim // 2) + gq * 8
                v8 = buffer_ops.buffer_load(r_rsrc, r * fx.Index(fx.Int32(r_ss)) + c0, vec_width=8, dtype=elem)
                if const_expr(tr.padded):
                    # Slack columns may hold NaN, and 0 * NaN is NaN.
                    v8 = masked8(v8, fx.Int32(c0), hdim_own)
                row.append(v8)
            res.append(row)
        res = [[Vec(res[t][sub]) for t in range(NB)] for sub in range(NSUB)]

        # ---- the visited q range of this key block
        k_hi = smin(k0 + fx.Int32(tr.block_kv), sk_i32) - fx.Int32(1)
        if const_expr(tr.causal):
            win_l, win_r = fmha.resolve_window(window_left, window_right, sq_i32, sk_i32, varlen_bits)
            q_lo = smax(k0 - win_r, fx.Int32(0))
            q_hi = smin(k_hi + win_l, sq_i32 - fx.Int32(1))
        else:
            q_lo = fx.Int32(0)
            q_hi = sq_i32 - fx.Int32(1)
        alive = (k0 < sk_i32) & (q_lo <= q_hi)
        t0 = alive.select(q_lo, fx.Int32(0)) // fx.Int32(BM)
        n_tiles = alive.select(q_hi // fx.Int32(BM) - t0 + fx.Int32(1), fx.Int32(0))
        t_last = t0 + n_tiles - fx.Int32(1)
        n_pairs = n_tiles // fx.Int32(2)

        # The key this lane turns into P and dS.
        soft_kv = kvt + fx.Int32(16) * z_sub + fx.Int32(lc)
        # Key kv sees q rows [kv - win_r, kv - win_r + band_span]: one unsigned compare per element
        # of rel = q - kv + win_r = q0 + band_s + (4gq - lc + r), band_s wave-uniform.
        if const_expr(tr.causal):
            empty = win_l + win_r < fx.Int32(0)
            band_span = empty.select(fx.Int32(0), win_l + win_r)
            band_s = empty.select(fx.Int32(1 << 30), win_r - kvt - fx.Int32(16) * z_sub + z_r0)

        # ---- LDS plans: X reads the wave's own area row-major, Y derives its transpose;
        # the exchange: partial S / dP at [sub][2 shard + role][lane]; P, dS at [sub][P, dS][lane].
        own_base = fx.Index(xg.a.lds_base) + fx.Index(wave) * own_bytes
        x_plan = gemm_read_plan(xg, lc, gq, tensor=wave)
        y_plan = gemm_read_plan(yg, lc, gq, tensor=wave)
        xs_at = fx.Index(lds_xs) + lane * 16
        xp_at = fx.Index(lds_xp) + lane * 8

        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        zero4 = fx.full(4, 0.0, fx.Float32)
        if const_expr(dropout):
            rng, ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            assert rng.randoms_per_offset == 4
            # Lane (lc, gq) draws q row 4gq + lc % 4 for keys 4 (lc // 4) + [0, 4); its quad
            # holds a 4x4 block, and a DPP broadcast deals each lane its key's bits.
            quad_shift = fx.Int32(lc % 4) + z_r0 * fx.Int32(4)

        ctx = flex_gemm.Ctx(tr, carried=('acc', 'dvd', 'side', 'bias'))
        ctx.dropout_scale = dropout_scale
        # Q rows, for softmax.lse_rows.
        ctx.set_q_rows(varlen_bits=varlen_bits,
                       q_batch=q_batch,
                       q_row_off=q_row_off,
                       sq=sq_i32,
                       max_seqlen_q=max_seqlen_q,
                       seqinfo_q0=seqinfo_q0,
                       seqinfo_q1=seqinfo_q1,
                       num_seqlens=num_seqlens,
                       num_head_q=num_head_q)
        lse_tokens = softmax.lse_tokens(ctx)

        # ---- the schedule's driver: steps in order, other work after given steps
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

        def phase_end():
            """LDS writes done, then the barrier; the tile loads stay in flight."""
            rocdl.s_waitcnt(lgkmcnt=0)
            cluster_barrier()

        # ---- per q head of this kv head's group: dK/dV accumulate over all of them
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
            lse_row0, lse_pitch = softmax.lse_rows(ctx, head=head_q, tokens=lse_tokens)
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
                if const_expr(tr.q_down):
                    return (t_last - j) * fx.Int32(BM)
                return (t0 + j) * fx.Int32(BM)

            # ---- private staging (this body's own): load i of a lane is 16 B chunk 64i + lane
            # of the wave's half-rows, whole rows a load; written to the own area as loaded.
            r_row = [fx.Int32((fx.Index(WAVE * i) + lane) // cpr) for i in range(n_ld)]
            r_col = [fx.Int32((fx.Index(WAVE * i) + lane) % cpr) for i in range(n_ld)]
            r_voff = [r_row[i] * o_rs_b + r_col[i] * fx.Int32(16) + shard * fx.Int32(head_dim) for i in range(n_ld)]

            def x_loads(j, out):
                """Own tile j into out (n_ld v8), one callable a load."""
                def ld(i):
                    v = rocdl.raw_ptr_buffer_load(Vec.make_type(8, elem), o_rsrc, r_voff[i], q0_of(j) * o_rs_b, 0)
                    if const_expr(tr.padded):
                        # Slack columns may hold NaN, and 0 * NaN is NaN.
                        v = masked8(v, r_col[i] * fx.Int32(8) + shard * fx.Int32(head_dim // 2), hdim_own)
                    out[i] = v
                out.extend([None] * n_ld)
                return [lambda i=i: ld(i) for i in range(n_ld)]

            def refill(xop):
                """Tile xop to the own area, after the derived loads of the tile before (the
                same wave's LDS ops run in order)."""
                for i in range_constexpr(n_ld):
                    at = own_base + fx.Index(r_row[i]) * own_stride + fx.Index(r_col[i]) * 16
                    lds_io.store(at, Vec(xop[i]).bitcast(fx.Int32), 16, 'own')

            # ---- memory: LSE/Delta and bias of the wave's softmax items
            def side_load(j):
                """LSE and Delta of q = q0 + 4gq + z_r0 + i, i < NI, as i32."""
                soff = q0_of(j) * pitch_b
                out = []
                for rsrc in (lse_rsrc, delta_rsrc):
                    for i in range_constexpr(NI):
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
                for i in range_constexpr(NI):
                    h = rocdl.raw_ptr_buffer_load(T.i16, b_rsrc, b_voff + fx.Int32(i) * b_rs_b, bsoff, 0)
                    out.append(fx.Int32(fx.Int16(h)))
                return out

            # ---- score hooks: the band mask, bias, dropout keep bits
            def keep_bits(q0):
                """Dropout keep of (q0 + 4gq + z_r0 + i, soft_kv) for i < NI, as i1."""
                kv4 = soft_kv - fx.Int32(lc % 4)
                row = q0 + fx.Int32(gq * 4 + lc % 4)
                return deal_keep_bits(rng, ph_seed, ph_base, ph_stride, row, kv4, idropout_p, quad_shift, NI)

            def p_ds(s_raw, dp_raw, lse, delta, bvals, q0, masked):
                """P (for dV) and dS of the softmax items, f32."""
                mask_of = None
                if const_expr(masked):
                    # The lane term afresh per tile: hoisted, it holds a VGPR through the loop.
                    lt = fx.Int32(lane // 16) * fx.Int32(4) - fx.Int32(lane % 16)
                    lt = fx.Int32(pin_values([lt])[0])
                    rel0 = q0 + band_s + lt

                    def mask_of(v, r, fill):
                        live = fx.Uint32(rel0 + fx.Int32(r)) <= fx.Uint32(band_span)
                        return live.select(v, fill)
                keep = None
                if const_expr(dropout):
                    keep = keep_bits(q0)

                def bias_of(r):
                    bv = Vec.from_elements([fx.Int16(bvals[r])], fx.Int16).bitcast(elem)
                    return fx.Float32(Vec(bv)[0].to(fx.Float32))
                return softmax.p_ds_elems(ctx,
                                          s_raw,
                                          dp_raw,
                                          lse_l2=lambda r: fx.Float32(lse[r]) * fx.Float32(LOG2E),
                                          delta=delta,
                                          bias_of=bias_of,
                                          mask_of=mask_of,
                                          keep=keep,
                                          c_scale=c_scale)

            # ---- the exchange: S/dP partials in, sums and P/dS out. Its layout follows the
            # shards' and roles' lane split, so it stays in this body.
            def z_sums():
                """Tile j's four partials of the wave's items, as loads (issued early)."""
                z_at = xs_at + fx.Index(z_sub) * XS_SUB + fx.Index(z_r0) * 4
                ni_t = Vec.make_type(NI, fx.Float32)
                return [lds_io.load(z_at + w * (WAVE * 16), ni_t, 4 * NI, 'xs') for w in range(2 * SHARDS)]

            def xp_put(ps, dss):
                xo = xp_at + fx.Index(z_sub) * XP_SUB + fx.Index(z_r0) * 2
                for k, vals in enumerate((ps, dss)):
                    lds_io.store(xo + k * (WAVE * 8), pack_items(vals), 2 * NI, 'xp')

            def write_partials(own):
                for sub in range_constexpr(NSUB):
                    lds_io.store(xs_at + fx.Index(wave_s) * (WAVE * 16) + sub * XS_SUB, own[sub], 16, 'xs')

            def xp_get():
                """The B operand of Y: dS (K-wave) or P (V-wave) of every column."""
                pk_off = (fx.Index(1) - fx.Index(role)) * (WAVE * 8)
                return [lds_io.load(xp_at + (sub * XP_SUB) + pk_off, Vec.make_type(4, elem), 8, 'xp')
                        for sub in range(NSUB)]

            # ---- GEMM steps: X = partial S or dP, Y = dK^T or dV^T, one step a block
            def x_steps(own):
                """X's A of every own block read now, then one MFMA step a block into own."""
                xop = [gemm_read_at(xg, x_plan, lds_io, 0, b) for b in range(NB)]
                return [lambda b=b: gemm_mma_joint_at((xg,), b, [xop[b]], (res,), [own]) for b in range(NB)]

            def derived_loads(out):
                """Y's derived dwords of every own block into out, one callable a block."""
                def rd(b):
                    out[b] = gemm_derived_load(yg, y_plan, lds_io, 0, b)
                out.extend([None] * NB)
                return [lambda b=b: rd(b) for b in range(NB)]

            # ---- the tile loop. Carried: the accumulators, tile j's derived dwords, LSE/Delta, bias.
            def iteration(j, masked):
                q0 = q0_of(j)
                lse = [fx.Int32(x).bitcast(fx.Float32) for x in ctx.side[:NI]]
                delta = [fx.Int32(x).bitcast(fx.Float32) for x in ctx.side[NI:]]
                # A(j): partial S or dP of tile j+1 on the MFMAs, the tile j+2 loads spread over
                # them; after them Z(j): S, dP summed from the exchange, P and dS to it.
                side_next = side_load(j + fx.Int32(1))
                bias_next = bias_load(j + fx.Int32(1))
                rocdl.sched_barrier(0)
                xs = z_sums()

                def z_work():
                    parts = [[Vec(x)[i] for i in range(NI)] for x in xs]
                    s_sum = [fx.Float32(parts[0][i]) + fx.Float32(parts[2][i]) for i in range(NI)]
                    dp_sum = [fx.Float32(parts[1][i]) + fx.Float32(parts[3][i]) for i in range(NI)]
                    xp_put(*p_ds(s_sum, dp_sum, lse, delta, ctx.bias, q0, masked))

                own = [zero4] * NSUB
                steps = x_steps(own)
                xop = []
                hooks = spread({}, len(steps), x_loads(j + fx.Int32(2), xop), tr.w_at)
                hooks.setdefault(min(tr.z_at, len(steps) - 1), []).append(z_work)
                run(steps, hooks)
                phase_end()
                # B(j): partials of tile j+1 out; dK^T += Q^T.dS or dV^T += dO^T.P of tile j
                # from last phase's derived dwords; tile j+1's derived loads, then tile j+2
                # replaces it in the own area.
                write_partials(own)
                pks = xp_get()
                acc = list(ctx.acc)
                steps = [lambda b=b: gemm_mma_joint_at((yg,), b, [gemm_derived_ops(yg, ctx.dvd[b])], (pks,), [acc])
                         for b in range(NB)]
                dvd_next = []
                hooks = spread({}, len(steps), derived_loads(dvd_next))
                hooks.setdefault(len(steps) - 1, []).append(lambda: refill(xop))
                run(steps, hooks)
                phase_end()
                ctx.acc, ctx.dvd, ctx.side, ctx.bias = acc, dvd_next, side_next, bias_next

            def step(j, carried):
                ctx.restore(carried)
                iteration(j, tr.causal)
                return ctx.carry()

            @flyc.jit
            def tail(j, carried):
                out = list(carried)
                if j < fx.Index(n_tiles):
                    out = step(fx.Int32(j), carried)
                return out

            # Prologue: tile 0 to the own area: X(0), derived loads; partials of tile 0 out;
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
            own0 = [zero4] * NSUB
            refill(xop0)
            for f in x_steps(own0):
                f()
            dvd0 = []
            for f in derived_loads(dvd0):
                f()
            refill(xop1)
            write_partials(own0)
            phase_end()

            # Pairs of tiles, then at most one more.
            ctx.acc, ctx.dvd, ctx.side, ctx.bias = list(accs), dvd0, side0, b0
            init = ctx.carry()
            res1 = init
            for jp, carried in range(0, fx.Index(n_pairs), 1, init=init):
                mid = step(fx.Int32(jp) * fx.Int32(2), carried)
                res1 = yield step(fx.Int32(jp) * fx.Int32(2) + fx.Int32(1), mid)
            out = tail(fx.Index(n_pairs) * 2, res1)
            return out[:n_acc]

        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range(n_acc)]
        results = acc0
        for g, carried in range(0, fx.Index(group), 1, init=acc0):
            results = yield head_pass(fx.Int32(g), carried)
        rocdl.s_waitcnt(vmcnt=0)

        # ---- epilogue: dK^T (K-wave, times sm_scale) or dV^T (V-wave; times 1/(1-p) if
        # dropout scales it here) of the wave's shard, transposed back by the store.
        dk_base = base_addr(DK, k_batch, head_k, k_row_off, stride_dk_batch, stride_dk_head, stride_dk_seq)
        dv_base = base_addr(DV, k_batch, head_k, k_row_off, stride_dv_batch, stride_dv_head, stride_dv_seq)
        o_ss = is_v.select(fx.Int64(stride_dv_seq), fx.Int64(stride_dk_seq))
        row_lim = is_v.select(row_vo, row_qk)
        rsrc = buffer_ops.create_buffer_resource_from_addr(is_v.select(dv_base, dk_base),
                                                           num_records_bytes=span_bytes(sk_i32, o_ss, row_lim))
        ss = fx.Index(fx.Int32(o_ss))
        # dS is against the scaled scores: sm_scale lands once, here.
        dv_scale = fx.Float32(1.0)
        if const_expr(dropout and tr.dropout_scale_at == softmax.P_DROP_AT_STORE):
            dv_scale = fx.Float32(dropout_scale)
        scale = is_v.select(dv_scale, fx.Float32(sm_scale))
        tensor_io.store_acc_derived16(yg,
                                      list(results),
                                      1,
                                      rsrc,
                                      ss,
                                      kvt,
                                      lc,
                                      gq,
                                      sk_i32,
                                      row_lim,
                                      fx.full(4, scale, fx.Float32),
                                      elem,
                                      col0=lambda: shard * fx.Int32(head_dim // 2))

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
        grid_x = (fx.Index(max_seqlen_k) + tr.block_kv - 1) // tr.block_kv
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
                                    value_attrs={'rocdl.waves_per_eu': tr.waves_per_eu,
                                                 'rocdl.flat_work_group_size':
                                                 f'{tr.block_size},{tr.block_size}'}).launch(
            grid=(grid_x, fx.Index(num_head_k), grid_z) if tile_fastest else (fx.Index(num_head_k), grid_x, grid_z),
            block=(tr.block_size, 1, 1),
            stream=stream)

    _launch = host.make_dkdv_launch(launch_fmha_bwd_dkdv_gfx942_xxwide,
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
