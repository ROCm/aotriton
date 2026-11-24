# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The xwide dK/dV body (head dims 224, 256) on `flex_gemm`, `softmax` and `tensor_io`.

    S  = Q.K^T                  P = exp2(sm_scale*log2e*S + log2e*bias - lse2)
    dP = dO.V^T                 dS = P * (dP - Delta)
    dV^T += dO^T.P              dK^T += Q^T.dS;  dK = sm_scale * dK

Two waves form a team owning 16 * kv_subs
keys. The K-wave holds K and dK^T, the V-wave V and dV^T, so each wave runs one
GEMM of each pair on its own tensor:
    K-wave: S = Q.K^T (`trait_ab`, Q row-major)    dK^T += Q^T.dS (`trait_tb`)
    V-wave: dP = dO.V^T                            dV^T += dO^T.P
Q and dO live in LDS row-major only (DMA'd by rows, `dma='rows'`); the transposed
A is derived on read (`derived_from`: four dword reads of one column pair, four
v_perm). Each 16-row q tile is three barrier-separated phases:
    X: S or dP from RM, streamed; both raw tiles to the team exchange; DMA RM(j+1)
    Z: P and dS of half the team's key columns (sub % 2 == role) to the exchange
    Y: dK^T or dV^T += tile j, streamed from the derived reads of RM(j)
At 512 (outside the policy) a tile's D is two column parts in a ring of three
slots. The policy's band mask zeroes P after exp2 (`mask_at` 'prob'; 'score'
costs ~3% causal and moves f16 roundings). Same kernel ABI and launcher
interface as the narrow body in `dkdv.py`.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
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
    gemm_dma_plan,
    gemm_dma_rows,
    gemm_read_plan,
    gemm_stream_joint,
)
from .utils import (
    LOG2E,
    OOB_VOFF,
    WAVE,
    Lds,
    alias_scopes,
    apply_mask,
    base_addr,
    cluster_barrier,
    col_mask,
    deal_keep_bits,
    lds_ptr_at,
    m0_at,
    pack_operand,
    philox_plane,
    philox_stream,
    pin_values,
    smax,
    smin,
    span_bytes,
    v4f,
)

_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "fmha_bwd_dkdv_gfx942_xwide.lds">'
LDS_SCOPES = ('rm0', 'rm1', 'rm2', 'xs', 'xp', 's0', 's1', 's2')
NUM_WAVES = 4
BM = 16                 # q rows per tile: the N of the 16x16x16 MFMA
XS_SUB = 2 * WAVE * 16  # exchange of one key column: raw S and dP, 16 B per lane each
XP_SUB = 2 * WAVE * 8   # P and dS packs of one key column, 8 B per lane each
SIDE_SLOT = 512         # LSE then Delta of one tile, 256 B per DMA issue


@dataclass(frozen=True)
class DkdvXwideTraits:
    """Everything one build fixes. Frozen Python values; the GEMMs derive from it."""
    head_dim: int
    padded: bool
    causal: bool
    dtype: str
    kv_subs: int                # 16-key MFMA columns per team
    waves_per_eu: int
    grid_axis_order: int
    bias: bool
    dropout: bool
    dropout_scale_at: str       # softmax.P_DROP_PER_ELEM or P_DROP_AT_STORE
    mask_at: str                # softmax.MASK_PROB or MASK_SCORE
    prefetch: int = 2           # RM reads in flight ahead of their MFMAs
    y_ahead: int = 2            # derived reads in flight ahead of their MFMAs

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
    def team_keys(self):
        return 16 * self.kv_subs

    @property
    def block_kv(self):
        return (NUM_WAVES // 2) * self.team_keys

    @property
    def pad_floor(self):
        return 0 if self.padded else self.head_dim

    @property
    def x_bytes(self):
        """The team exchange and LSE/Delta slots."""
        return (NUM_WAVES // 2) * self.kv_subs * (XS_SUB + XP_SUB) + 3 * SIDE_SLOT

    @property
    def parts(self):
        """A tile's D in column parts, each its own RM slot: two whole-tile slots, or (at
        512, where those and the exchange outgrow LDS) a ring of three half-tiles."""
        whole = flex_gemm.RowA(BM, self.head_dim, 0, self.block_size, 0, 'rm', dma='rows').tile_bytes
        return 1 if 4 * whole + self.x_bytes <= 65536 else 2

    @property
    def rm_slots(self):
        return 2 if self.parts == 1 else 3

    @property
    def period(self):
        """Tiles until the RM slot pattern repeats."""
        return 2 if self.parts == 1 else 3

    @property
    def kv_pin(self):
        """The resident K or V pinned to VGPRs ('v') or AGPRs ('a', MFMA reads B from
        either); padded builds leave LLVM free (the wide body's regalloc aborts). 512
        dense: 'a' 344 TFLOPS (36 spills), 'v' 323 (138); 384 ties."""
        return '' if self.padded else ('a' if self.head_dim > 384 else 'v')

    @property
    def ops_vgpr(self):
        """Keep RM operands out of AGPRs: hd512 sits at a register-allocation cliff, and
        the causal mask tips it (B2 H32 S8k: 222 with this, 135 without)."""
        return self.causal and self.head_dim > 384 and not self.padded

    @property
    def q_down(self):
        """Walk q tiles last to first: a causal head's blocks then all start on its last
        tiles and share them in L2, instead of each starting at its own diagonal."""
        return self.causal

    def gemms(self):
        """(X, Y) of one column part and the dO tile's DMA view: X = S or dP on the
        wave's own RM tile (Q at 0, dO a tile later; the role picks at run time),
        Y = dK^T or dV^T derived from it. RM slots [Q | dO] from LDS 0."""
        dp = self.head_dim // self.parts

        def rows(lds_base, buf_stride):
            return flex_gemm.trait_ab(self.dtype,
                                      BM,
                                      dp,
                                      self.pad_floor,
                                      self.block_size,
                                      lds_base,
                                      'rm',
                                      dma='rows',
                                      mfma='16x16x16',
                                      buffers=self.rm_slots,
                                      b_tiles=self.kv_subs,
                                      buf_stride=buf_stride)

        rm_tile = rows(0, 0).a.tile_bytes
        xg = rows(0, 2 * rm_tile)
        y = flex_gemm.trait_tb(self.dtype,
                               BM,
                               dp,
                               self.block_size,
                               0,
                               'rm',
                               mfma='16x16x16',
                               derived_from=xg.a,
                               b_tiles=self.kv_subs)
        return xg, rows(rm_tile, 2 * rm_tile), y


def build_fmha_bwd_dkdv_gfx942_xwide(head_dim,
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
    """build_fmha_bwd_dkdv_gfx942's interface and kernel ABI, for head dims 224 and 256."""
    assert head_dim % 32 == 0 and 64 <= head_dim <= 512, head_dim
    host_causal_type, causal_type = host.builder_causal_types(causal, causal_type, bias)
    if kv_subs is None:
        kv_subs = tuning.body_kv_subs('xwide', head_dim)
    assert kv_subs in (2, 4), kv_subs
    if grid_axis_order is None:
        grid_axis_order = tuning.default_grid_axis_order(causal_type != 0)
    tr = DkdvXwideTraits(head_dim=head_dim,
                         padded=padded_head,
                         causal=causal_type != 0,
                         dtype=dtype_str,
                         kv_subs=kv_subs,
                         waves_per_eu=waves_per_eu,
                         grid_axis_order=grid_axis_order,
                         bias=bias,
                         dropout=dropout,
                         dropout_scale_at=dropout_scale_at or tuning.default_dropout_scale_at('xwide'),
                         mask_at=mask_at or tuning.default_mask_at('xwide'))
    xg, dog, yg = tr.gemms()
    PARTS = tr.parts
    NSUB = tr.kv_subs
    NT_P = xg.reads                 # 16 B RM reads per part
    NDT_P = yg.out_tiles            # 16-row d tiles of a part
    n_acc = PARTS * yg.n_acc
    soft_subs = NSUB // 2           # key columns each role turns into P and dS
    rm_tensor = xg.a.tile_bytes
    rm_buf = xg.a.buf_bytes
    part_bytes = xg.a.row_bytes
    n_dma = xg.a.n_dma + dog.a.n_dma        # per wave and part
    # LDS: RM slots, then the exchange of raw S/dP (xs) and of P/dS packs (xp), then LSE/Delta.
    lds_xs = tr.rm_slots * rm_buf
    xs_team = NSUB * XS_SUB
    lds_xp = lds_xs + (NUM_WAVES // 2) * xs_team
    xp_team = NSUB * XP_SUB
    lds_side = lds_xp + (NUM_WAVES // 2) * xp_team
    lds_bytes = lds_side + tr.period * SIDE_SLOT
    assert lds_bytes <= 65536, lds_bytes
    tag = 'dkdv_xwide_' + flex_gemm.cache_tag(tr, flex_gemm, softmax, tensor_io, utils)
    tile_fastest = grid_axis_order == tuning.GRID_AXIS_TILE_FASTEST
    elem = flex_gemm.elem_type(dtype_str)

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
        team = wave_s >> fx.Int32(1)
        is_v = role == fx.Int32(1)

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

        # ---- the resident B operand: K (K-wave) or V (V-wave), key kvt + 16 sub + lc,
        # d = 32t + 8gq + [0, 8); as B packs [block][read]
        k0 = fx.Int32(kv_tile) * fx.Int32(tr.block_kv)
        kvt = k0 + team * fx.Int32(tr.team_keys)
        k_base = base_addr(K, k_batch, head_k, k_row_off, stride_k_batch, stride_k_head, stride_k_seq)
        v_base = base_addr(V, k_batch, head_k, k_row_off, stride_v_batch, stride_v_head, stride_v_seq)
        r_ss = is_v.select(fx.Int64(stride_v_seq), fx.Int64(stride_k_seq))
        r_rsrc = buffer_ops.create_buffer_resource_from_addr(is_v.select(v_base, k_base),
                                                             num_records_bytes=span_bytes(sk_i32,
                                                                                          r_ss,
                                                                                          is_v.select(row_vo, row_qk)))
        nt = PARTS * NT_P
        res = []
        for t in range_constexpr(nt):
            row = []
            for sub in range_constexpr(NSUB):
                r = fx.Index(kvt + fx.Int32(16 * sub)) + lc
                c0 = fx.Index(32 * t) + gq * 8
                v8 = buffer_ops.buffer_load(r_rsrc, r * fx.Index(fx.Int32(r_ss)) + c0, vec_width=8, dtype=elem)
                if const_expr(tr.padded):
                    # Slack columns may hold NaN, and 0 * NaN is NaN.
                    v8 = masked8(v8, fx.Int32(c0), hdim_own)
                row.append(v8)
            res.append(row)
        if const_expr(tr.kv_pin):
            flat = pin_values([x for row in res for x in row], tr.kv_pin)
            res = [[flat[t * NSUB + sub] for sub in range(NSUB)] for t in range(nt)]
        res = [[Vec(res[t][sub]) for t in range(nt)] for sub in range(NSUB)]

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
        n_groups = n_tiles // fx.Int32(tr.period)

        # The key columns this wave turns into P and dS: sub = 2i + role.
        soft_kv = [kvt + fx.Int32(16) * (fx.Int32(2 * i) + role) + fx.Int32(lc) for i in range(soft_subs)]
        # Key kv sees q rows [kv - win_r, kv - win_r + band_span]: one unsigned compare per element
        # of rel = q - kv + win_r = q0 + band_s[i] + (4gq - lc + r), band_s wave-uniform.
        if const_expr(tr.causal):
            empty = win_l + win_r < fx.Int32(0)
            band_span = empty.select(fx.Int32(0), win_l + win_r)
            band_s = [empty.select(fx.Int32(1 << 30), win_r - kvt - fx.Int32(16) * (fx.Int32(2 * i) + role))
                      for i in range(soft_subs)]

        # ---- LDS plans: X reads the wave's own RM tile, Y derives its transpose; the
        # team exchange (xs, xp) and LSE/Delta are this body's own areas.
        rm_plan = gemm_read_plan(xg, lc, gq, tensor=role)
        y_plan = gemm_read_plan(yg, lc, gq, tensor=role)
        xs_at = fx.Index(lds_xs) + fx.Index(team) * xs_team + lane * 16
        xp_at = fx.Index(lds_xp) + fx.Index(team) * xp_team + lane * 8
        side_rd = fx.Index(lds_side) + gq * 16
        c_scale = fx.Float32(sm_scale) * fx.Float32(LOG2E)
        m0_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_base + fx.Index(xg.a.lds_base))))
        dma_plan = gemm_dma_plan(xg, wave_s, lane, None, m0_base)

        zero4 = fx.full(4, 0.0, fx.Float32)
        if const_expr(dropout):
            rng, ph_seed, ph_off = philox_stream(philox_seed_ptr, philox_offset1, philox_offset2)
            assert rng.randoms_per_offset == 4
            # Lane (lc, gq) draws q row 4gq + lc % 4 for keys 4 (lc // 4) + [0, 4); its quad
            # holds a 4x4 block, and a DPP broadcast deals each lane its key's bits.
            quad_shift = fx.Int32(lc % 4)

        ctx = flex_gemm.Ctx(tr, carried=('acc', 'bias'))
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

        def cluster_end():
            """LDS reads land before the next phase overwrites them (s_barrier does not wait)."""
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
            q_rs_b = fx.Int32(stride_q_seq) * fx.Int32(2)
            do_rs_b = fx.Int32(stride_do_seq) * fx.Int32(2)
            q_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(q_base, num_records_bytes=q_span))
            do_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(do_base,
                                                                                        num_records_bytes=do_span))
            lse_row0, lse_pitch = softmax.lse_rows(ctx, head=head_q, tokens=lse_tokens)
            pitch_b = fx.Int32(lse_pitch) * fx.Int32(4)
            side_bytes = fx.Int64(fx.Index(sq_i32) * lse_pitch) * fx.Int64(4)
            lse_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(LSE)) + fx.Int64(lse_row0) * fx.Int64(4), num_records_bytes=side_bytes))
            delta_rsrc = rocdl.get_buffer_rsrc(buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(Delta)) + fx.Int64(lse_row0) * fx.Int64(4), num_records_bytes=side_bytes))
            side_dma_voff = (lane < fx.Index(BM)).select(fx.Int32(lane) * pitch_b, fx.Int32(OOB_VOFF))

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
                ph_base, ph_stride = philox_plane(rng, ph_off, z_i32, num_head_q, head_q, max_seqlen_q, max_seqlen_k)

            def q0_of(j):
                if const_expr(tr.q_down):
                    return (t_last - j) * fx.Int32(BM)
                return (t0 + j) * fx.Int32(BM)

            # ---- memory: RM DMA by rows (part `part` of the columns), LSE/Delta, bias
            def rows_dma(j, part, slot):
                """Part `part` of RM(j) -> slot, one callable per row: wave w copies rows
                4w..4w+3 of Q, then of dO; the part's column rides in soffset."""
                q0 = q0_of(j)
                return (gemm_dma_rows(xg, dma_plan, q_rsrc, scopes, q0, q_rs_b, slot, col_byte=part * part_bytes)
                        + gemm_dma_rows(dog, dma_plan, do_rsrc, scopes, q0, do_rs_b, slot, col_byte=part * part_bytes))

            def side_dma(j, sslot):
                """LSE(j) and Delta(j) -> side slot, 256 B each; lane L < 16 copies row q0 + L."""
                soff = q0_of(j) * pitch_b
                for i, rsrc in enumerate((lse_rsrc, delta_rsrc)):
                    m0 = fx.Int64(m0_at(m0_base, lds_side - xg.a.lds_base + sslot * SIDE_SLOT + 256 * i))
                    lds_io.tag(rocdl.raw_ptr_buffer_load_lds(rsrc,
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
                lse = Vec(lds_io.load(side_rd + sslot * SIDE_SLOT, v4i, 16, f's{sslot}')).bitcast(fx.Float32)
                dlt = Vec(lds_io.load(side_rd + (sslot * SIDE_SLOT + 256), v4i, 16, f's{sslot}')).bitcast(fx.Float32)
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

            # ---- score hooks: slack columns, the band mask, bias, dropout keep bits
            def slack(h):
                """X read t of part h with its slack columns zeroed."""
                def fix(g, t, x8):
                    if const_expr(not tr.padded):
                        return x8
                    return Vec(masked8(x8, fx.Int32(32 * (h * NT_P + t)) + fx.Int32(gq * 8), hdim_own))
                return fix

            def keep_bits(q0, i):
                """Dropout keep of (q0 + 4gq + r, soft_kv[i]) for r < 4, as four i1."""
                kv4 = soft_kv[i] - fx.Int32(lc % 4)
                row = q0 + fx.Int32(gq * 4 + lc % 4)
                return deal_keep_bits(rng, ph_seed, ph_base, ph_stride, row, kv4, idropout_p, quad_shift)

            def p_ds(i, s_raw, dp_raw, lse, delta, bvals, q0, masked):
                """P (for dV) and dS packs of key column soft_kv[i]."""
                sv = Vec(s_raw) * fx.full(4, fx.Float32(c_scale), fx.Float32)
                dpv = Vec(dp_raw)
                mask_of = None
                if const_expr(masked):
                    # The lane term afresh per tile: hoisted, it holds a VGPR through the loop.
                    lt = fx.Int32(lane // 16) * fx.Int32(4) - fx.Int32(lane % 16)
                    lt = fx.Int32(pin_values([lt])[0])
                    rel0 = q0 + band_s[i] + lt

                    def mask_of(v, r, fill):
                        live = fx.Uint32(rel0 + fx.Int32(r)) <= fx.Uint32(band_span)
                        return live.select(v, fill)
                keep = None
                if const_expr(dropout):
                    keep = keep_bits(q0, i)

                def bias_of(r):
                    bv = Vec.from_elements([fx.Int16(bvals[4 * i + r])], fx.Int16).bitcast(elem)
                    return fx.Float32(Vec(bv)[0].to(fx.Float32))
                ps, dss = softmax.p_ds_elems(ctx,
                                             sv,
                                             dpv,
                                             lse_l2=lambda r: fx.Float32(lse[r]) * fx.Float32(LOG2E),
                                             delta=delta,
                                             bias_of=bias_of,
                                             mask_of=mask_of,
                                             keep=keep)
                return pack_operand(ps, elem), pack_operand(dss, elem)

            # ---- the team exchange: raw S/dP [sub][role] (xs), P/dS packs [sub][P, dS] (xp).
            # Its layout follows the teams' lane and role split, so it stays in this body.
            def xs_put(own):
                for sub in range_constexpr(NSUB):
                    lds_io.store(xs_at + fx.Index(role) * (WAVE * 16) + sub * XS_SUB, own[sub], 16, 'xs')

            def xs_get(i):
                sub_off = (fx.Index(2 * i) + fx.Index(role)) * XS_SUB
                return (lds_io.load(xs_at + sub_off, v4f(), 16, 'xs'),
                        lds_io.load(xs_at + sub_off + WAVE * 16, v4f(), 16, 'xs'))

            def xp_put(i, p_pk, ds_pk):
                xo = xp_at + (fx.Index(2 * i) + fx.Index(role)) * XP_SUB
                lds_io.store(xo, Vec(p_pk).bitcast(fx.Int32), 8, 'xp')
                lds_io.store(xo + WAVE * 8, Vec(ds_pk).bitcast(fx.Int32), 8, 'xp')

            def xp_get():
                """The B operand of Y: dS (K-wave) or P (V-wave) of every column."""
                pk_off = (fx.Index(1) - fx.Index(role)) * (WAVE * 8)
                return [lds_io.load(xp_at + (sub * XP_SUB) + pk_off, Vec.make_type(4, elem), 8, 'xp')
                        for sub in range(NSUB)]

            # ---- the tile loop: X, Z, Y of tile j; `ph` = j % period (trace time) picks the RM slots.
            def slot(p, h):
                return (PARTS * p + h) % tr.rm_slots

            def iteration(j, ph, masked):
                q0 = q0_of(j)
                j1 = j + fx.Int32(1)
                nxt = {}

                def side_next():
                    side_dma(j1, (ph + 1) % tr.period)
                    nxt['bias'] = bias_load(j1)
                n_side = 2 + (4 * soft_subs if bias else 0)

                # X: S = Q.K^T (K-wave) or dP = dO.V^T (V-wave) of tile j, part by part; part 0's
                # hooks DMA part 0 of RM(j+1) and LSE/Delta/bias(j+1). Then both raw tiles to xs.
                own = [zero4] * NSUB
                for h in range_constexpr(PARTS):
                    hooks = rows_dma(j1, 0, slot(ph + 1, 0)) + [side_next] if const_expr(h == 0) else []
                    sl_x = slot(ph, h)
                    own = gemm_stream_joint((xg,),
                                            flex_gemm.gemm_plan_at(xg, rm_plan, sl_x),
                                            lds_io,
                                            sl_x,
                                            ([[blk[h * NT_P + t] for t in range(NT_P)] for blk in res],),
                                            (own,),
                                            tr.prefetch,
                                            hooks,
                                            a_mod=slack(h),
                                            a_vgpr=tr.ops_vgpr)[0]
                    if const_expr(h + 1 < PARTS):
                        # Everything this part issued is newer than part 1 of RM(j) (Y(j-1)).
                        rocdl.s_waitcnt(vmcnt=n_dma + n_side)
                        cluster_barrier()
                xs_put(own)
                cluster_end()

                # Z: P and dS of key columns 2i + role from both raw tiles, packed to xp.
                lse, delta = side_read(ph)
                for i in range_constexpr(soft_subs):
                    s_raw, dp_raw = xs_get(i)
                    xp_put(i, *p_ds(i, s_raw, dp_raw, lse, delta, ctx.bias, q0, masked))
                cluster_end()

                # Y: dK^T += Q^T.dS (K-wave) or dV^T += dO^T.P (V-wave) of tile j, part by part,
                # A derived from RM; once part 0's slot is free, part 1 of RM(j+1) DMAs into it.
                pks = xp_get()
                for h in range_constexpr(PARTS):
                    hooks = rows_dma(j1, 1, slot(ph + 1, 1)) if const_expr(h == 1) else []
                    sl_y = slot(ph, h)
                    part = gemm_stream_joint((yg,),
                                             flex_gemm.gemm_plan_at(yg, y_plan, sl_y),
                                             lds_io,
                                             sl_y,
                                             (pks,),
                                             (ctx.acc[h * yg.n_acc:(h + 1) * yg.n_acc],),
                                             tr.y_ahead,
                                             hooks)[0]
                    ctx.acc = ctx.acc[:h * yg.n_acc] + part + ctx.acc[(h + 1) * yg.n_acc:]
                    if const_expr(h + 1 == PARTS):
                        # Part 0 of RM(j+1) (and LSE/Delta/bias) landed; part 1 may still fly.
                        rocdl.s_waitcnt(vmcnt=n_dma if PARTS == 2 else 0)
                    cluster_end()
                ctx.bias = nxt['bias']

            def step(j, carried, ph):
                ctx.restore(carried)
                iteration(j, ph, tr.causal)
                return ctx.carry()

            @flyc.jit
            def tail_a(j, carried):
                out = list(carried)
                if j < fx.Index(n_tiles):
                    out = step(fx.Int32(j), carried, 0)
                return out

            @flyc.jit
            def tail_b(j, carried):
                out = list(carried)
                if j < fx.Index(n_tiles):
                    out = step(fx.Int32(j), carried, 1)
                return out

            # Prologue: RM(0) DMA (every part), LSE/Delta/bias(0).
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()
            for h in range_constexpr(PARTS):
                for f in rows_dma(fx.Int32(0), h, h):
                    f()
            side_dma(fx.Int32(0), 0)
            b0 = bias_load(fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0)
            cluster_barrier()

            # Groups of `period` tiles (the slot pattern's), then up to period - 1 more.
            ctx.acc = list(accs)
            ctx.bias = b0
            init = ctx.carry()
            res1 = init
            for jp, carried in range(0, fx.Index(n_groups), 1, init=init):
                jg = fx.Int32(jp) * fx.Int32(tr.period)
                cur = carried
                for p in range_constexpr(tr.period - 1):
                    cur = step(jg + fx.Int32(p), cur, p)
                res1 = yield step(jg + fx.Int32(tr.period - 1), cur, tr.period - 1)
            j_rest = fx.Index(n_groups) * tr.period
            out = tail_a(j_rest, res1)
            if const_expr(tr.period == 3):
                out = tail_b(j_rest + fx.Index(1), out)
            return out[:n_acc]

        acc0 = [fx.full(4, 0.0, fx.Float32) for _ in range(n_acc)]
        results = acc0
        for g, carried in range(0, fx.Index(group), 1, init=acc0):
            results = yield head_pass(fx.Int32(g), carried)
        rocdl.s_waitcnt(vmcnt=0)

        # ---- epilogue: dK^T (K-wave, times sm_scale) or dV^T (V-wave; times 1/(1-p) if
        # dropout scales it here), transposed back by the store.
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
                                      PARTS,
                                      rsrc,
                                      ss,
                                      kvt,
                                      lc,
                                      gq,
                                      sk_i32,
                                      row_lim,
                                      fx.full(4, scale, fx.Float32),
                                      elem)

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

    _launch = host.make_dkdv_launch(launch_fmha_bwd_dkdv_gfx942_xwide,
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
