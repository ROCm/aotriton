# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Device helpers shared by the gfx942 attention kernels, and the host-side body
dispatch of their entry modules.

Kernels import these by name (`from .utils import x`): FlyDSL's JIT
cache hashes the source of same-directory helpers imported that way, not of
ones reached through a module alias. FlyDSL does not rewrite `if` here, so
every branch is on a Python value.
"""

import importlib

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import ktrace, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

import fmha_utils as fmha
from fmha_utils import bitcast_i32
from philox import Philox

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
WAVE = 64
OOB_VOFF = 0x7FFFFFF0   # a voffset past any range: the load returns 0
# sched_barrier mask: VALU, SALU, VMEM and DS writes may cross; DS reads and MFMAs may not.
PIN_MASK = 0x002 | 0x004 | 0x010 | 0x020 | 0x040 | 0x200 | 0x400
# sched_group_barrier masks
SG_VALU = 0x2
SG_MFMA = 0x8
SG_VMEM = 0x10
SG_DS_READ = 0x100
SG_DS_WRITE = 0x200


# MLIR types need a live context, so they are built at trace time.
def v4f():
    return Vec.make_type(4, fx.Float32)


def v16f():
    return Vec.make_type(16, fx.Float32)


def mfma_32x32x8(a, b, c, elem):
    """c + a.b on 32x32x8: a, b four `elem` each, c 16 f32."""
    if elem is fx.BFloat16:
        return rocdl.mfma_f32_32x32x8bf16_1k(v16f(), [Vec(a).bitcast(fx.Int16), Vec(b).bitcast(fx.Int16), c])
    return rocdl.mfma_f32_32x32x8f16(v16f(), [a, b, c])


def mfma_16x16x16(a, b, c, elem):
    """c + a.b on 16x16x16: a, b four `elem` each, c 4 f32."""
    if elem is fx.BFloat16:
        return rocdl.mfma_f32_16x16x16bf16_1k(v4f(), [Vec(a).bitcast(fx.Int16), Vec(b).bitcast(fx.Int16), c])
    return rocdl.mfma_f32_16x16x16f16(v4f(), [a, b, c])


def exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, raw(x)))

def bf16_round_nonan_pack(f32_vals, elem_dtype):
    """Pack an even count of f32 values into bf16, rounding to nearest, ties away.

    `gfx1201.utils.bf16_round_nonan_pack_v8` for any length: same rounding, same
    caveat (a NaN may come out as a zero, so only for values that cannot be NaN,
    such as P).
    """
    pairs = []
    for j in range(len(f32_vals) // 2):
        lo = bitcast_i32(f32_vals[j * 2]) + fx.Int32(0x8000)
        hi = bitcast_i32(f32_vals[j * 2 + 1]) + fx.Int32(0x8000)
        pairs.append(fx.Int32(rocdl.perm_b32(hi, lo, fx.Int32(0x07060302))))
    return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()


def pack_operand(vals, elem):
    """f32 values -> one MFMA operand. bf16 rounds to nearest, ties away, NaN-unsafe
    (bf16_round_nonan_pack; numerical lessons, #16); never truncate."""
    if elem is fx.BFloat16:
        return bf16_round_nonan_pack(vals, elem)
    return Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem).ir_value()


def smin(x, y):
    return (x < y).select(x, y)


def smax(x, y):
    return (x > y).select(x, y)


def pin_values(vals, rc='v'):
    """Opaque copies in register class `rc`: LLVM can neither look through them nor move them."""
    irs = [raw(v) for v in vals]
    if not irs:
        return []
    if len(irs) == 1:
        # LLVM aborts on a struct return from a one-output asm (gfx950.dualwave_helpers._anchor_v_o).
        return [llvm.inline_asm(irs[0].type, irs, '', f'={rc},0', has_side_effects=True)]
    ty = ir.Type.parse('!llvm.struct<(' + ', '.join(str(x.type) for x in irs) + ')>')
    cons = ','.join([f'={rc}'] * len(irs) + [str(i) for i in range(len(irs))])
    ret = llvm.inline_asm(ty, irs, '', cons, has_side_effects=True)
    return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(len(irs))]


def lds_ptr_at(base_i64):
    """An LDS pointer at byte address `base_i64`: the M0 operand of an LDS DMA."""
    return buffer_ops.create_llvm_ptr(fx.Int64(base_i64), address_space=3)


def alias_scopes(domain, scopes, name):
    """(alias_scopes, noalias_scopes) of LDS buffer `name` among `scopes`; without them
    LLVM waits vmcnt(0) for any DMA before any LDS read."""
    def arr(names):
        return ir.Attribute.parse('[' + ', '.join(
            f'#llvm.alias_scope<id = "{n}", domain = {domain}>' for n in names) + ']')
    return arr([name]), arr([n for n in scopes if n != name])


def set_alias_scopes(op, al, noal):
    """Tag an op (an LDS DMA) with `alias_scopes`' pair."""
    op = getattr(op, 'owner', op)
    op = getattr(op, 'operation', op)
    op.attributes['alias_scopes'] = al
    op.attributes['noalias_scopes'] = noal


class Lds:
    """A kernel's LDS buffer at trace time: typed views, alias-scoped loads and stores."""

    def __init__(self, ptr, elem, domain, scopes):
        self.ptr = ptr
        self.elem = elem
        self.domain = domain
        self.scopes = scopes

    def scope(self, name):
        return alias_scopes(self.domain, self.scopes, name)

    def view(self, off, e, align):
        return fx.recast_iter(fx.PointerType.get(e.ir_type, fx.AddressSpace.Shared, align), self.ptr + fx.Int64(off))

    def load(self, off, ty, align, scope):
        al, noal = self.scope(scope)
        return llvm.LoadOp(ty,
                           self.view(off, self.elem, align).llvm_ptr,
                           alignment=align,
                           alias_scopes=al,
                           noalias_scopes=noal).result

    def store(self, off, val, align, scope):
        """Store `val` (i32 words) at byte `off`."""
        al, noal = self.scope(scope)
        llvm.StoreOp(raw(val), self.view(off, fx.Int32, align).llvm_ptr, alignment=align, alias_scopes=al, noalias_scopes=noal)

    def tag(self, op, scope):
        """Tag an LDS DMA writing buffer `scope`."""
        set_alias_scopes(op, *self.scope(scope))


def cluster_barrier(name=None, barrier=True):
    """End of a cluster; under FLYDSL_KTRACE_ENABLE=1, a `name` marks its end and the barrier release."""
    rocdl.sched_barrier(0)
    if name is not None:
        ktrace.mark(name)
    if barrier:
        rocdl.s_barrier()
    if name is not None:
        ktrace.mark(name + '>')
    rocdl.sched_barrier(0)


def base_addr(ptr, batch, h, row_off, sb, sh, ss):
    """Byte address of row `row_off` of head `h` of batch `batch` (strides in elements)."""
    elems = fx.Int64(batch) * sb + fx.Int64(h) * sh + fx.Int64(row_off) * ss
    return fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(2)


def span_bytes(rows, ss, row_elems):
    """Bytes from a sequence's first row to its last row's last owned column
    (gfx950.dualwave_helpers._slab_span_elems): past it reads 0 and stores drop."""
    span = (fx.Int64(rows) - fx.Int64(1)) * ss + fx.Int64(row_elems)
    return (fx.Int32(rows) > fx.Int32(0)).select(span, fx.Int64(0)) * fx.Int64(2)


def rsrc_for(ptr, batch, h, row_off, sb, sh, ss, rows, row_elems):
    """A buffer resource over one sequence of one head (base_addr, span_bytes)."""
    return buffer_ops.create_buffer_resource_from_addr(base_addr(ptr, batch, h, row_off, sb, sh, ss),
                                                       num_records_bytes=span_bytes(rows, ss, row_elems))


def col_mask_words(c0, hdim):
    """Four dword masks keeping 16-bit columns c0 + [0, 8) below `hdim`."""
    words = []
    for w in range(4):
        lo = (c0 + fx.Int32(2 * w) < fx.Int32(hdim)).select(fx.Int32(0xFFFF), fx.Int32(0))
        hi = (c0 + fx.Int32(2 * w + 1) < fx.Int32(hdim)).select(fx.Int32(-65536), fx.Int32(0))
        words.append(lo | hi)
    return words


def col_mask(c0, hdim):
    """col_mask_words as one vector."""
    return Vec.from_elements(col_mask_words(c0, hdim), fx.Int32)


def apply_mask(v8, mask, elem):
    """Eight `elem` with a col_mask applied: slack columns become 0, NaN included."""
    return Vec(Vec(v8).bitcast(fx.Int32) & mask).bitcast(elem).ir_value()


def philox_stream(seed_ptr, offset1, offset2):
    """(rng, seed, offset base) of the Triton kernels' dropout stream (philox.py; width 32 on gfx942)."""
    rng = Philox.for_arch('gfx942')
    return rng, fmha.philox_seed_value(seed_ptr), fmha.philox_offset_base(offset1, offset2)


def philox_plane(rng, offset_base, z, num_head_q, head, max_seqlen_q, max_seqlen_k):
    """(base, stride) of query head `head` of batch or sequence `z` (philox.py grid_plane)."""
    return rng.grid_plane(offset_base, z * fx.Int32(num_head_q) + fx.Int32(head), max_seqlen_q, max_seqlen_k)


def barrier_if(flag_i32):
    """s_barrier on the waves whose wave-uniform `flag_i32` is nonzero. Clobbers SCC:
    undeclared, a live SCC across it misroutes a branch and the barriers mismatch."""
    llvm.inline_asm(ir.Type.parse('!llvm.void'),
                    [raw(flag_i32)],
                    's_cmp_eq_u32 $0, 0\n\ts_cbranch_scc1 1f\n\ts_barrier\n\t1:',
                    's,~{scc}',
                    has_side_effects=True)


def m0_at(base, const=0):
    """base + const in an SGPR at the use: hoisted, the DMAs' M0 values would pin SGPRs."""
    return fx.Int32(llvm.inline_asm(T.i32,
                                    [raw(fx.Int32(base))],
                                    f's_add_u32 $0, $1, {const}',
                                    '=s,s,~{scc}',
                                    has_side_effects=True))


def sgb(mask, n):
    """sched_group_barrier for `n` > 0 instructions of `mask`; nothing for 0."""
    if n > 0:
        rocdl.sched_group_barrier(mask, n, 0)


def sched_spread(n_mfma, n_rd=0, n_wr=0, n_vmem=0, n_valu=0, valu_per_mfma=None, lead=0):
    """Ask the scheduler for LDS reads (`lead` of them first), LDS writes, VMEM and VALU
    spread evenly over `n_mfma` MFMAs; `valu_per_mfma` puts that many VALU after each."""
    sgb(SG_DS_READ, lead)
    n_rd -= lead
    for i in range(n_mfma):
        sgb(SG_MFMA, 1)
        sgb(SG_DS_READ, (i + 1) * n_rd // n_mfma - i * n_rd // n_mfma)
        sgb(SG_DS_WRITE, (i + 1) * n_wr // n_mfma - i * n_wr // n_mfma)
        sgb(SG_VMEM, (i + 1) * n_vmem // n_mfma - i * n_vmem // n_mfma)
        if valu_per_mfma is None:
            sgb(SG_VALU, (i + 1) * n_valu // n_mfma - i * n_valu // n_mfma)
        else:
            sgb(SG_VALU, valu_per_mfma)


def pinned():
    """Fix the order of LDS reads and MFMAs; everything else may cross (PIN_MASK)."""
    rocdl.sched_barrier(PIN_MASK)


def pipelined(n, ahead, read, use, hooks=()):
    """Software-pipelined LDS operands: read t + `ahead`, then use t, each step pinned.
    `hooks` (other work) run spread over the steps, in order."""
    at = {}
    for i, h in enumerate(hooks):
        at.setdefault(min(n - 1, (2 * i + 1) * n // (2 * len(hooks))), []).append(h)
    pending = [read(t) for t in range(min(ahead, n))]
    for t in range(n):
        if t + ahead < n:
            pending.append(read(t + ahead))
        pinned()
        use(t, pending[t])
        for h in at.get(t, []):
            h()
        pinned()


def band_window(causal, window_left, window_right, sq, sk, varlen_bits):
    """(win_l, win_r): row i keeps key j iff 0 <= j < sk and i - win_l <= j <= i + win_r
    (gSWA, docs/Generalized Sliding Window Attention.md, A-2), resolved per sequence.
    Without a mask, unbounded: no row index exceeds sq, no column sk."""
    if causal:
        return fmha.resolve_window(window_left, window_right, sq, sk, varlen_bits)
    return sq, sk


def kv_tiles(q_start, block_m, block_n, sq, sk, win_l, win_r):
    """(q_first, q_last, kv_base, n_tiles) of the q tile at `q_start`: its keys' tiles start
    at kv_base. A tile past a short sequence (varlen), or whose band misses every key, has none."""
    q_first = fx.Int32(q_start)
    q_last = smin(q_first + fx.Int32(block_m), sq) - fx.Int32(1)
    kv_lo = smax(q_first - win_l, fx.Int32(0))
    kv_hi = smin(q_last + win_r + fx.Int32(1), sk)
    alive = (q_first < sq) & (kv_hi > kv_lo)
    t0 = fx.Index(alive.select(kv_lo, fx.Int32(0))) // block_n
    kv_base = t0 * block_n
    n_tiles = alive.select((fx.Index(kv_hi) + block_n - 1) // block_n - t0, fx.Index(0))
    return q_first, q_last, kv_base, n_tiles


def row_band(row, win_l, win_r, sk):
    """(band_lo, band_span): row `row` keeps keys [band_lo, band_lo + band_span], one unsigned
    compare per key. An empty band moves band_lo out of reach of any key, so all fail."""
    row = fx.Int32(row)
    hi = smin(row + win_r, sk - fx.Int32(1))
    lo = smax(row - win_l, fx.Int32(0))
    empty = hi < lo
    return empty.select(fx.Int32(-(1 << 30)), lo), empty.select(fx.Int32(0), hi - lo)


def clean_tiles(q_first, q_last, kv_base, block_n, sk, win_l, win_r):
    """(lead, clean_end): tile r from kv_base is clean, unmasked for every row of the q tile,
    iff lead <= r < clean_end."""
    def n_rel(kv):
        return fx.Index(smax(smin(kv, sk), fx.Int32(0)))

    lead = (n_rel(q_last - win_l - fx.Int32(kv_base)) + block_n - 1) // block_n
    clean_end = n_rel(smin(q_first + win_r, sk - fx.Int32(1)) + fx.Int32(1) - fx.Int32(kv_base)) // block_n
    return lead, clean_end


def mask_band(s_vals, rel0, offsets, band_span, fill):
    """s_vals[r] where key rel0 + offsets[r] (relative to band_lo) is in the band, else `fill`."""
    return [(fx.Uint32(rel0 + fx.Int32(off)) > fx.Uint32(band_span)).select(fill, s) for s, off in zip(s_vals, offsets)]


def bias_slab(ptr, batch, head, q_row_off, k_row_off, sb, sh, ss, sq, sk):
    """(raw buffer resource, misalignment in bytes) of one head's (Sq, Sk) bias slab, Sk
    contiguous. Rows offset by Q's varlen start, columns by K's (as gfx1201). Odd-pitch rows
    start mid-dword, so the base is aligned down. The range ends on the slab's last element
    and is never rounded up: the caller owns nothing past it
    (gfx950.dualwave_helpers._bias_slab_num_records_bytes). A dword not wholly in range reads 0,
    so a read that can reach that element must go 2 bytes at a time (bias_load4's `exact`)."""
    elems = fx.Int64(batch) * sb + fx.Int64(head) * sh + fx.Int64(q_row_off) * ss + fx.Int64(k_row_off)
    addr = fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(2)
    mis = fx.Int32(addr & fx.Int64(3))
    span = smax(fx.Int64(sq - fx.Int32(1)) * ss + fx.Int64(sk), fx.Int64(0))
    # Op order kept as the bodies had it: register allocation follows it.
    base = addr - fx.Int64(mis)
    rsrc = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=span * fx.Int64(2) + fx.Int64(mis))
    return rocdl.get_buffer_rsrc(rsrc), mis


def funnel_words(dw, shift, n=2):
    """`n` dwords starting `shift` bits into the dwords `dw` (llvm.fshr)."""
    return [fx.Int32(llvm.call_intrinsic(T.i32, 'llvm.fshr.i32', [raw(dw[w + 1]), raw(dw[w]), raw(shift)], [], []))
            for w in range(n)]


def bias_load4(rsrc, off, soff, shift, exact, elem):
    """Four 16-bit bias values. `exact`: four 2-byte loads at byte `off`; otherwise three
    dwords at the aligned `off`, funnel-shifted by `shift` bits."""
    if exact:
        hs = [rocdl.raw_ptr_buffer_load(T.i16, rsrc, off + fx.Int32(2 * e), soff, 0) for e in range(4)]
        return Vec.from_elements([fx.Int16(h) for h in hs], fx.Int16).bitcast(elem).ir_value()
    dw = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(3, fx.Int32), rsrc, off, soff, 0))
    return Vec.from_elements(funnel_words(dw, shift), fx.Int32).bitcast(elem).ir_value()


def deal_keep_bits(rng, seed, base, stride, row, kv4, threshold, shift, n=4):
    """Dropout keep flags dealt over a lane quad. The lane draws (row, kv4 + [0, 4)); a DPP
    broadcast gives every lane its quad's four draws, and after `>> shift` flag r is bit 4r."""
    vals = rng.u32(seed, rng.grid_offset(base, stride, row, kv4))
    bits = fx.Int32(0)
    for s in range(4):
        bits = bits | (fx.Int32(vals[s]) > fx.Int32(threshold)).select(fx.Int32(1 << s), fx.Int32(0))
    word = fx.Int32(0)
    for r in range(4):
        got = fx.Int32(rocdl.update_dpp(T.i32, raw(fx.Int32(0)), raw(bits), 0x55 * r, 0xF, 0xF, True))
        word = word | (got << fx.Int32(4 * r))
    word = word >> shift
    return [(word & fx.Int32(1 << (4 * r))) != fx.Int32(0) for r in range(n)]


def body_module(bodies, body):
    """The module of `body` in `bodies`, an entry module's body table."""
    return importlib.import_module(bodies[body][0])


def body_builder(bodies, body):
    """The builder of `body` in `bodies`, an entry module's body table."""
    return getattr(body_module(bodies, body), bodies[body][1])


def build_body_primary(bodies, meta, knobs):
    """The body `knobs.body` names in `bodies`, built for a resolved (meta, knobs)
    pair; the table lists the knob fields each body's builder takes.

    The builders take AOTriton's causal_type as is: 1/2 are host sentinels, 3 is
    gSWA, and all three compile to 3."""
    if knobs.body not in bodies:
        raise ValueError(f'knobs.body is {knobs.body!r}: pass knobs resolved for meta')
    return body_builder(bodies, knobs.body)(meta.block_dmodel,
                                            padded_head=meta.padded_head,
                                            grid_axis_order=knobs.GRID_AXIS_ORDER,
                                            causal_type=meta.causal_type,
                                            dtype_str=meta.dtype_str,
                                            bias=meta.bias,
                                            dropout=meta.dropout,
                                            **{f: getattr(knobs, f) for f in bodies[knobs.body][2]})
