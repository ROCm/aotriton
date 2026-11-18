# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Helpers shared by every arch's flyc attention kernels.

Keeps the `fmha_` prefix: arch modules import this by bare name, and as
`utils.py` it would be shadowed by `gfx1201/utils.py` whenever that directory
is on `sys.path`. Only root modules no arch module imports (the `tuning_*.py`
dispatchers) share a name with an arch module.

Two halves. The device half emits IR: column masking (`MaskedAxis`),
varlen and LSE addressing, sliding-window resolution, and the Philox
seed/offset plumbing. Helpers that branch at run time are `@flyc.jit`, so
FlyDSL rewrites their Python `if` into `scf.if`. The host half is the attention ABI: plain Python that turns
torch tensors and Python arguments into the kernel signature's scalars
and pointers before a launch, and emits no IR.

Everything the *caller* has to get right, in one place. Every flyc kernel on
every arch shares this ABI, and before this module existed each kernel carried
its own copy of all of it. That is not merely repetitive:
`VarlenBits` is a **wire encoding**, and three copies of a wire encoding are
three chances for it to drift while every test still passes, because each
kernel is validated against its own copy.

The three functions that take a build knob as their first argument
(`window_args`, `dropout_args`, `varlen_args`) are the only ones that are
not pure. They were closures over the builder's `const_expr` state; making the
dependency an argument is what let them move, and it also makes visible which
host behaviour a knob reaches -- something the closure form hid.
"""

from __future__ import annotations

import contextlib
import math
import weakref
from dataclasses import dataclass

# `common_utils` aliases the polyfill for `ssel` and friends, as in
# `gfx1201/utils.py`.
import flyc_polyfill as common_utils
from philox import dropout_threshold

# `torch` and `torch.float32` are imported lazily, inside the three functions
# that need them, because the build venv must never have torch and neither use
# is reached by the AOT compile driver, which passes philox_seed=None.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec

__all__ = [
    "lse_row_addressing",
    "bitcast_i32",
    "MaskedAxis",
    "cond_load",
    "seqinfo_addr",
    "decode_addressing",
    "lse_token_pitch",
    "WINDOW_TOPLEFT",
    "WINDOW_BOTRIGHT",
    "resolve_window",
    "CausalRegions",
    "decompose_causal_regions",
    "philox_offset_base",
    "philox_seed_value",
    "philox_report",
    "VARLEN_STACKED",
    "VARLEN_LENGTH_MAX",
    "VARLEN_LENGTH_CUMULATIVE",
    "VARLEN_LENGTH_INDIVIDUAL",
    "VARLEN_POSITION_IMPLIED",
    "VARLEN_POSITION_REUSE",
    "VARLEN_POSITION_ARRAY",
    "VARLEN_LSE_LAYOUT_HT",
    "VARLEN_LSE_LAYOUT_TH",
    "VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT",
    "VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT",
    "VARLEN_DENSE",
    "VARLEN_COMPACT_SIDE",
    "CAUSAL_SENTINEL",
    "varlen_bits",
    "varlen_dense",
    "varlen_is_dense",
    "dense_row_shape",
    "varlen_compact",
    "varlen_padded",
    "varlen_strided",
    "varlen_seqused_k",
    "NULL_PTR",
    "ptr_arg",
    "strides_of",
    "row_tensor_arg",
    "resolve_scale",
    "prep_tensors",
    "lse_args",
    "window_args",
    "bias_args",
    "dropout_args",
    "dropout_outputs",
    "u64_scalar",
    "varlen_args",
    "COMPILE_HINTS",
    "kernel_attrs",
    "run_compiled",
    "new_compiled_cache",
    "dtype_to_elem_type",
]


# --------------------------------------------------------------------------
# Device helpers
# --------------------------------------------------------------------------


def lse_row_addressing(varlen_bits, batch, head, num_head_q, tokens, row_off):
    """`(base, pitch)` for a row-wise f32 side input -- logsumexp, and delta.

    The element for absolute query row `r` is at `base + r * pitch`. Factored
    that way because `base` is loop-invariant while `r` is not: the four
    callers that had open-coded this each recomputed the whole offset per row.

    Two layouts, one select, decoded from VarlenBits bits 17:16:

        _HT   (H, T), T contiguous -- AOTriton's and this kernel's default.
              base = (batch * H + head) * tokens + row_off,   pitch = 1
        _TH   (T, H), H contiguous -- Transformer Engine's.
              base = (batch * tokens + row_off) * H + head,   pitch = H

    `tokens` is `lse_token_pitch`'s answer, not `max_seqlen_q`: a stacked
    layout runs to the batch total rather than padding each row-group.

    **Delta is required to share LSE's layout.** It is produced beside it by
    the same caller, and giving it its own decode would double the work for no
    expressiveness -- so both side inputs use this one function.

    Four callers of one fact: the forward *writes* LSE through it, and all
    three backward kernels *read* LSE and delta through it. It was two
    spellings and a third that had already factored out the base, which is the
    form kept here.
    """
    tok = fx.Index(tokens)
    nhq = fx.Index(num_head_q)
    base_ht = (batch * nhq + head) * tok + row_off
    base_th = (batch * tok + row_off) * nhq + head
    is_th = ((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)) != fx.Int32(0)
    base = fx.Index(is_th.select(fx.Index(base_th), fx.Index(base_ht)))
    pitch = fx.Index(is_th.select(nhq, fx.Index(1)))
    return base, pitch


def lse_row_step_per_head(varlen_bits, tokens):
    """How far `lse_row_addressing`'s `base` moves when `head` rises by one.

    `base` is affine in `head` under both layouts -- `_HT` has it inside the
    `* tokens` product, `_TH` outside it -- so a caller walking a GQA group can
    add this per head instead of re-running the decode. That is not a peephole:
    it is what stops `varlen_bits`, `batch` and `row_off` from being live across
    the walk, and on gfx950 that liveness is what drives the backward kernel's
    scalar spilling. See `BwdDkDvKernelContext._init_q_head_invariants`.

    Exactly the mirror of the `pitch` select above -- `_HT` pitches by 1 and
    steps by `tokens`, `_TH` pitches by `H` and steps by 1 -- so the two are
    written the same way, from the same bits, and cannot drift apart.

    `num_head_q` is deliberately not a parameter: it does not appear in either
    step, and taking it would suggest the caller must keep it live.
    """
    tok = fx.Index(tokens)
    is_th = ((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)) != fx.Int32(0)
    return fx.Index(is_th.select(fx.Index(1), tok))


def bitcast_i32(value):
    return fx.Float32(value).bitcast(fx.Int32)


class MaskedAxis:
    """One axis of a tile whose index can run past the real extent.

    Out-of-range indices always need two things, and keeping them together is
    the point: an address that is *safe to issue*, and a way to *discard what
    it returns*. Issuing the access unconditionally and throwing the value away
    beats branching around it, but only once the address has been redirected
    somewhere legal -- element 0 of the axis, which always exists.

    **One class covers rows and columns**, because the apparent difference
    between them is not about the axes. An access reads `width` contiguous
    elements *along one axis*; for that axis the extent boundary can fall
    inside the access, so validity is per element. For every other axis the
    index is a single scalar and the whole access stands or falls together.
    In this kernel the vector runs along the column axis, which is why columns
    look "per element" and rows "whole" -- but that is a property of the
    access, not of the axis, and `valid(idx)` is exactly `mask(idx, 1)`.

    `active=False` compiles the masking away, for an axis whose extent is known
    to be a multiple of the access width.
    """

    __slots__ = ("extent", "active", "elem_dtype", "bitmask")

    def __init__(self, extent, active=True, elem_dtype=None, bitmask=False):
        self.extent = extent
        self.active = active
        # See `discard`. Off by default: it trades in-loop instructions for
        # live registers, and only the caller knows whether that is affordable.
        self.bitmask = bitmask
        # Only `discard` needs this, so the row axis leaves it unset. It is a
        # property of the tensor and every access along an axis shares it.
        #
        # `width` is deliberately *not* bound here. It belongs to the access,
        # not the axis, and the two accesses along the QK column axis agree
        # only by coincidence: the cooperative loads are `VEC_WIDTH` wide while
        # the Q preload is 8 wide because `load_global_v8f16` is tied to the
        # WMMA operand shape. Both are 8 today from unrelated definitions, and
        # binding one would make the Q preload silently follow `VEC_WIDTH`.
        self.elem_dtype = elem_dtype

    def _bound(self):
        """The extent as a traced value.

        Resolved lazily so the object can be built on the host when the extent
        is a `const_expr` int.
        """
        return fx.Index(self.extent) if isinstance(self.extent, int) else self.extent

    def valid(self, idx):
        """`fx.Boolean`: is this index inside the extent?

        A signed compare, on every axis. Signedness is a property of the
        *type* -- `_make_binop` reads each operand's class-level `signed` --
        so `fx.Int64` is what makes `<` emit `slt` here, and it is also why
        this cannot simply compare the `fx.Index` values it is handed:
        `fx.Index` is unsigned and would give `ult`. The answer agrees, since
        every index and extent is non-negative, but the two are not the same
        code and mixing them per axis was a difference with no reason behind
        it.

        Spelled with the operator rather than `arith.cmpi(CmpIPredicate.slt,
        ...)` because `cmpi` is stable while `CmpIPredicate` is not, and
        because the `fx.Boolean` this returns has a stable `select`. The
        `index_cast` that `fx.Int64` inserts folds away -- index is already
        64-bit here -- which was measured, not assumed.
        """
        return fx.Int64(idx) < fx.Int64(self._bound())

    def mask(self, idx, width):
        """i1 vector, element j set iff `idx + j` is inside the extent.

        Built from a loop-invariant index at every current caller, so it hoists
        out of the KV loop and costs one vector select per access inside it.
        """
        return Vec.from_elements(
            [self.valid(idx + fx.Index(j)) for j in range_constexpr(width)],
            fx.Boolean,
        )

    def safe(self, idx, addressed=None):
        """`addressed` if `idx` is inside the extent, else 0.

        `addressed` defaults to `idx`, which is the column case. Rows need the
        two to differ: the bound is on the *absolute* row, `start_q + ...`,
        while the address is built from the row's offset *within the tile*, so
        the tested and the redirected quantity are not the same value.
        """
        if addressed is None:
            addressed = idx
        if not self.active:
            return addressed
        return fx.Index(self.valid(idx).select(addressed, fx.Index(0)))

    def discard(self, vec, idx, width):
        """Zero the elements of `vec` whose index is past the extent.

        **AND with a precomputed bit mask, not a per-element select**, whenever
        the elements are 16-bit and pair up evenly into dwords.

        The obvious spelling -- `mask(idx, width).select(vec, zeros)` -- asks
        for a *per-element* choice over 16-bit lanes that live packed two to a
        VGPR, so the backend has to take them apart and put them back. Measured
        on a head_dim 32 build in the 64 tile, against an unpadded 64:

            v_cndmask_b32   +419
            v_lshrrev_b32   +208     <- pure repacking
            v_perm_b32      +192     <- pure repacking
            total          +1022 instructions (3585 against 2563)

        Two thirds of that is the register file being shuffled rather than any
        masking work. A dword whose two halves are `0xFFFF` or `0` expresses the
        same choice with one `v_and_b32` and no repacking, and the mask itself
        is loop-invariant -- it depends on the extent and a loop-invariant
        column base -- so it hoists out and costs nothing inside the loop.

        Exact for any extent, including one that splits a pair: the two halves
        of a dword carry independent masks.

        **Opt-in, because it is not free and not always a win.** The masks are
        loop-invariant, so they hoist -- and then stay *live* for the whole
        loop, `width/2` registers per masked access. Measured on the gfx950
        parity kernel, that is 16 registers in the 64-wide tile and 32 in the
        128-wide one, and the 128 build already sits at 238 of 256:

            tile    spills   TFLOP/s at the padded rungs
             64        0     +21% (head_dim 16/32/48)
            128       61     -43% (head_dim 80/96/112)

        So the caller decides. `bitmask=False` keeps the per-element select,
        which is also the path the fp8 and odd-width callers were verified on.
        """
        if not self.active:
            return vec
        if not self.bitmask or self.elem_dtype is None or self.elem_dtype.width != 16 or width % 2:
            zeros = Vec.filled(width, 0.0, self.elem_dtype)
            return self.mask(idx, width).select(Vec(vec), zeros).ir_value()

        keep_lo = fx.Int32(0x0000FFFF)
        keep_hi = fx.Int32(0xFFFF0000)
        zero_i32 = fx.Int32(0)
        dwords = [
            self.valid(idx + fx.Index(2 * d)).select(keep_lo, zero_i32)
            | self.valid(idx + fx.Index(2 * d + 1)).select(keep_hi, zero_i32)
            for d in range_constexpr(width // 2)
        ]
        raw = Vec(vec).bitcast(fx.Int32)
        return (raw & Vec.from_elements(dwords, fx.Int32)).bitcast(self.elem_dtype).ir_value()

    def gate(self, idx, addressed=None):
        """`(valid(idx), safe(idx, addressed))` -- the two halves together.

        An out-of-range index needs both, always, and returning them as a pair
        is what stops a caller taking the address and forgetting the flag.
        """
        return self.valid(idx), self.safe(idx, addressed)


# --------------------------------------------------------------------------
# Varlen prologue: VarlenBits -> per-sequence addressing
# --------------------------------------------------------------------------


@flyc.jit
def cond_load(cond, addr, default):
    """Load i32 from `addr` when `cond`, else `default`. The load is skipped.

    A real `scf.if`, not a select, and that is the point: the sequence-info
    pointers are **null** whenever their mode is off, so a select -- which
    evaluates both arms -- would fault. Inside the region the load is never
    issued; verified against a null pointer.

    `addr` is computed by the caller and may be derived from a null pointer --
    address arithmetic touches no memory.
    """
    value = fx.Int32(default)
    if cond:
        value = fx.Int32(fx.ptr_load(addr, fx.Int32))
    return value


def philox_offset_base(offset1, offset2):
    """`offset2 + *offset1` -- the Philox counter, split the way torch splits it.

    `at::cuda::PhiloxCudaState` carries the offset two ways. Outside a graph
    capture it is an immediate. Under capture the counter has to advance
    between replays, so it lives in device memory that the graph re-reads,
    while the per-call increment stays baked in as an immediate;
    `at::cuda::philox::unpack` is `*offset_.ptr + offset_intragraph_`. Passing
    one pre-summed scalar instead is what freezes a captured graph onto a
    single dropout mask, because the sum is done once at capture time and the
    replay never sees the counter move.

    AOTriton spells the pair `philox_offset1` (the pointer) and
    `philox_offset2` (the immediate), and this is the same ABI so that the two
    are drop-in for each other.

    A null `offset1` is the uncaptured case. The load is *skipped*, not
    selected away -- `select` evaluates both arms and would fault.
    """
    return fx.Int64(offset2) + _load_u64_or_zero(offset1)


def philox_seed_value(seed_ptr):
    """`*seed_ptr`, or 0 when null. The seed side of the same graph story.

    A captured graph must see the seed move too, so torch keeps it in device
    memory exactly as it keeps the offset counter, and AOTriton takes it as
    `philox_seed_ptr` rather than a value. Splitting the offset but leaving the
    seed an immediate would give a replay a moving counter under a frozen key,
    which is not the same stream torch's own RNG would have produced.

    Null reads as 0 rather than faulting, matching AOTriton's `dropout_rng`.
    """
    return _load_u64_or_zero(seed_ptr)


@flyc.jit
def philox_report(seed_output, offset_output, seed, offset_base):
    """Write back the `(seed, offset)` this launch actually drew from.

    Only the backward can say why this exists. It has to regenerate the
    forward's stream, and under graph capture the effective offset is
    `*offset1 + offset2` -- a sum formed *on the device*, from a counter the
    host cannot read without synchronising. So the forward records what it
    used and the backward is handed that, instead of both sides trying to
    re-derive it and being wrong in different ways.

    One workgroup stores, not all of them: every workgroup computed the same
    two values, so the rest would be writing the same bytes to the same two
    addresses for no reason. `block_idx` raw rather than the flipped
    `q_tile_idx`, matching AOTriton's `program_id` guard -- which workgroup is
    designated does not matter, only that exactly one is.

    Either output may be null, which is how a caller says it does not want the
    value; both are skipped independently.
    """
    first = (
        (fx.Index(gpu.block_idx.x) == fx.Index(0))
        & (fx.Index(gpu.block_idx.y) == fx.Index(0))
        & (fx.Index(gpu.block_idx.z) == fx.Index(0))
    )
    if first:
        _store_u64_if_nonnull(seed_output, seed)
        _store_u64_if_nonnull(offset_output, offset_base)


@flyc.jit
def _load_u64_or_zero(ptr):
    """`*ptr` as an i64, or 0 when `ptr` is null.

    The load is *skipped*, not selected away -- `select` evaluates both arms
    and would fault on the null.
    """
    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    value = fx.Int64(0)
    if nonnull:
        value = fx.Int64(fx.ptr_load(fx.recast_iter(_i64_global_ptr_ty(), ptr), fx.Int64))
    return value


@flyc.jit
def _store_u64_if_nonnull(ptr, value):
    nonnull = fx.Int64(fx.ptrtoint(ptr)) != fx.Int64(0)
    if nonnull:
        fx.ptr_store(fx.Int64(value), fx.recast_iter(_i64_global_ptr_ty(), ptr))


def _i64_global_ptr_ty():
    """A u64 counter in global memory. Alignment 8, for the same reason as i32."""
    return fx.PointerType.get(
        elem_ty=fx.Int64.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=8,
    )


def _i32_global_ptr_ty():
    """An i32 pointer into global memory, alignment 4.

    Spelled out rather than `fx.recast_iter(fx.Int32, ptr)`, which inherits the
    source pointer's alignment: a kernel argument arrives as `u8` with
    alignment 1, and the shorthand then raises "alignment must be a positive
    multiple of element byte size (4), got 1". Same construction as
    `kernels/moe/moe_a8w4_mxscale_gfx1250.py` and
    `kernels/gemm/mxfp4_preshuffle.py`.

    A function, not a module constant: `PointerType.get` needs an MLIR context,
    and at import time there is none.
    """
    return fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=4,
    )


def seqinfo_addr(ptr, index):
    """`&ptr[index]` for an i32 sequence-info array. No memory is touched.

    A typed `!fly.ptr`, not an `!llvm.ptr`: that is what lets `cond_load` read
    it with the stable `fx.ptr_load` instead of a raw `llvm.LoadOp`.
    """
    return fx.recast_iter(_i32_global_ptr_ty(), ptr) + fx.Int64(index)


def decode_addressing(varlen_bits, bits_shift, max_seqlen, s0, s1, z):
    """One side of VarlenBits: where this workgroup's sequence lives.

    Returns `(seqlen, row_off, batch)` -- how long this sequence is, which row
    it starts at, and which batch index to use. Called once for Q and once for
    K. See section 3.1 of `sdpa-varlen-plan.md`; the axes are STACKED (bit 0),
    LENGTH (bits 2:1) and POSITION (bits 4:3).

    The LSE token pitch is *not* here, though it decodes from the same bits: it
    describes the logsumexp output rather than where Q or K live, and only the
    Q side needs it. See `lse_token_pitch`.

    Every load goes through `cond_load`, so the shape is flat -- fetch what
    each mode might need, then select. `s0[z]` serves both length modes and,
    under REUSE, the position too, which is why three loads cover five modes.
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    lenmode = (bits >> fx.Int32(1)) & fx.Int32(3)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)

    cumulative = lenmode == fx.Int32(1)
    individual = lenmode == fx.Int32(2)
    reuse = posmode == fx.Int32(1)  # position already read as `cur`
    array = posmode == fx.Int32(2)  # position from its own array
    zero = fx.Int32(0)

    cur = cond_load(lenmode != zero, seqinfo_addr(s0, z), zero)
    nxt = cond_load(cumulative, seqinfo_addr(s0, z + fx.Int32(1)), zero)
    pos = cond_load(array, seqinfo_addr(s1, z), zero)

    seqlen = common_utils.ssel(
        cumulative,
        nxt - cur,
        common_utils.ssel(individual, cur, fx.Int32(max_seqlen)),
    )
    row_off = common_utils.ssel(
        array,
        pos,
        common_utils.ssel(
            reuse,
            cur,
            common_utils.ssel(stacked, z * fx.Int32(max_seqlen), zero),
        ),
    )
    # **An empty sequence gets row 0 of the tensor, not row 0 of itself.**
    #
    # A zero-length sequence has no row to point at. Under a packed layout a
    # *trailing* one puts `row_off` at exactly `total_tokens`, so even row 0 of
    # that sequence is one past the last row of the tensor -- and every clamp
    # downstream is powerless, because `MaskedAxis.safe` redirects to element 0
    # *of the axis* and an empty axis has none. Three kernels read past the end
    # of Q, dO and K/V that way; two of the three were reproduced as HIP
    # memory-access faults whose address was bit-for-bit the end of the tensor.
    #
    # Redirecting the offset itself is what there always is a legal answer for:
    # row 0 of the whole tensor exists whenever any workgroup runs at all, since
    # a zero-row tensor makes the grid empty.
    #
    # Nothing else needs to change, because a workgroup on an empty sequence
    # already does no work: every loop is zero-trip (`decompose_causal_regions`
    # inverts its range when `alive` is false) and every store is guarded. The
    # prologue preloads were the one access that ran regardless, and their
    # results are discarded -- so making the address legal is the whole fix,
    # and skipping the transaction as well would buy nothing.
    #
    # Inert for every non-empty sequence, and for the padded layouts where
    # `row_off` is already zero.
    row_off = common_utils.ssel(seqlen > zero, row_off, zero)
    batch = common_utils.ssel(stacked, zero, z)
    return seqlen, row_off, batch


def lse_token_pitch(varlen_bits, bits_shift, max_seqlen, s0, s1, num_seqlens):
    """Row pitch of the logsumexp output, in tokens. Q side only.

    Batched layouts pad every row-group to `max_seqlen`; stacked ones run to
    the batch total, which lives in slot [N] of whichever array supplies
    positions -- the prefix-sum assumption of plan section 9.4, asserted host
    side.

    Derived from the bits rather than passed because the logsumexp tensor,
    alone among the tensors here, is always compact: its strides are a function
    of the bits, and passing them would be a second source of truth for one
    fact (plan section 4.2).
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)
    reuse = posmode == fx.Int32(1)
    array = posmode == fx.Int32(2)
    zero = fx.Int32(0)

    total_s0 = cond_load(stacked & reuse, seqinfo_addr(s0, num_seqlens), zero)
    total_s1 = cond_load(stacked & array, seqinfo_addr(s1, num_seqlens), zero)
    return common_utils.ssel(
        stacked,
        common_utils.ssel(
            reuse,
            total_s0,
            common_utils.ssel(array, total_s1, fx.Int32(num_seqlens) * fx.Int32(max_seqlen)),
        ),
        fx.Int32(max_seqlen),
    )


# --------------------------------------------------------------------------
# Sliding-window attention: resolving the window, and cutting the KV range
# --------------------------------------------------------------------------

WINDOW_TOPLEFT = -2147483647  # 0x80000001
WINDOW_BOTRIGHT = -2147483646  # 0x80000002


VARLEN_PRINCIPAL_DIAGONAL_SHIFT = 18  # VarlenBits bit 18: principal_diagonal_choice


def resolve_window(window_left, window_right, seqlen_q, seqlen_k, varlen_bits):
    """`(window_left, window_right)` with the causal sentinels resolved.

    `Window_left` / `Window_right` may carry `WINDOW_TOPLEFT` or
    `WINDOW_BOTRIGHT` instead of a literal bound, and they are resolved
    against *this sequence's* lengths rather than on the host. That is the
    whole reason the sentinels exist: host resolution works only when there is
    one length to resolve against, and under varlen bottom-right needs
    `seqlen_k[z] - seqlen_q[z]`, which differs per sequence. Matches
    AOTriton's `parse_window`.

    `varlen_bits` carries `principal_diagonal_choice` (bit 18): 0 measures a
    numeric bound from the top-left diagonal, 1 from this sequence's
    bottom-right one (`left - d`, `right + d` with `d = seqlen_k - seqlen_q`).
    See docs/Generalized Sliding Window Attention.md, A-2.

    Both sentinels give an unbounded left edge -- no row reaches further back
    than the start of its own sequence -- so they differ only in the right one.

    **Everything derived from a window stays i32.** Window bounds go negative;
    that is what a sentinel and a leading masked region are. `fx.Int32` is
    signed, so `<`/`>` emit `slt`/`sgt`, while `fx.Index` is unsigned and
    64-bit -- widening any of these even once makes the same comparison
    unsigned and a negative bound comes out enormous.
    """
    left = fx.Int32(window_left)
    right = fx.Int32(window_right)
    left_is_sentinel = (left == fx.Int32(WINDOW_TOPLEFT)) | (left == fx.Int32(WINDOW_BOTRIGHT))
    right_is_sentinel = (right == fx.Int32(WINDOW_TOPLEFT)) | (right == fx.Int32(WINDOW_BOTRIGHT))
    # Principal diagonal (VarlenBits bit 18). A NUMERIC bound is measured from
    # the top-left diagonal j = i when the bit is 0, and from this sequence's
    # bottom-right diagonal j = i + (seqlen_k - seqlen_q) when it is 1, which is
    # flash-attention's convention. Sentinels already name their diagonal and
    # are never shifted (shifting BOTRIGHT would give 2 * (seqlen_k - seqlen_q)).
    # Arithmetic rather than a select on the bit: shift is 0 or d.
    # Known limitation, by design: the shifted bounds are i32 with no overflow
    # guard, so a numeric window near INT32_MIN/MAX wraps when p = 1.
    p = (fx.Int32(varlen_bits) >> fx.Int32(VARLEN_PRINCIPAL_DIAGONAL_SHIFT)) & fx.Int32(1)
    shift = p * (seqlen_k - seqlen_q)
    left = common_utils.ssel(left_is_sentinel, seqlen_q, left - shift)
    # Every comparison below is against the caller's raw value, never a
    # shifted one, so a shifted bound can never be mistaken for a sentinel.
    right = common_utils.ssel(
        right_is_sentinel,
        common_utils.ssel(
            fx.Int32(window_right) == fx.Int32(WINDOW_BOTRIGHT),
            seqlen_k - seqlen_q,
            fx.Int32(0),
        ),
        right + shift,
    )
    return left, right


@dataclass(frozen=True, slots=True)
class CausalRegions:
    """The three contiguous KV block runs a causal/windowed Q block walks.

    Every field is a traced `fx.Int32`, not a Python int -- these are values
    the kernel computes per workgroup. Signed, because `right_col0` goes
    negative when the window admits no key at all; see
    `decompose_causal_regions`.

    A dataclass rather than a `NamedTuple` to match `Philox` next door, and
    because no caller destructures it positionally -- the kernel reads the
    seven fields by name. Being a Python object it is subject to the usual
    rule: do not let one live across a dynamic `if` (see "How to hand a helper
    object to kernel code"). The kernel unpacks it immediately, which is why
    it is safe here.
    """

    n_left: fx.Int32  # masked tiles before the full run
    n_full: fx.Int32  # tiles with no mask at all
    n_right: fx.Int32  # masked tiles after it
    left_col0: fx.Int32  # first KV column of each run
    full_col0: fx.Int32
    right_col0: fx.Int32
    masked_col0: fx.Int32  # first column of the masked run, whichever side


def decompose_causal_regions(start_q, q_len, k_len, window_left, window_right, block_m, block_n, alive):
    """Cut this Q block's visited KV range into `[masked][full][masked]`.

    **Three regions, not two.** A left window kills columns at the *start* of
    the range as well as the end, so masked tiles are a prefix as well as a
    suffix and tile 0 is not automatically live. A negative `window_left` is
    the sharpest case: it pushes the whole band right of the diagonal, so the
    leading masked run can span several tiles rather than clipping one. Do not
    carry the non-causal two-region intuition in here.

    The three are contiguous and non-overlapping *by construction*, because
    they are derived by cutting one visited range rather than intersected as
    three independent intervals. That collapses two of the three special cases
    in `sdpa-gswa-plan.md` section 2.2: a window narrower than a block leaves
    the full region empty, which is detected once and turns the other two into
    a single masked run, and an irregular `seqlen_q` needs no special handling
    because `q_hi` already bounds the rows.

    A column c is live for row i iff `i - window_left <= c <= i + window_right`,
    so over the block the live columns span
    `[start_q - window_left, (q_hi - 1) + window_right]`, and a tile is *fully*
    live iff every one of its columns is live for every row -- worst case the
    largest row on the left and the smallest on the right.

    `alive` is false for a workgroup whose rows all sit past `q_len`, which the
    varlen grid dispatches because its Q extent is sized from `Max_seqlen_q`.
    The kernel is one single-exit trace and cannot return out of those (plan
    section 6.1), so the visited range is *inverted* instead and every region
    count falls to zero. Dropping this makes those workgroups walk real tiles.

    Everything here is i32 and signed, deliberately: `left_col0` and friends
    go negative when the window admits no key at all. See `resolve_window`.
    """
    one = fx.Int32(1)
    zero = fx.Int32(0)
    bn = fx.Int32(block_n)

    q_start = fx.Int32(start_q)
    q_hi = common_utils.smin(q_start + fx.Int32(block_m), q_len)
    q_last = q_hi - one

    # Blocks that exist at all, and the last block that is *whole*. Splitting
    # these is section 2.2 case 3: a ragged seqlen_k leaves a partial final
    # tile, which must be masked rather than counted as full.
    blk_last = common_utils.sdiv_rd_pow2(k_len - one, block_n)
    blk_last_whole = common_utils.sdiv_rd_pow2(k_len, block_n) - one

    # The visited range: outside it every column is dead for every row in this
    # Q block, so those tiles are not walked at all.
    v_lo = common_utils.smax(common_utils.sdiv_rd_pow2(q_start - window_left, block_n), zero)
    v_hi = common_utils.smin(blk_last, common_utils.sdiv_rd_pow2(q_last + window_right, block_n))
    v_hi = common_utils.ssel(alive, v_hi, v_lo - one)

    # Rounded *up* on the left: a block is fully live only once its first
    # column clears the leftmost row's window. Rounding down would send a
    # partly-masked tile through the unmasked loop body -- invisible to a
    # tolerance test, not to the bitwise one.
    l_first_full = common_utils.sdiv_rd_pow2(q_last - window_left + fx.Int32(block_n - 1), block_n)
    r_first_mask = common_utils.sdiv_rd_pow2(q_start + window_right + one, block_n)

    fb_lo = common_utils.smax(l_first_full, v_lo)
    fb_hi = common_utils.smin(common_utils.smin(r_first_mask - one, blk_last_whole), v_hi)
    fb_empty = fb_lo > fb_hi

    # Cut [v_lo, v_hi] at the full region. With no full region the whole range
    # becomes one masked run -- section 2.2 case 2, the window narrower than a
    # block, falling out for free.
    lb_hi = common_utils.ssel(fb_empty, v_hi, fb_lo - one)
    rb_lo = common_utils.ssel(fb_empty, v_hi + one, fb_hi + one)

    n_left = common_utils.smax(lb_hi - v_lo + one, zero)
    n_full = common_utils.smax(fb_hi - fb_lo + one, zero)
    n_right = common_utils.smax(v_hi - rb_lo + one, zero)

    left_col0 = v_lo * bn
    right_col0 = rb_lo * bn
    full_col0 = fb_lo * bn
    # First tile of the masked run, which is also what the full loop's last
    # prefetch must fetch: the two loops are adjacent only when the left run is
    # empty. Clamped, because with a window admitting no key at all every run
    # is empty and `rb_lo` sits below zero -- and this value still reaches the
    # prologue's address computation.
    masked_col0 = common_utils.smax(common_utils.ssel(n_left > zero, left_col0, right_col0), zero)
    return CausalRegions(n_left, n_full, n_right, left_col0, full_col0, right_col0, masked_col0)


# --------------------------------------------------------------------------
# Host ABI
# --------------------------------------------------------------------------

# Resolving in one place removes the class of bug rather than the
# instance.
CAUSAL_SENTINEL = {
    1: WINDOW_TOPLEFT,  # j <= i
    2: WINDOW_BOTRIGHT,  # j <= i + (seqlen_k - seqlen_q)
}

# ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
#
# One byte per side, decoded by the same kernel-side function twice, plus
# the LSE layout in byte 2. `0` is BHSD / MAX / IMPLIED on both sides with
# an (H, T) logsumexp -- the conventional dense case, and the default.
VARLEN_STACKED = 1
VARLEN_LENGTH_MAX = 0 << 1  # noqa: F841  (the table is the wire spec)
VARLEN_LENGTH_CUMULATIVE = 1 << 1
VARLEN_LENGTH_INDIVIDUAL = 2 << 1
VARLEN_POSITION_IMPLIED = 0 << 3
VARLEN_POSITION_REUSE = 1 << 3
VARLEN_POSITION_ARRAY = 2 << 3
# `_HT` is AOTriton's and this kernel's default: shape (H, T), T
# contiguous. `_TH` is Transformer Engine's (T, H).
VARLEN_LSE_LAYOUT_HT = 0 << 16
VARLEN_LSE_LAYOUT_TH = 1 << 16  # noqa: F841  (ditto)
# Bit 18, principal_diagonal_choice: which diagonal a NUMERIC window is
# measured from. Sentinel windows ignore it. See `resolve_window`.
VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT = 0 << VARLEN_PRINCIPAL_DIAGONAL_SHIFT
VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT = 1 << VARLEN_PRINCIPAL_DIAGONAL_SHIFT


def varlen_bits(q_side=0,
                k_side=0,
                lse_layout=VARLEN_LSE_LAYOUT_HT,
                principal_diagonal=VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT):
    """Assemble VarlenBits from per-side bytes, the LSE layout and the principal diagonal."""
    if principal_diagonal not in (VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT, VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT):
        raise ValueError(f"principal_diagonal must be VARLEN_PRINCIPAL_DIAGONAL_*, got {principal_diagonal:#x}")
    for name, b in (("q_side", q_side), ("k_side", k_side)):
        if not 0 <= b <= 0xFF:
            raise ValueError(f"{name} must fit in a byte, got {b:#x}")
        if (b >> 3) & 3 == 1 and (b >> 1) & 3 != 1:
            # REUSE takes a *position* out of the length array, which is
            # only a position when the lengths are cumulative.
            raise ValueError(f"{name}={b:#04x}: POSITION=REUSE requires " f"LENGTH=CUMULATIVE (plan section 1, axis C)")
        if (b >> 1) & 3 == 3 or (b >> 3) & 3 == 3:
            raise ValueError(f"{name}={b:#04x} uses a reserved code")
    return q_side | (k_side << 8) | lse_layout | principal_diagonal


VARLEN_DENSE = 0  # noqa: F841  (ditto)
# Mask of the two side bytes. A word whose side bytes are both `VARLEN_DENSE`
# describes a dense call; it still carries bits 18:16.
VARLEN_SIDES_MASK = 0xFFFF
VARLEN_COMPACT_SIDE = VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_REUSE  # 0x0B
VARLEN_PADDED_SIDE = VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_IMPLIED
# 0x02

VARLEN_STRIDED_SIDE = VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_ARRAY  # 0x13
VARLEN_SEQUSED_PACKED_SIDE = VARLEN_STACKED | VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_ARRAY  # 0x15
VARLEN_SEQUSED_CACHE_SIDE = VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_IMPLIED  # 0x04


def varlen_dense(principal_diagonal=VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT,
                 lse_layout=VARLEN_LSE_LAYOUT_HT):
    """A dense call's VarlenBits word: dense Q and K sides, nothing else read.

    The C++ ABI sends a VarlenBits word on dense calls too, and this is that
    word. It exists for the fields that are not layout facts and so apply to
    dense calls as well: `principal_diagonal_choice` (bit 18) and the LSE
    layout (bits 17:16). The word is the only place either is carried;
    `varlen=None` is the same as `varlen_dense()`, i.e. word 0.

    It has no `seqinfo_*` tensors and no `max_seqlen_*`: the max lengths are
    the call's `seqlen_q` / `seqlen_k` (see `varlen_args`).
    """
    return dict(bits=varlen_bits(VARLEN_DENSE,
                                 VARLEN_DENSE,
                                 lse_layout=lse_layout,
                                 principal_diagonal=principal_diagonal))


def varlen_is_dense(varlen):
    """True for `varlen=None` and for a dense word (`varlen_dense`).

    Both describe a dense call: a real batch axis, uniform `seqlen_q` /
    `seqlen_k`, no sequence-info tensors.
    """
    return varlen is None or int(varlen["bits"]) & VARLEN_SIDES_MASK == 0


def dense_row_shape(varlen, batch_size, num_head_q, seqlen_q):
    """The (rows, cols) a dense call's LSE/Delta must have, per the LSE layout.

    HT is `(batch * num_head_q, seqlen_q)`, TH is `(batch * seqlen_q,
    num_head_q)`. Only meaningful when `varlen_is_dense(varlen)`.
    """
    layout = 0 if varlen is None else int(varlen["bits"]) & (3 << 16)
    if layout == VARLEN_LSE_LAYOUT_TH:
        return (int(batch_size) * int(seqlen_q), int(num_head_q))
    return (int(batch_size) * int(num_head_q), int(seqlen_q))


def varlen_compact(
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT
):
    """Classical packed varlen: 1THD tensors, `cu_seqlens` for both roles.

    `seqinfo_?1` is deliberately **not** passed: `POSITION = REUSE` takes
    the position out of the cumulative length value already loaded, so
    this configuration reads no position array at all (plan section 1.2).
    """
    return dict(
        bits=varlen_bits(VARLEN_COMPACT_SIDE, VARLEN_COMPACT_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_padded(
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT
):
    """BHSD tensors whose sequences are short: lengths only, no positions."""
    return dict(
        bits=varlen_bits(VARLEN_PADDED_SIDE, VARLEN_PADDED_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_strided(
    cu_seqlens_q,
    cu_seqlens_k,
    seq_strides_q,
    seq_strides_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Packed tensors with padding *between* sequences (TE's layout).

    Differs from `varlen_compact` in one thing only: positions come from a
    second array instead of being reused from the length array. That is
    the whole of AOTriton's `StridedVarlen`, and the reason the two roles
    must never be swapped -- `seq_strides` differences are *padded*
    extents, not lengths.
    """
    return dict(
        bits=varlen_bits(VARLEN_STRIDED_SIDE, VARLEN_STRIDED_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=seq_strides_q,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=seq_strides_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_seqused_k(
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    k_is_cache=False,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Packed Q against a KV cache with per-sequence *used* lengths.

    `torch.nn.attention.varlen`'s `seqused_k`, and the configuration no
    `VarlenType` can express: the K side takes its **length** from an
    individual array and its **position** from a cumulative one, so the
    two axes read different tensors.

    `k_is_cache=True` is the rectangular variant -- a BHSD cache with no
    `cu_seqlens_k` at all, where the position is implied by the batch
    index.
    """
    k_side = VARLEN_SEQUSED_CACHE_SIDE if k_is_cache else VARLEN_SEQUSED_PACKED_SIDE
    return dict(
        bits=varlen_bits(VARLEN_COMPACT_SIDE, k_side, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=seqused_k,
        seqinfo_k1=None if k_is_cache else cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


NULL_PTR = flyc.from_c_void_p(fx.Uint8, 0)


def ptr_arg(t):
    if hasattr(t, "data_ptr"):
        type_name = type(t).__name__
        module_name = type(t).__module__
        ptr = 0 if type_name == "FakeTensor" or "fake_tensor" in module_name else t.data_ptr()
        return flyc.from_c_void_p(fx.Uint8, ptr)
    return t


def strides_of(t, name):
    """`(batch, head, seq)` strides of a rank-4 BHSD tensor, in elements.

    **Shape and layout are different things**, and this reads the second.
    *Shape* is `t.shape`, and the kernel requires it to be BHSD --
    `(batch, num_heads, seq_len, head_dim)` -- because that is what fixes
    which axis each of the three returned slots describes. *Layout* is how
    the data actually sits in memory, and any permutation is accepted so
    long as D is innermost, which is what `stride(3) == 1` checks. A BSHD
    *layout* is therefore fine; pass it as `t.transpose(1, 2)`, which has
    BHSD shape.

    The order is the ABI. AOTriton dispatches the compiled hsaco directly
    rather than through the Python wrapper, so these three slots are the
    contract, and a caller that fills them from a BSHD shape swaps head
    with sequence -- reading heads where the kernel means tokens, which
    produces finite garbage and never faults.
    """
    if not hasattr(t, "stride"):
        raise TypeError(f"{name} must be a rank-4 tensor so its strides can be read, " f"got {type(t).__name__}")
    if t.dim() != 4:
        raise ValueError(f"{name} must be rank 4, got shape {tuple(t.shape)}")
    if t.stride(3) != 1:
        raise ValueError(f"{name} must have a contiguous last dimension, got " f"stride(3)={t.stride(3)}")
    # **The D-axis pitch must cover a whole 8-element chunk.**
    #
    # Every 16-bit access on this axis is 8 columns wide and the guard is
    # `col < head_dim`, so the chunk containing the last live column runs to
    # `ceil8(head_dim)`. That is a real address bound and not a "discard the
    # surplus", but only while the allocation reaches that far: at a tighter
    # pitch the chunk lands in the next row, and on the final row it leaves the
    # tensor. The store side is the sharp one -- dQ, dK, dV and O *write* those
    # columns, so a tight pitch corrupts the next row rather than merely
    # reading it. Measured at head_dim 20 with a packed dQ: 4 elements past the
    # allocation and 41% error in the last row.
    #
    # Checked here rather than in each interface because there were three
    # private copies of this rule and each had a different hole -- dQ's omitted
    # `dq` itself, and none of them ran on the builder path that every test and
    # both `tooling/` probes use. `strides_of` is on every path by construction.
    #
    # Stated against `shape[3]` rather than the compiled tile: the tile is not
    # visible here, and it is not the bound anyway. Vacuous whenever head_dim
    # is itself a multiple of 8, which every ladder rung is.
    _pitch_needed = (t.shape[3] + 7) // 8 * 8
    if t.stride(2) < _pitch_needed:
        raise ValueError(
            f"{name} has a D-axis pitch of {t.stride(2)} elements, under the "
            f"{_pitch_needed} that head_dim {t.shape[3]} needs. Accesses on this axis are 8 "
            f"columns wide, so the chunk holding the last column runs to ceil8({t.shape[3]})"
            f"={_pitch_needed} and would otherwise read -- and for an output, write -- past "
            f"the row. Allocate the last dimension padded to {_pitch_needed} and slice."
        )
    return t.stride(0), t.stride(1), t.stride(2)


def prep_tensors(named, *, q_heads=(), k_heads=()):
    """`(pointers, (nhq, nhk, hqk, hvo), strides)` in launch order.

    `named` is `[("Q", Q), ("K", K), ...]` -- order is the launch order for
    both the pointers and the three-per-tensor strides, and the names are what
    the diagnostics quote. "Q", "K" and "V" must be present; they are what the
    head counts and head dims are read from.

    `q_heads` / `k_heads` name the *other* tensors that must carry Q's head
    count or K's. That is the only thing that varied between the four kernels'
    copies of this: the forward checks O against Q, dQ checks dO and dQ, dK/dV
    checks dO against Q and dK/dV against K. Everything else -- the stride
    sweep, the GQA divisibility rule, reading head_dim from Q and head_dim_v
    from V -- was identical three times over.

    Deliberately does **not** flatten: `t.reshape(-1)` materialises a copy for
    any non-contiguous tensor, which would silently defeat the whole point of
    reading strides. `data_ptr()` is the tensor's base either way.
    """
    by_name = dict(named)
    for req in ("Q", "K", "V"):
        if req not in by_name:
            raise ValueError(f"prep_tensors needs a {req!r} entry, got {[n for n, _ in named]}")
    st = []
    for name, t in named:
        st.extend(strides_of(t, name))
    q, k, v = by_name["Q"], by_name["K"], by_name["V"]
    # BHSD: axis 1 is heads, axis 3 the head dim. Read rather than assumed --
    # under MQA/GQA K and V carry fewer heads than Q.
    nhq, nhk = q.shape[1], k.shape[1]
    hqk, hvo = q.shape[3], v.shape[3]
    if v.shape[1] != nhk:
        raise ValueError(f"K and V must share num_heads, got {nhk} and {v.shape[1]}")
    for name in q_heads:
        if by_name[name].shape[1] != nhq:
            raise ValueError(f"{name} must carry num_heads_q ({nhq}), got {by_name[name].shape[1]}")
    for name in k_heads:
        if by_name[name].shape[1] != nhk:
            raise ValueError(f"{name} must carry num_heads_k ({nhk}), got {by_name[name].shape[1]}")
    if nhq % nhk:
        raise ValueError(f"num_heads_q ({nhq}) must be divisible by num_heads_k ({nhk})")
    return [ptr_arg(t) for _, t in named], (nhq, nhk, hqk, hvo), st


def resolve_scale(q, scale, padded_head, default_scale):
    """`sm_scale` from the tensor's *real* head dim, not the compiled tile.

    A builder can only default to `1/sqrt(BLOCK_DMODEL)`, since it has no idea
    what `hdim_qk` will be -- and under a padded head that is the wrong number.
    Deriving it from `q.shape[3]` is right in both cases, and identical to the
    builder's default whenever the two coincide.
    """
    if scale is not None:
        return float(scale)
    if padded_head and hasattr(q, "shape"):
        return 1.0 / math.sqrt(q.shape[3])
    return float(default_scale)


def row_tensor_arg(t, name, num_head_q, seq_len, varlen):
    """Check a rank-2 f32 row tensor -- logsumexp, or delta -- and take its pointer.

    The two share one offset computation in the kernel, so they must share a
    layout, and checking them the same way here is what makes that safe.
    Unlike Q/K/V the kernel derives their pitches from VarlenBits rather than
    reading strides, so contiguity is required rather than merely convenient --
    and the host is the only place the caller's actual layout can be verified.
    """
    from torch import float32 as torch_f32  # lazy: the build venv has no torch

    if t is None:
        raise ValueError(f"{name} is required")
    if t.dtype != torch_f32:
        raise ValueError(f"{name} must be float32, got {t.dtype}")
    if t.dim() != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous: the kernel derives its pitches from VarlenBits")
    layout = 0 if varlen is None else (int(varlen["bits"]) >> 16) & 3
    if layout == 0:
        want_last = int(seq_len) if varlen_is_dense(varlen) else varlen.get("lse_tokens")
        if want_last is not None and t.shape[1] != int(want_last):
            raise ValueError(f"{name} with LSE_LAYOUT_HT wants (*, {int(want_last)}), got {tuple(t.shape)}")
    elif t.shape[1] != num_head_q:
        raise ValueError(f"{name} with LSE_LAYOUT_TH wants (*, {num_head_q}), got {tuple(t.shape)}")
    return ptr_arg(t)


def lse_args(lse, seq_len, varlen, num_head_q):
    """The logsumexp pointer, and a check that its layout matches the bits.

    Returns only a pointer: unlike Q/K/V/O this tensor is always compact,
    so the kernel derives both pitches from `LSE_LAYOUT`, `num_head_q` and
    the token count (sdpa-varlen-plan.md section 4.2). Inferring strides
    here as well would give one fact two sources.

    What the host can do instead -- and could not while it was inferring --
    is verify the caller's tensor actually has the declared layout.
    """
    from torch import float32 as torch_f32  # lazy: the build venv has no torch

    if lse is None:
        return NULL_PTR
    if lse.dtype != torch_f32:
        raise ValueError(f"logsumexp must be float32, got {lse.dtype}")
    if lse.dim() != 2:
        raise ValueError(f"logsumexp must be rank 2, got shape {tuple(lse.shape)}")
    if not lse.is_contiguous():
        raise ValueError(
            "logsumexp must be contiguous: the kernel derives its pitches "
            "from VarlenBits rather than reading strides"
        )
    _layout = 0 if varlen is None else (int(varlen["bits"]) >> 16) & 3
    if _layout == 0:
        # The token pitch the kernel will derive. Only checkable when the
        # caller supplies it: under a stacked layout it lives in
        # `seqinfo[N]` on the device, and reading that back would cost a
        # sync for a validation.
        want_last = int(seq_len) if varlen_is_dense(varlen) else varlen.get("lse_tokens")
        if want_last is not None and lse.shape[1] != int(want_last):
            raise ValueError(f"VARLEN_LSE_LAYOUT_HT wants (*, {int(want_last)}), got " f"{tuple(lse.shape)}")
    else:
        if lse.shape[1] != num_head_q:
            raise ValueError(f"VARLEN_LSE_LAYOUT_TH wants (*, {num_head_q}), got " f"{tuple(lse.shape)}")
    return ptr_arg(lse)


def window_args(causal_type, host_causal_type, window, seqlen_q, seqlen_k):
    """(window_left, window_right), signed, as the kernel wants them.

    Non-causal still forwards a pair so both arms share one ABI and stay
    directly comparable -- the same reason the strides are always passed
    even under strides_constexpr.

    Causal alignments forward a *sentinel*, not a bound: the kernel
    resolves it per sequence, which is the only correct thing to do once
    there is more than one sequence.
    """
    if causal_type == 0:
        if window is not None:
            # Silently dropping it would return dense attention that is
            # the right shape, finite, and wrong -- and a window is only
            # ever passed by a caller who believes it is being applied.
            # The non-causal arm has no left-masked region to apply one
            # with, so this is a build-time choice, not a runtime one.
            raise ValueError(
                "window= requires a causal build; this one has "
                "causal=False. Pass causal=True, causal_type=3 for "
                "generalized sliding-window attention"
            )
        return 0, 0
    if host_causal_type in CAUSAL_SENTINEL:
        if window is not None:
            raise ValueError(
                f"causal_type={host_causal_type} already fixes the window; " "pass causal_type=3 to choose one"
            )
        _s = CAUSAL_SENTINEL[host_causal_type]
        wl, wr = _s, _s
    else:
        if window is None:
            raise ValueError(
                "causal_type=3 is generalized sliding-window attention and "
                "requires window=(left, right); "
                f"pass ({seqlen_q}, 0) for top-left causal or "
                f"({seqlen_q}, {seqlen_k - seqlen_q}) for bottom-right"
            )
        wl, wr = window
    return int(wl), int(wr)


def u64_scalar(value, device, stream=None):
    """A one-element device u64 holding `value`, or `value` if already one.

    `None` stays `None` (the null case). A tensor is taken as-is and *not*
    copied, which is the point: under graph capture the caller owns a counter
    the graph re-reads, and copying it here would freeze it again.

    An int is materialised, which is what AOTriton's host does with its own
    `DEFAULT_PHILOX_SEED`. `torch.int64` and not a uint64: torch has no
    unsigned 64-bit dtype, and the kernel reads the same eight bytes either
    way. Allocated on `stream` when one is given, because the kernel that
    reads it runs there.

    **The returned tensor must stay referenced until the launch is enqueued.**
    Only its raw pointer reaches the kernel, so nothing else keeps it alive --
    callers bind it to a local that outlives the launch call.
    """
    import torch  # lazy: only reached for a plain int seed; AOT passes None

    if value is None or hasattr(value, "data_ptr"):
        if value is not None and value.numel() < 1:
            raise ValueError("a philox scalar tensor must hold at least one element")
        if value is not None and value.element_size() != 8:
            raise ValueError(f"a philox scalar tensor must be 8 bytes per element, got {value.dtype}")
        return value
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        return torch.tensor([int(value)], dtype=torch.int64, device=device)


def dropout_outputs(enable_dropout, seed_output, offset_output):
    """(seed_output, offset_output) in launch order -- the forward's write-back.

    The forward records the `(seed, offset)` it actually drew from so the
    backward can be handed them rather than re-deriving them. That matters
    only under graph capture, where the effective offset is a sum formed on
    the device out of a counter the host cannot read without synchronising.
    Either may be `None`, meaning the caller does not want the value.
    """
    if not enable_dropout:
        return NULL_PTR, NULL_PTR
    return (
        NULL_PTR if seed_output is None else ptr_arg(seed_output),
        NULL_PTR if offset_output is None else ptr_arg(offset_output),
    )


def bias_args(bias_type, emit_db, bias, dbias, like):
    """`(B, DB, b_strides, db_strides)`, each stride set a `(batch, head, seq_q)`
    triple in launch order.

    **Two triples, not one.** B and DB are separate tensors and may be laid out
    differently -- `dbias` is the caller's buffer, and nothing obliges it to
    match the bias it is the gradient of. Deriving one from the other happens
    to work whenever both are contiguous, which is why it survived a first
    pass, and writes to the wrong addresses the moment either is a view.

    The bias is `(B, H, Sq, Sk)`, so its three leading strides are batch, head
    and *query row* -- the last axis is the KV column and is the contiguous one
    the kernel's inner loop walks. That is the forward's layout, unchanged, so
    a caller can hand the backward the very tensor it gave the forward.

    `emit_db` is whether the *kernel* has a DB slot at all -- true for dQ,
    false for dK/dV and the fused kernel, which take a bias but do not write
    its gradient. It is a fact about the signature, not a knob: a dQ build with
    a bias always emits dB, since dB is the `dS` it already forms.

    Both pointers are always in the kernarg signature and are `NULL_PTR` when
    the bias axis is off, which is how the dropout arguments behave: a kernel
    argument that exists unconditionally is one fewer ABI variant to keep in
    step, and `const_expr` means the null is never dereferenced.

    `dbias` is checked against `bias` for shape rather than being inferred,
    because a dB that is silently the wrong shape writes out of bounds.
    """
    if not bias_type:
        if bias is not None:
            raise ValueError("bias= was passed but this build has bias=False; rebuild with bias=True")
        return NULL_PTR, NULL_PTR, (0, 0, 0), (0, 0, 0)
    if bias is None:
        raise ValueError("this build has bias=True and requires bias=")
    if bias.dim() != 4:
        raise ValueError(f"bias must be (B, H, Sq, Sk), got {tuple(bias.shape)}")
    if bias.dtype != like.dtype:
        raise ValueError(f"bias dtype {bias.dtype} must match q's {like.dtype}")
    if bias.stride(3) != 1:
        # The kernel loads eight adjacent KV columns per accumulator group in
        # one v8, exactly as the forward does; the same check lives there.
        raise ValueError(f"bias must have a contiguous last (Sk) dimension, got stride(3)={bias.stride(3)}")
    db = NULL_PTR
    if emit_db:
        if dbias is None:
            raise ValueError("a dQ build with bias=True always writes dB and requires dbias=")
        if tuple(dbias.shape) != tuple(bias.shape):
            raise ValueError(f"dbias shape {tuple(dbias.shape)} must equal bias's {tuple(bias.shape)}")
        if dbias.dtype != bias.dtype:
            raise ValueError(f"dbias dtype {dbias.dtype} must equal bias's {bias.dtype}")
        if dbias.stride(3) != 1:
            raise ValueError(f"dbias must have a contiguous last (Sk) dimension, got stride(3)={dbias.stride(3)}")
        db = ptr_arg(dbias)
    elif dbias is not None:
        raise ValueError("dbias= was passed to a kernel that does not write dB; that is the dQ kernel's")
    sb = bias.stride()[:3]
    sdb = dbias.stride()[:3] if emit_db else (0, 0, 0)
    return ptr_arg(bias), db, sb, sdb


def dropout_args(enable_dropout, dropout_p, seed, offset1, offset2, device=None, stream=None):
    """(seed, offset1, offset2, threshold, 1/(1-p), keepalive) in launch order.

    The trailing `keepalive` is not a kernel argument. It is whatever tensor
    `u64_scalar` had to materialise for an int seed, returned so the caller
    can hold it across the launch; see that function.

    The counter is the pair torch splits it into, not one pre-summed scalar:
    `offset1` is a device pointer the kernel adds in when non-null, `offset2`
    an immediate. See `philox_offset_base` for why a captured CUDA graph
    needs the pointer half, and `philox_offset1`/`philox_offset2` in AOTriton
    for the ABI this matches.

    `offset1` may be a tensor holding one u64, or None for the uncaptured
    case. A one-element `torch.int64` tensor is the same 8 bytes as the `u64`
    the kernel reads; torch has no unsigned 64-bit dtype to spell it with, and
    the counter is far from the sign bit.

    The threshold and the scale are computed here, once per call, rather
    than per element in the kernel -- `philox.dropout_threshold` turns the
    probability into an i32 the raw random can be compared against, so the
    hot path never converts a random to a float.
    """
    if not enable_dropout:
        return NULL_PTR, NULL_PTR, 0, 0, 1.0, None
    if dropout_p is None:
        raise ValueError("this build has dropout=True and requires dropout_p=")
    p = float(dropout_p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {p}")
    seed_t = u64_scalar(seed, device, stream)
    off1_t = u64_scalar(offset1, device, stream)
    return (
        NULL_PTR if seed_t is None else ptr_arg(seed_t),
        NULL_PTR if off1_t is None else ptr_arg(off1_t),
        int(offset2),
        dropout_threshold(p),
        1.0 / (1.0 - p),
        (seed_t, off1_t),
    )


def varlen_args(strides_constexpr, varlen, seqlen_q, seqlen_k, q, batch_size, num_seqlens):
    """(bits, q0, q1, k0, k1, max_q, max_k) in launch order, plus two checks.

    `varlen` is None for the dense case (word 0), else a dict with `bits`
    and whichever `seqinfo_*` tensors that configuration reads. A dict whose
    side bytes are both dense (`varlen_dense`) is a dense call that carries
    bits 18:16: it takes no `seqinfo_*` and no `max_seqlen_*`, and its max
    lengths are `seqlen_q` / `seqlen_k`. Unread slots stay **null**, which
    is safe because the kernel's decode branches rather than selects -- see
    the prologue.

    **`batch_size` and `num_seqlens` are different quantities and never share
    a variable on the host.** `batch_size` is `q.size(0)`, always, whatever the
    layout. `num_seqlens` is how many sequences are packed into a 1HTD tensor,
    and is 0 when nothing is packed. For a dense BHSD call they are `(B, 0)`;
    for a packed `(1, H, T, D)` call holding N sequences they are `(1, N)` --
    genuinely different numbers, which is why one variable cannot serve.

    Neither is returned: both are already the caller's, and this function's job
    for them is to *check*, because each has a second, independent source of
    truth. `batch_size` must be `q.size(0)`, and a packed `num_seqlens` must be
    `len(cu_seqlens_q) - 1`. Passing N where `batch_size` belongs is the
    specific mistake -- it launches N programs over a tensor whose batch axis
    is 1, and every one of them addresses a plausible row.
    """
    if int(batch_size) != int(q.shape[0]):
        raise ValueError(
            f"batch_size={int(batch_size)} but q.size(0)={int(q.shape[0])}. batch_size is the "
            f"tensor's batch extent whatever the layout; a packed 1HTD tensor has 1, and its "
            f"sequence count goes in num_seqlens."
        )
    if varlen_is_dense(varlen):
        # `varlen=None`, or a dense word (`varlen_dense`) carrying only bits
        # 18:16. Handled before the `strides_constexpr` check, which is about
        # real varlen: a dense word changes no layout.
        if int(num_seqlens):
            raise ValueError(f"num_seqlens={int(num_seqlens)} on a dense call; a dense call packs no sequences")
        if varlen is None:
            return (0, NULL_PTR, NULL_PTR, NULL_PTR, NULL_PTR, int(seqlen_q), int(seqlen_k))
        given = [k for k in ("seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1") if varlen.get(k) is not None]
        if given:
            raise ValueError(
                f"a dense VarlenBits word (both side bytes 0) reads no sequence info, but {given} "
                f"were given; set the side bytes, or drop the tensors"
            )
        return (int(varlen["bits"]), NULL_PTR, NULL_PTR, NULL_PTR, NULL_PTR, int(seqlen_q), int(seqlen_k))
    if strides_constexpr:
        raise ValueError(
            "strides_constexpr derives the layout from the shape, which "
            "varlen invalidates; it is a dense-only diagnostic arm"
        )
    # No implemented-subset gate: every encodable side byte now decodes,
    # since the decoder is one function covering all three axis values.
    # `varlen_bits` rejects the combinations that are not *meaningful*
    # (reserved codes, REUSE without cumulative lengths).
    bits = int(varlen["bits"])
    # A STACKED Q side is the packed one, and the only one the kernel reads
    # `num_seqlens` for (`lse_token_pitch`, to reach slot [N] of the array
    # holding the batch total). A non-stacked varlen side -- `varlen_padded`,
    # BHSD tensors with short sequences -- has a real batch axis and packs
    # nothing, so its count is 0 like the dense case.
    if bits & VARLEN_STACKED:
        if varlen.get("seqinfo_q0") is None:
            raise ValueError("a STACKED Q side needs seqinfo_q0 (cu_seqlens_q) to count its sequences")
        packed = int(varlen["seqinfo_q0"].numel()) - 1
        if int(num_seqlens) != packed:
            raise ValueError(f"num_seqlens={int(num_seqlens)} but cu_seqlens_q describes {packed} packed sequences")
    elif int(num_seqlens):
        raise ValueError(
            f"num_seqlens={int(num_seqlens)} but the Q side is not STACKED, so nothing is packed "
            f"and the batch axis is real; this configuration wants num_seqlens=0"
        )
    got = tuple(
        ptr_arg(varlen[k]) if varlen.get(k) is not None else NULL_PTR
        for k in ("seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1")
    )
    return (bits,) + got + (int(varlen["max_seqlen_q"]), int(varlen["max_seqlen_k"]))


# --------------------------------------------------------------------------
# Build options
# --------------------------------------------------------------------------

# Compile hints every attention builder sets on the `@flyc.jit` it returns.
# On the JitFunction, not around the call: `aotriton.flyc_compile` calls the
# JitFunction directly, so hints applied by a JIT-only wrapper never reach an
# AOT build. FlyDSL 0.4.2 reads lower-case keys only.
#
#   fast_fp_math    the ambient fastmath, `fast`, for traced ops that carry no
#                   flags of their own; also the ROCDL target's `fast`.
#   unsafe_fp_math  the ROCDL target's `unsafe-math`. That and `fast` only set
#                   ocml's `__oclc_*` controls, so both are inert while no
#                   kernel calls ocml; kept so that one that does agrees.
#   llvm_options    LLVM cl::opts for the whole compile.
COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}

# f32 denormals flushed (DAZ); f16/f64 stay IEEE.
_F32_DAZ = (
    "#llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee,"
    " float_output_mode = preservesign, float_input_mode = preservesign>"
)


def kernel_attrs(block_size, waves_per_eu=None, sched_strategy=""):
    """`value_attrs` for an attention kernel's `gpu.func`; call while tracing.

    Every kernel flushes f32 denormals. The LLVM function attributes take the
    `llvm.` prefix: the gpu.func lowering keeps `llvm.passthrough` and
    `llvm.denormal_fpenv`, and silently drops `passthrough` and any other bare
    `llvm.func` attribute name. `denormal_fpenv` because LLVM no longer reads
    the `denormal-fp-math-f32` string attribute of an in-memory module.
    """
    attrs = {
        "rocdl.flat_work_group_size": f"{block_size},{block_size}",
        "llvm.denormal_fpenv": ir.Attribute.parse(_F32_DAZ),
    }
    if waves_per_eu:
        attrs["rocdl.waves_per_eu"] = int(waves_per_eu)
    if sched_strategy:
        attrs["llvm.passthrough"] = [["amdgpu-sched-strategy", sched_strategy]]
    return attrs


def run_compiled(cache, exe, *args):
    """First call compiles and runs; later calls fast-dispatch the compiled fn.

    `cache` is the caller's `weakref.WeakKeyDictionary`, not a module-level one:
    four kernels share this function and a single dict keyed by launcher would
    work, but passing it keeps the lifetime obviously the caller's.

    `flyc.compile` caches internally (`JitFunction._mem_cache`, keyed on the
    full argument signature), so this is not about avoiding recompilation --
    it is about dispatch overhead, measured at **157 us per call** at
    (head_dim 64, N 512), where the whole call is 63 us with the cache and
    220 us without. The gap closes as the kernel grows.

    A `WeakKeyDictionary` rather than an `exe._cf` attribute: that attribute is
    not one FlyDSL defines, so writing it mutates a FlyDSL-owned object. Weak
    keys so an entry dies with its launcher instead of pinning one compiled
    artifact -- and the GPU code object it owns -- per configuration for the
    life of the process.
    """
    cf = cache.get(exe)
    if cf is None:
        cache[exe] = flyc.compile(exe, *args)
    else:
        cf(*args)


def new_compiled_cache():
    """A cache for `run_compiled`. One per kernel module."""
    return weakref.WeakKeyDictionary()


def dtype_to_elem_type(dtype_str):
    """`"f16"` / `"bf16"` -> the FlyDSL Numeric class the kernel builds with."""
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
