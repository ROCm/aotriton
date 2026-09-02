#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""VarlenBits: how a workgroup finds its sequence.

Variable-length attention is not one feature but a product of three
independent choices, taken per side (Q and K):

    A STACKED   bit 0     0 = BHSD (rank 4)      1 = THD (packed tokens,
                                                     batch size 1)
    B LENGTH    bits 2:1  0 = MAX  1 = CUMULATIVE  2 = INDIVIDUAL
    C POSITION  bits 4:3  0 = IMPLIED  1 = REUSE   2 = ARRAY

    bits  7:0   Q side
    bits 15:8   K side
    bits 17:16  LSE_LAYOUT  0 = _HT (AOTriton's, and the default)  1 = _TH
    bits 31:18  reserved (bits 31:24 for paged KV)

`Varlen_bits == 0` is BHSD / MAX / IMPLIED on both sides -- the conventional
dense case.

The two arrays each side may carry are named by *role*, never by which mode
passes them:

    seqinfo_?0   length source, read at [z] and [z+1] when LENGTH != MAX
    seqinfo_?1   position source, read at [z] when POSITION == ARRAY

POSITION == REUSE takes the position out of the *length* array, which is sound
only because CUMULATIVE makes that array hold positions too -- and it reuses
the value the length decode already loaded rather than issuing a second
access.  That is the whole reason REUSE exists as a distinct state: classical
(compact) varlen reads no position array at all.

See `../../../../FlyDSL/kernels/attention/parity/sdpa-varlen-plan.md` sections
2, 3.1 and 3.2 for the derivation.
"""

import triton
import triton.language as tl

# Layout constants.  Plain Python ints so host code can import them too.
VARLEN_STACKED = 0x01
VARLEN_LENGTH_SHIFT = 1  # 0 MAX, 1 CUMULATIVE, 2 INDIVIDUAL
VARLEN_POSITION_SHIFT = 3  # 0 IMPLIED, 1 REUSE, 2 ARRAY
VARLEN_K_SIDE_SHIFT = 8
VARLEN_LSE_LAYOUT_SHIFT = 16  # 0 _HT (AOTriton/default), 1 _TH (Transformer Engine)

# Only the addressing bytes decide whether the persistent loop is usable.
# LSE_LAYOUT lives above them and must not drag a dense call onto the varlen
# fallback -- see the execution plan section 3.2.  The host copy of
# `unsupported_by_persistent` must spell this identically.
VARLEN_ADDRESSING_MASK = 0xFFFF


class VarlenLength:
    MAX = 0
    CUMULATIVE = 1
    INDIVIDUAL = 2


class VarlenPosition:
    IMPLIED = 0
    REUSE = 1
    ARRAY = 2


class VarlenLseLayout:
    HT = 0  # (H, T), T contiguous.  AOTriton's, and today's behaviour.
    TH = 1  # (T, H), H contiguous.  Transformer Engine's.


def make_varlen_side(stacked, length, position):
    """One side's byte from its three axis values.

    `REUSE` is only meaningful under `CUMULATIVE`: it takes a position out of
    the length array, which holds positions only when that array is a prefix
    sum.  Rejected here rather than left to produce a plausible wrong address.
    """
    assert length in (VarlenLength.MAX, VarlenLength.CUMULATIVE, VarlenLength.INDIVIDUAL)
    assert position in (VarlenPosition.IMPLIED, VarlenPosition.REUSE, VarlenPosition.ARRAY)
    assert not (position == VarlenPosition.REUSE and length != VarlenLength.CUMULATIVE), \
        'POSITION == REUSE requires LENGTH == CUMULATIVE'
    return ((VARLEN_STACKED if stacked else 0)
            | (length << VARLEN_LENGTH_SHIFT)
            | (position << VARLEN_POSITION_SHIFT))


def make_varlen_bits(q_side, k_side, lse_layout=VarlenLseLayout.HT):
    """Assemble the wire word from two per-side bytes and the LSE layout.

    Preconditions the kernel assumes and does not check (checking them means
    reading the arrays back to the host, i.e. a device sync on the hot path):

    - a position array -- whichever of `seqinfo_?0` / `seqinfo_?1` supplies the
      position -- is a prefix sum whose total sits in slot `[N]`.  The LSE
      token pitch reads that slot; an arbitrary scatter of starts would need
      the total supplied separately.
    - sequences do not overlap.  A start plus a length that runs past the next
      sequence's start is representable, and the kernel would happily read the
      neighbour's tokens.
    """
    assert 0 <= q_side <= 0xFF and 0 <= k_side <= 0xFF
    assert lse_layout in (VarlenLseLayout.HT, VarlenLseLayout.TH)
    return (q_side
            | (k_side << VARLEN_K_SIDE_SHIFT)
            | (lse_layout << VARLEN_LSE_LAYOUT_SHIFT))


# Named shorthands, so callers spell the mode rather than the hex.
VARLEN_SIDE_DENSE = 0x00     # BATCHED, MAX, IMPLIED
VARLEN_SIDE_COMPACT = 0x0B   # STACKED, CUMULATIVE, REUSE
VARLEN_SIDE_PADDED = 0x02    # BATCHED, CUMULATIVE, IMPLIED
VARLEN_SIDE_STRIDED = 0x13   # STACKED, CUMULATIVE, ARRAY
VARLEN_SIDE_SEQUSED_PACKED = 0x15  # STACKED, INDIVIDUAL, ARRAY
VARLEN_SIDE_SEQUSED_BHSD = 0x04    # BATCHED, INDIVIDUAL, IMPLIED

VARLEN_BITS_DENSE = 0x0000
VARLEN_BITS_COMPACT = 0x0B0B
VARLEN_BITS_PADDED = 0x0202
VARLEN_BITS_STRIDED = 0x1313

assert VARLEN_BITS_DENSE == make_varlen_bits(VARLEN_SIDE_DENSE, VARLEN_SIDE_DENSE)
assert VARLEN_BITS_COMPACT == make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_COMPACT)
assert VARLEN_BITS_PADDED == make_varlen_bits(VARLEN_SIDE_PADDED, VARLEN_SIDE_PADDED)
assert VARLEN_BITS_STRIDED == make_varlen_bits(VARLEN_SIDE_STRIDED, VARLEN_SIDE_STRIDED)
assert 0x150B == make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_SEQUSED_PACKED)
assert 0x040B == make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_SEQUSED_BHSD)
assert 0x000B == make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_DENSE)


@triton.jit
def decode_addressing(Varlen_bits, BITS_SHIFT: tl.constexpr,
                      Max_seqlen, s0, s1, z):
    """One side of VarlenBits -> (seqlen, row_off, batch_index).

    Called twice: BITS_SHIFT=0 for Q, 8 for K.

    Ordinary `if`/`elif` rather than selects: Triton lowers a Python `if` on a
    scalar condition to real control flow, so a mode that is off never issues
    the load its array would need -- which matters because `s0`/`s1` are null
    pointers exactly then.  The prologue this replaces already relied on that.

    Bit positions are spelled as literals because Triton refuses to read a
    plain module global from inside a traced function; the module docstring and
    the VARLEN_* constants above are the reference.
    """
    bits = (Varlen_bits >> BITS_SHIFT) & 0xFF
    stacked = (bits & 0x01) != 0  # VARLEN_STACKED
    lenmode = (bits >> 1) & 3  # VARLEN_LENGTH_SHIFT
    posmode = (bits >> 3) & 3  # VARLEN_POSITION_SHIFT

    seqlen = Max_seqlen
    cur = 0
    if lenmode != 0:
        cur = tl.load(s0 + z)
        if lenmode == 1:  # CUMULATIVE
            seqlen = tl.load(s0 + z + 1) - cur
        else:  # INDIVIDUAL
            seqlen = cur

    if posmode == 1:  # REUSE -- already in a register, do not load again
        row_off = cur
    elif posmode == 2:  # ARRAY
        row_off = tl.load(s1 + z)
    elif stacked:  # IMPLIED, uniform stacking
        row_off = z * Max_seqlen
    else:  # IMPLIED, batched
        row_off = 0

    batch_index = z
    if stacked:
        batch_index = 0
    return seqlen, row_off, batch_index


@triton.jit
def lse_token_pitch(Varlen_bits, Max_seqlen_q, s0, s1, N):
    """Row pitch of the LSE/Delta output, in tokens.  Q side only.

    A batched layout pads every row-group to `Max_seqlen_q`; a stacked one runs
    to the batch total, which lives in slot `[N]` of whichever array supplies
    positions.  Derived rather than passed: the logsumexp buffer is the one
    tensor here that is always compact, so its strides are a function of the
    bits and passing them would be a second source of truth for one fact.
    """
    bits = Varlen_bits & 0xFF
    stacked = (bits & 0x01) != 0  # VARLEN_STACKED
    posmode = (bits >> 3) & 3  # VARLEN_POSITION_SHIFT

    tokens = Max_seqlen_q
    if stacked:
        if posmode == 1:  # REUSE: the length array is the prefix sum
            tokens = tl.load(s0 + N)
        elif posmode == 2:  # ARRAY
            tokens = tl.load(s1 + N)
        else:  # IMPLIED: uniform stacking
            tokens = N * Max_seqlen_q
    return tokens


@triton.jit
def lse_row_addressing(Varlen_bits, batch_index, head, Num_head_q,
                       lse_tokens, row_off):
    """(base, pitch) for a row-wise fp32 side output: LSE and Delta.

    The element for query row `r` within this sequence is at
    `base + r * pitch`.  Factored that way because `base` is loop-invariant
    while `r` is not.

        _HT   (H, T), T contiguous -- AOTriton's, and the default.
              base = (batch * H + head) * tokens + row_off,   pitch = 1
        _TH   (T, H), H contiguous -- Transformer Engine's.
              base = (batch * tokens + row_off) * H + head,   pitch = H

    Delta shares LSE's layout by construction -- it is a row-wise fp32 side
    output produced beside it -- so both go through this one function,
    including the pitch.
    """
    lse_layout = (Varlen_bits >> 16) & 3  # VARLEN_LSE_LAYOUT_SHIFT
    if lse_layout == 0:
        base = (batch_index * Num_head_q + head) * tl.cast(lse_tokens, tl.int64) + row_off
        pitch = 1
    else:
        base = (batch_index * tl.cast(lse_tokens, tl.int64) + row_off) * Num_head_q + head
        pitch = Num_head_q
    return base, pitch
