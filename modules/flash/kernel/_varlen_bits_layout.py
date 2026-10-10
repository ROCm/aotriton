#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side tensor builders, one per VarlenBits axis triple.

Given `(side_bits, seqlens)` this produces a tensor laid out the way those bits
*claim* it is, together with the `seqinfo_?0` / `seqinfo_?1` arrays that
describe it.  `test_varlen_bits.py` then asks the kernel to read it back and
compares against N separate dense calls.

**A bug here looks exactly like a bug in the kernel** -- both present as a
wrong answer at a plausible address.  Two rules follow, and both are load
bearing:

1. **Built independently of the decoder.**  Nothing in this file imports
   `decode_addressing`.  A builder that computed row offsets with the same
   expression the kernel decodes could only confirm that two copies agree; it
   could not catch a wrong expression.
2. **Total over the valid space.**  Every one of the 14 valid per-side values
   gets a layout, including the ones no shipped mode uses (`BATCHED + ARRAY`,
   "unusual but well-defined").  A builder that silently declined a valid
   combination would turn an untested configuration into an invisible one.

Slots versus lengths
--------------------
`seqlens[z]` is how many tokens of sequence `z` participate.  `slots[z]` is how
many rows the layout reserves for it, which may be larger -- that is what
`seqused_k` and the strided layout are for.  Everything the kernel must not
read (padding between sequences, the unused tail of a cache slot, rows past
`seqlen` in a padded batch) is filled from `[1, 2)` while the payload is drawn
from `[0, 1)`.  Two properties, and both are needed:

- the ranges are disjoint, so a human debugging a failure can see at a glance
  that a wrong address landed in filler;
- the *magnitudes* are comparable, so reading filler changes the answer.  An
  earlier version used a large negative constant, which made every filler
  score underflow to zero in the softmax -- the results then came out
  numerically identical and the "must differ from the full slot" half of the
  `seqused_k` gate passed vacuously.
"""

import numpy as np
import torch

from varlen_bits import (
    VARLEN_STACKED,
    VARLEN_LENGTH_SHIFT,
    VARLEN_POSITION_SHIFT,
    VARLEN_K_SIDE_SHIFT,
    VARLEN_LSE_LAYOUT_SHIFT,
    VarlenLength,
    VarlenPosition,
    VarlenLseLayout,
)

FILLER_LO = 1.0  # payload is drawn from [0, 1); filler from [1, 2)


def split_side(side_bits):
    """(stacked, length, position) from one per-side byte."""
    assert 0 <= side_bits <= 0xFF
    stacked = bool(side_bits & VARLEN_STACKED)
    length = (side_bits >> VARLEN_LENGTH_SHIFT) & 3
    position = (side_bits >> VARLEN_POSITION_SHIFT) & 3
    assert length in (VarlenLength.MAX, VarlenLength.CUMULATIVE, VarlenLength.INDIVIDUAL)
    assert position in (VarlenPosition.IMPLIED, VarlenPosition.REUSE, VarlenPosition.ARRAY)
    assert not (position == VarlenPosition.REUSE and length != VarlenLength.CUMULATIVE)
    return stacked, length, position


def split_bits(bits):
    """(q_side, k_side, lse_layout)."""
    return (bits & 0xFF,
            (bits >> VARLEN_K_SIDE_SHIFT) & 0xFF,
            (bits >> VARLEN_LSE_LAYOUT_SHIFT) & 3)


class SideLayout:
    """One side (Q or K) of a VarlenBits configuration, materialised.

    Attributes:
        tensor      logical (Z, H, S, D); D is always stride-1
        seqinfo0    length source, or None
        seqinfo1    position source, or None
        seqlens     per-sequence used lengths (list of int)
        max_seqlen  what the kernel is told Max_seqlen_? is
        refs        per-sequence dense (H, seqlen, D) contiguous copies -- what
                    the oracle feeds to its dense calls
    """

    def __init__(self, tensor, seqinfo0, seqinfo1, seqlens, max_seqlen, refs):
        self.tensor = tensor
        self.seqinfo0 = seqinfo0
        self.seqinfo1 = seqinfo1
        self.seqlens = seqlens
        self.max_seqlen = max_seqlen
        self.refs = refs


def _null(device):
    return torch.empty([0], device=device, dtype=torch.int32)


def _i32(values, device):
    return torch.tensor([int(v) for v in values], dtype=torch.int32, device=device)


def build_side(side_bits, seqlens, num_heads, head_dim, dtype, device,
               generator, slots=None):
    """Materialise one side.  See the module docstring for `slots`."""
    stacked, length, position = split_side(side_bits)
    seqlens = [int(s) for s in seqlens]
    n = len(seqlens)
    longest_used = max(seqlens) if seqlens else 0

    if slots is None:
        slots = list(seqlens)
    slots = [int(s) for s in slots]
    assert len(slots) == n and all(sl >= s for sl, s in zip(slots, seqlens))
    # What the kernel is told Max_seqlen_? is: the physical capacity, not the
    # longest used length. Those differ only when a slot is larger than what it
    # holds, which is the `seqused_k` case; everywhere else `slots == seqlens`
    # and this is just `max(seqlens)`.
    max_seqlen = max(slots) if n else 0

    if length == VarlenLength.MAX:
        # Every sequence is the full length by construction, so nothing shorter
        # is expressible here.
        assert all(s == longest_used for s in seqlens), \
            'LENGTH == MAX cannot express ragged lengths'

    # The per-sequence payload. Generated first, placed second, so the
    # reference is never derived from the address arithmetic under test.
    refs = [torch.rand((num_heads, s, head_dim), dtype=dtype, device=device,
                       generator=generator)
            for s in seqlens]

    # Where each sequence starts along the token axis, computed here by an
    # ordinary cumsum rather than by asking the decoder.
    if not stacked:
        starts = [0] * n
    elif position == VarlenPosition.IMPLIED:
        starts = [z * max_seqlen for z in range(n)]
        slots = [max_seqlen] * n
    elif position == VarlenPosition.REUSE:
        # The position comes out of the CUMULATIVE length array, so the slots
        # are the lengths -- no gaps are expressible.
        assert slots == seqlens, 'POSITION == REUSE cannot express padded slots'
        starts = np.cumsum([0] + seqlens[:-1]).tolist()
    else:  # ARRAY
        starts = np.cumsum([0] + slots[:-1]).tolist()

    if stacked:
        total = (starts[-1] + slots[-1]) if n else 0
        # THD in memory: (1, T, H, D) transposed to a logical (1, H, T, D).
        buf = FILLER_LO + torch.rand((1, total, num_heads, head_dim),
                                     dtype=dtype, device=device, generator=generator)
        tensor = buf.transpose(1, 2)
        for z in range(n):
            tensor[0, :, starts[z]:starts[z] + seqlens[z], :] = refs[z]
    else:
        buf = FILLER_LO + torch.rand((n, max_seqlen, num_heads, head_dim),
                                     dtype=dtype, device=device, generator=generator)
        tensor = buf.transpose(1, 2)
        for z in range(n):
            tensor[0 + z, :, :seqlens[z], :] = refs[z]

    if length == VarlenLength.MAX:
        seqinfo0 = _null(device)
    elif length == VarlenLength.CUMULATIVE:
        seqinfo0 = _i32(np.cumsum([0] + seqlens), device)
    else:  # INDIVIDUAL. Slot [N] is never read for lengths, but the LSE token
        # pitch may read [N] of the *position* array, which is seqinfo1 here.
        seqinfo0 = _i32(seqlens + [0], device)

    if position == VarlenPosition.ARRAY:
        # Prefix sum with its total in slot [N] -- the precondition
        # `lse_token_pitch` relies on.
        seqinfo1 = _i32(starts + [starts[-1] + slots[-1] if n else 0], device)
    else:
        seqinfo1 = _null(device)

    return SideLayout(tensor, seqinfo0, seqinfo1, seqlens, max_seqlen, refs)


class VarlenCase:
    """A full (Q side, K side, LSE layout) configuration, materialised."""

    def __init__(self, bits, seqlens_q, seqlens_k,
                 num_head_q, num_head_k, hdim_qk, hdim_vo,
                 dtype=torch.float16, device='cuda', seed=20,
                 slots_q=None, slots_k=None):
        q_side, k_side, lse_layout = split_bits(bits)
        assert len(seqlens_q) == len(seqlens_k)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        self.bits = bits
        self.lse_layout = lse_layout
        self.n = len(seqlens_q)
        self.num_head_q = num_head_q
        self.num_head_k = num_head_k
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.dtype = dtype
        self.device = device

        self.q = build_side(q_side, seqlens_q, num_head_q, hdim_qk,
                            dtype, device, gen, slots=slots_q)
        self.k = build_side(k_side, seqlens_k, num_head_k, hdim_qk,
                            dtype, device, gen, slots=slots_k)
        # V shares K's addressing exactly -- same bits, same slots, same
        # sequence lengths -- so it is built with the same call.
        self.v = build_side(k_side, seqlens_k, num_head_k, hdim_vo,
                            dtype, device, gen, slots=slots_k)

    @property
    def seqlens_q(self):
        return self.q.seqlens

    @property
    def seqlens_k(self):
        return self.k.seqlens

    def lse_tokens(self):
        """What the kernel's `lse_token_pitch` must produce, computed here."""
        stacked, length, position = split_side(self.bits & 0xFF)
        if not stacked:
            return self.q.max_seqlen
        if position == VarlenPosition.IMPLIED:
            return self.n * self.q.max_seqlen
        if position == VarlenPosition.REUSE:
            return int(self.q.seqinfo0[self.n].item())
        return int(self.q.seqinfo1[self.n].item())

    def lse_batches(self):
        stacked, _, _ = split_side(self.bits & 0xFF)
        return 1 if stacked else self.n

    def new_lse(self, fill=float('nan')):
        """A correctly shaped, contiguous LSE (or Delta) buffer."""
        tokens = self.lse_tokens()
        zdim = self.lse_batches()
        if self.lse_layout == VarlenLseLayout.HT:
            shape = (zdim * self.num_head_q, tokens)
        else:
            shape = (zdim * tokens, self.num_head_q)
        return torch.full(shape, fill, device=self.device, dtype=torch.float32)

    def lse_slice(self, lse, z):
        """Rows `[0, seqlen_q[z])` of sequence `z`, as (H, seqlen_q).

        Indexed by the layout this case declares, computed here rather than by
        the kernel's formula.
        """
        stacked, _, position = split_side(self.bits & 0xFF)
        if not stacked:
            batch, start = z, 0
        else:
            batch = 0
            if position == VarlenPosition.IMPLIED:
                start = z * self.q.max_seqlen
            elif position == VarlenPosition.REUSE:
                start = int(self.q.seqinfo0[z].item())
            else:
                start = int(self.q.seqinfo1[z].item())
        s = self.q.seqlens[z]
        tokens = self.lse_tokens()
        h = self.num_head_q
        if self.lse_layout == VarlenLseLayout.HT:
            view = lse.reshape(-1)[batch * h * tokens:(batch + 1) * h * tokens]
            return view.reshape(h, tokens)[:, start:start + s]
        view = lse.reshape(-1)[batch * tokens * h:(batch + 1) * tokens * h]
        return view.reshape(tokens, h)[start:start + s, :].transpose(0, 1)

    def out_like_q(self, fill=float('nan')):
        """An output tensor with Q's addressing and V's head dim."""
        t = self.q.tensor
        buf = torch.full((t.shape[0], t.shape[2], self.num_head_q, self.hdim_vo),
                         fill, dtype=self.dtype, device=self.device)
        return buf.transpose(1, 2)

    def q_row_slice(self, tensor, z):
        """Rows of `tensor` (Q-addressed) belonging to sequence `z`."""
        stacked, _, position = split_side(self.bits & 0xFF)
        s = self.q.seqlens[z]
        if not stacked:
            return tensor[z, :, :s, :]
        if position == VarlenPosition.IMPLIED:
            start = z * self.q.max_seqlen
        elif position == VarlenPosition.REUSE:
            start = int(self.q.seqinfo0[z].item())
        else:
            start = int(self.q.seqinfo1[z].item())
        return tensor[0, :, start:start + s, :]

    def k_row_slice_full(self, tensor, z, length):
        """The first `length` rows of sequence `z`'s *slot*, used or not.

        The `seqused_k` gate needs this: the result must equal a dense call on
        the truncated K and *differ* from one on the full slot.
        """
        return self.k_row_slice(tensor, z, length=length)

    def k_row_slice(self, tensor, z, length=None):
        """Rows of `tensor` (K-addressed) belonging to sequence `z`."""
        stacked, _, position = split_side((self.bits >> VARLEN_K_SIDE_SHIFT) & 0xFF)
        s = self.k.seqlens[z] if length is None else length
        if not stacked:
            return tensor[z, :, :s, :]
        if position == VarlenPosition.IMPLIED:
            start = z * self.k.max_seqlen
        elif position == VarlenPosition.REUSE:
            start = int(self.k.seqinfo0[z].item())
        else:
            start = int(self.k.seqinfo1[z].item())
        return tensor[0, :, start:start + s, :]
