#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""JIT-level gates for VarlenBits.

**The oracle is N separate dense calls through the same kernel with
`Varlen_bits == 0`, compared bitwise.**  Not against a PyTorch reference with
tolerances: an addressing bug that lands inside the right allocation returns
plausible data, and a tolerance compare accepts it.  A varlen workgroup and its
dense counterpart cover the same tiles in the same order with the same values;
only the base address differs, so every floating-point operation is identical
and `torch.equal` is the right comparison.

Two caveats a later reader should not weaken:

- The grid differs between the two runs.  That changes nothing any surviving
  workgroup computes, so it does not weaken the gate.
- LSE and Delta live in a different layout.  Compare per-sequence slices,
  never whole buffers.

Run with `TRITON_F32_DEFAULT=ieee`; `attn_torch_function` asserts it at import.
"""

import numpy as np
import pytest
import torch
import triton

from flash import (
    attn_fwd as bare_attn_fwd,
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq,
)
from varlen_bits import (
    VARLEN_BITS_DENSE,
    VARLEN_BITS_COMPACT,
    VARLEN_BITS_PADDED,
    VARLEN_BITS_STRIDED,
    VARLEN_SIDE_COMPACT,
    VARLEN_SIDE_DENSE,
    VARLEN_SIDE_SEQUSED_BHSD,
    VARLEN_SIDE_SEQUSED_PACKED,
    VarlenLseLayout,
    make_varlen_bits,
)
from _varlen_bits_layout import VarlenCase

CAUSAL_NONE = 0
CAUSAL_WINDOWED = 3
WINDOW_TOP_LEFT = -2147483647
WINDOW_BOTTOM_RIGHT = -2147483646

BLOCK_M = 32
BLOCK_N = 32
BLOCK_DMODEL = 64
DTYPE = torch.float16
SM_SCALE = 1.2


def causal_window(causal):
    """(CAUSAL_TYPE, Window_left, Window_right).

    Always the sentinels, never a resolved bound: the sentinel is what makes
    the window resolve against *this sequence's* lengths, which is the only
    thing that lets a varlen call and its per-sequence dense counterpart agree.
    """
    if causal is None or causal is False:
        return CAUSAL_NONE, 0, 0
    if causal == 'top_left':
        return CAUSAL_WINDOWED, WINDOW_TOP_LEFT, WINDOW_TOP_LEFT
    if causal == 'bottom_right':
        return CAUSAL_WINDOWED, WINDOW_BOTTOM_RIGHT, WINDOW_BOTTOM_RIGHT
    left, right = causal
    return CAUSAL_WINDOWED, left, right


def null_i32(device):
    return torch.empty([0], device=device, dtype=torch.int32)


def launch_fwd(bits, q, k, v, out, lse,
               seqinfo_q0, seqinfo_k0, seqinfo_q1, seqinfo_k1,
               max_seqlen_q, max_seqlen_k, n_seq,
               num_head_q, num_head_k, hdim_qk, hdim_vo, causal):
    device = q.device
    nul = null_i32(device)
    u64nul = torch.empty([0], device=device, dtype=torch.uint64)
    b = torch.empty((0, 0, 0, 0), device=device, dtype=q.dtype)
    alibi = torch.empty((0, 0), device=device, dtype=q.dtype)
    causal_type, window_left, window_right = causal_window(causal)
    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), num_head_q, n_seq)
    bare_attn_fwd[grid](
        q, k, v, b, alibi, SM_SCALE, lse, out,
        0, 0, 0, 0, 0,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        *b.stride(), *alibi.stride(),
        Num_head_q=num_head_q,
        Num_head_k=num_head_k,
        Varlen_bits=bits,
        seqinfo_q0=seqinfo_q0,
        seqinfo_k0=seqinfo_k0,
        Max_seqlen_q=max_seqlen_q,
        Max_seqlen_k=max_seqlen_k,
        seqinfo_q1=seqinfo_q1,
        seqinfo_k1=seqinfo_k1,
        BLOCK_DMODEL=BLOCK_DMODEL,
        Hdim_qk=hdim_qk,
        Hdim_vo=hdim_vo,
        PADDED_HEAD=BLOCK_DMODEL != hdim_qk,
        ENABLE_DROPOUT=False,
        dropout_p=0.0,
        philox_seed_ptr=u64nul,
        philox_offset1=u64nul,
        philox_offset2=0,
        philox_seed_output=u64nul,
        philox_offset_output=u64nul,
        RETURN_ENCODED_SOFTMAX=False,
        encoded_softmax=None,
        CAUSAL_TYPE=causal_type,
        Window_left=window_left,
        Window_right=window_right,
        BIAS_TYPE=0,
        USE_ALIBI=False,
        INT8=False,
        INT8_KV=False,
        USE_P_SCALE=False,
        PERSISTENT_TYPE=0,
        persistent_atomic_counter=nul,
        Num_CU=0,
        GRID_CU_MULTIP=2,
        Batch=n_seq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PRE_LOAD_V=False,
        NUM_XCDS=1,
        num_warps=4,
        num_stages=1,
    )


def launch_bwd(bits, q, k, v, out, do, lse, delta, dq, dk, dv,
               seqinfo_q0, seqinfo_k0, seqinfo_q1, seqinfo_k1,
               max_seqlen_q, max_seqlen_k, n_seq,
               num_head_q, num_head_k, hdim_qk, hdim_vo, causal):
    device = q.device
    nul = null_i32(device)
    u64nul = torch.empty([0], device=device, dtype=torch.uint64)
    b = torch.empty((0, 0, 0, 0), device=device, dtype=q.dtype)
    db = torch.empty((0, 0, 0, 0), device=device, dtype=q.dtype)
    causal_type, window_left, window_right = causal_window(causal)
    padded_head = BLOCK_DMODEL != hdim_qk

    grid_prep = (triton.cdiv(max_seqlen_q, BLOCK_M), num_head_q, n_seq)
    bare_bwd_preprocess[grid_prep](
        out, do, delta,
        *out.stride(), *do.stride(),
        seqinfo_q0=seqinfo_q0,
        varlen_bits=bits,
        max_seqlen_q=max_seqlen_q,
        seqinfo_q1=seqinfo_q1,
        hdim_vo=hdim_vo,
        BLOCK_M=BLOCK_M, D_HEAD=BLOCK_DMODEL,
        PADDED_HEAD=padded_head,
    )
    grid_dk_dv = (triton.cdiv(max_seqlen_k, BLOCK_N), num_head_k, n_seq)
    bare_bwd_kernel_dk_dv[grid_dk_dv](
        q, k, v, b, SM_SCALE, do,
        dk, dv,
        lse, delta,
        *q.stride(), *k.stride(), *v.stride(), *b.stride(),
        *do.stride(), *dk.stride(), *dv.stride(),
        num_head_q=num_head_q,
        num_head_k=num_head_k,
        seqinfo_q0=seqinfo_q0,
        seqinfo_k0=seqinfo_k0,
        varlen_bits=bits,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqinfo_q1=seqinfo_q1,
        seqinfo_k1=seqinfo_k1,
        hdim_qk=hdim_qk,
        hdim_vo=hdim_vo,
        dropout_p=0.0,
        philox_seed_ptr=u64nul,
        philox_offset1=u64nul,
        philox_offset2=0,
        Window_left=window_left,
        Window_right=window_right,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL_TYPE=causal_type,
        ENABLE_DROPOUT=False,
        PADDED_HEAD=padded_head,
        BIAS_TYPE=0,
        NUM_XCDS=1,
        num_warps=4, num_stages=1,
    )
    grid_dq = (triton.cdiv(max_seqlen_q, BLOCK_M), num_head_q, n_seq)
    bare_bwd_kernel_dq[grid_dq](
        q, k, v, b, SM_SCALE, do,
        dq, db,
        lse, delta,
        *q.stride(), *k.stride(), *v.stride(), *b.stride(),
        *do.stride(), *dq.stride(),
        0, 0, 0, 0,  # db strides: all zero means "no bias gradient"
        num_head_q=num_head_q,
        num_head_k=num_head_k,
        seqinfo_q0=seqinfo_q0,
        seqinfo_k0=seqinfo_k0,
        varlen_bits=bits,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqinfo_q1=seqinfo_q1,
        seqinfo_k1=seqinfo_k1,
        hdim_qk=hdim_qk,
        hdim_vo=hdim_vo,
        dropout_p=0.0,
        philox_seed_ptr=u64nul,
        philox_offset1=u64nul,
        philox_offset2=0,
        Window_left=window_left,
        Window_right=window_right,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL_TYPE=causal_type,
        ENABLE_DROPOUT=False,
        PADDED_HEAD=padded_head,
        BIAS_TYPE=0,
        NUM_XCDS=1,
        num_warps=4, num_stages=1,
    )


class Result:
    def __init__(self, out, lse, delta=None, dq=None, dk=None, dv=None):
        self.out = out
        self.lse = lse
        self.delta = delta
        self.dq = dq
        self.dk = dk
        self.dv = dv


def run_case(case, causal, backward=False, do_refs=None):
    """Run the whole batch through one varlen launch."""
    out = case.out_like_q()
    lse = case.new_lse()
    launch_fwd(case.bits, case.q.tensor, case.k.tensor, case.v.tensor, out, lse,
               case.q.seqinfo0, case.k.seqinfo0, case.q.seqinfo1, case.k.seqinfo1,
               case.q.max_seqlen, case.k.max_seqlen, case.n,
               case.num_head_q, case.num_head_k, case.hdim_qk, case.hdim_vo,
               causal)
    if not backward:
        return Result(out, lse)
    do = case.out_like_q(fill=0.0)
    for z in range(case.n):
        case.q_row_slice(do, z).copy_(do_refs[z])
    delta = case.new_lse()
    dq = torch.full_like(case.q.tensor, 0.0)
    dk = torch.full_like(case.k.tensor, 0.0)
    dv = torch.full_like(case.v.tensor, 0.0)
    launch_bwd(case.bits, case.q.tensor, case.k.tensor, case.v.tensor, out, do,
               lse, delta, dq, dk, dv,
               case.q.seqinfo0, case.k.seqinfo0, case.q.seqinfo1, case.k.seqinfo1,
               case.q.max_seqlen, case.k.max_seqlen, case.n,
               case.num_head_q, case.num_head_k, case.hdim_qk, case.hdim_vo,
               causal)
    return Result(out, lse, delta, dq, dk, dv)


def run_dense_one(qz, kz, vz, causal, num_head_q, num_head_k, hdim_qk, hdim_vo,
                  backward=False, doz=None):
    """One sequence, dense, through the same kernel with Varlen_bits == 0."""
    device = qz.device
    sq, sk = qz.shape[2], kz.shape[2]
    nul = null_i32(device)
    out = torch.full((1, num_head_q, sq, hdim_vo), float('nan'),
                     device=device, dtype=qz.dtype)
    lse = torch.full((num_head_q, sq), float('nan'), device=device, dtype=torch.float32)
    launch_fwd(VARLEN_BITS_DENSE, qz, kz, vz, out, lse,
               nul, nul, nul, nul, sq, sk, 1,
               num_head_q, num_head_k, hdim_qk, hdim_vo, causal)
    if not backward:
        return Result(out, lse)
    delta = torch.full_like(lse, float('nan'))
    dq = torch.zeros_like(qz)
    dk = torch.zeros_like(kz)
    dv = torch.zeros_like(vz)
    launch_bwd(VARLEN_BITS_DENSE, qz, kz, vz, out, doz, lse, delta, dq, dk, dv,
               nul, nul, nul, nul, sq, sk, 1,
               num_head_q, num_head_k, hdim_qk, hdim_vo, causal)
    return Result(out, lse, delta, dq, dk, dv)


def dense_inputs(case, z):
    """The per-sequence dense tensors, taken from the builder's own refs."""
    qz = case.q.refs[z].unsqueeze(0).contiguous()
    kz = case.k.refs[z].unsqueeze(0).contiguous()
    vz = case.v.refs[z].unsqueeze(0).contiguous()
    return qz, kz, vz


def assert_matches_dense(case, causal, backward=False, seed=1234):
    """The headline gate, for every sequence in `case`."""
    do_refs = None
    if backward:
        gen = torch.Generator(device=case.device)
        gen.manual_seed(seed)
        do_refs = [torch.rand((case.num_head_q, s, case.hdim_vo),
                              dtype=case.dtype, device=case.device, generator=gen)
                   for s in case.seqlens_q]
    got = run_case(case, causal, backward=backward, do_refs=do_refs)
    for z in range(case.n):
        if case.seqlens_q[z] == 0 or case.seqlens_k[z] == 0:
            # No dense oracle exists for an empty sequence: a zero-row tensor
            # has a null data pointer, so the dense call cannot be made. The
            # varlen call still ran over it, which is what keeps the case
            # useful -- it proves the empty sequence does not corrupt its
            # neighbours, since those are compared.
            continue
        qz, kz, vz = dense_inputs(case, z)
        doz = do_refs[z].unsqueeze(0).contiguous() if backward else None
        want = run_dense_one(qz, kz, vz, causal, case.num_head_q, case.num_head_k,
                             case.hdim_qk, case.hdim_vo,
                             backward=backward, doz=doz)
        assert torch.equal(case.q_row_slice(got.out, z), want.out[0]), \
            f'out mismatch on sequence {z} of {case.seqlens_q}/{case.seqlens_k}'
        assert torch.equal(case.lse_slice(got.lse, z), want.lse), \
            f'lse mismatch on sequence {z}'
        if backward:
            assert torch.equal(case.lse_slice(got.delta, z), want.delta), \
                f'delta mismatch on sequence {z}'
            assert torch.equal(case.q_row_slice(got.dq, z), want.dq[0]), \
                f'dq mismatch on sequence {z}'
            assert torch.equal(case.k_row_slice(got.dk, z), want.dk[0]), \
                f'dk mismatch on sequence {z}'
            assert torch.equal(case.k_row_slice(got.dv, z), want.dv[0]), \
                f'dv mismatch on sequence {z}'
    return got


# ---------------------------------------------------------------------------
# Gate 1 -- dense is bit-identical to the pre-change kernel
# ---------------------------------------------------------------------------

def test_dense_bit_identity():
    """`Varlen_bits == 0` must fold to what the tri-state prologue produced.

    The golden was captured from the pre-change kernel by `gen_dense_golden.py`;
    `test_forward.py` / `test_backward.py` compare against a reference with
    tolerances and so cannot see a bit-level change.
    """
    import os
    import gen_dense_golden as golden
    if not os.path.exists(golden.GOLDEN_NPZ):
        pytest.skip(f'{golden.GOLDEN_NPZ} not present; run gen_dense_golden.py')
    ref = np.load(golden.GOLDEN_NPZ)
    blob = golden.collect()
    bad = [key for key in ref.files if not np.array_equal(ref[key], blob[key])]
    assert not bad, f'dense output changed for: {bad}'


# ---------------------------------------------------------------------------
# Suite A -- one mode (compact), many length patterns
# ---------------------------------------------------------------------------

LENGTH_PATTERNS = {
    'single': ([37], [37]),
    'uniform': ([64, 64, 64], [64, 64, 64]),
    'one_long': ([4, 4, 4, 200], [4, 4, 4, 200]),
    'ragged_primes': ([7, 23, 37, 53, 67, 73, 89], [11, 29, 41, 59, 71, 79, 97]),
    'zero_leading': ([0, 33, 64], [0, 33, 64]),
    'zero_middle': ([33, 0, 64], [33, 0, 64]),
    'zero_trailing': ([33, 64, 0], [33, 64, 0]),
    # One sequence with no keys at all, among sequences that have some. An
    # all-empty K side is not expressible: the packed tensor would have zero
    # rows and so a null data pointer, which is not a layout any caller hands
    # over.
    'one_dead_k': ([33, 40, 24], [17, 0, 24]),
    # seqlen_q == 0 with a live K. Worth its own row: a *trailing* empty Q
    # sequence puts its row offset at exactly the total token count, one past
    # the last row of the packed tensor, and the K-partitioned backward grid
    # still dispatches workgroups for it because seqlen_k > 0.
    'zero_q_live_k_middle': ([33, 0, 64], [33, 24, 64]),
    'zero_q_live_k_trailing': ([33, 64, 0], [33, 64, 24]),
    # k - q varies per sequence, and is not the batch maximum for most of them.
    # A host-side bottom-right resolution would silently use the batch-wide
    # difference and pass every length set where k - q is uniform.
    'varying_kq_delta': ([16, 48, 33], [21, 88, 36]),
}

CAUSAL_MODES = [None, 'top_left', 'bottom_right', (24, 8)]


@pytest.mark.parametrize('pattern', list(LENGTH_PATTERNS))
@pytest.mark.parametrize('causal', CAUSAL_MODES, ids=str)
def test_suite_a_compact(pattern, causal):
    seqlens_q, seqlens_k = LENGTH_PATTERNS[pattern]
    case = VarlenCase(VARLEN_BITS_COMPACT, seqlens_q, seqlens_k,
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE)
    assert_matches_dense(case, causal)


def test_single_sequence_reduction():
    """Gate 2: N = 1 compact must be bitwise identical to the dense call."""
    case = VarlenCase(VARLEN_BITS_COMPACT, [64], [64],
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE)
    assert_matches_dense(case, None)


# ---------------------------------------------------------------------------
# Suite B -- one awkward length set, every mode
# ---------------------------------------------------------------------------

# Ragged, N = 7, one zero, one much longer, seqlen_q != seqlen_k, and a
# per-sequence k - q difference that is not the batch maximum.
AWKWARD_Q = [13, 0, 64, 5, 129, 32, 7]
AWKWARD_K = [40, 0, 64, 71, 33, 96, 7]
# LENGTH == MAX cannot express ragged lengths, so the modes that use it get
# their own uniform set.
UNIFORM_Q = [48] * 7
UNIFORM_K = [48] * 7

SUITE_B = {
    'dense_0x0000': (VARLEN_BITS_DENSE, UNIFORM_Q, UNIFORM_K, None, None),
    'compact_0x0B0B': (VARLEN_BITS_COMPACT, AWKWARD_Q, AWKWARD_K, None, None),
    'padded_0x0202': (VARLEN_BITS_PADDED, AWKWARD_Q, AWKWARD_K, None, None),
    'strided_0x1313': (VARLEN_BITS_STRIDED, AWKWARD_Q, AWKWARD_K,
                       [q + g for q, g in zip(AWKWARD_Q, [0, 5, 1, 64, 3, 0, 17])],
                       [k + g for k, g in zip(AWKWARD_K, [7, 0, 32, 1, 9, 2, 0])]),
    # Mixed: packed Q against a dense (BHSD, uniform) K. This is the gate for
    # the batch_index / k_batch_index split.
    'mixed_0x000B': (make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_DENSE),
                     AWKWARD_Q, [48] * 7, None, None),
    # seqused_k on packed KV: K length from an INDIVIDUAL array, K position
    # from a CUMULATIVE one -- two different tensors.
    'seqused_packed_0x150B': (
        make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_SEQUSED_PACKED),
        AWKWARD_Q, AWKWARD_K, None,
        [k + g for k, g in zip(AWKWARD_K, [9, 16, 0, 5, 40, 1, 3])]),
    # seqused_k on a rectangular BHSD cache.
    'seqused_bhsd_0x040B': (
        make_varlen_bits(VARLEN_SIDE_COMPACT, VARLEN_SIDE_SEQUSED_BHSD),
        AWKWARD_Q, AWKWARD_K, None, None),
}


@pytest.mark.parametrize('mode', list(SUITE_B))
@pytest.mark.parametrize('causal', [None, 'bottom_right'], ids=str)
def test_suite_b_modes(mode, causal):
    bits, seqlens_q, seqlens_k, slots_q, slots_k = SUITE_B[mode]
    case = VarlenCase(bits, seqlens_q, seqlens_k,
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE,
                      slots_q=slots_q, slots_k=slots_k)
    assert_matches_dense(case, causal)


def test_suite_b_gqa_compact():
    """Gate 7: GQA is orthogonal only because strides are per tensor.

    A THD K carries `num_head_k * D` as its per-token stride while Q carries
    `num_head_q * D`, so the two row offsets are scaled by different
    multipliers. That works by construction here and is worth one test.
    """
    case = VarlenCase(VARLEN_BITS_COMPACT, AWKWARD_Q, AWKWARD_K,
                      num_head_q=8, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE)
    assert_matches_dense(case, None)


# ---------------------------------------------------------------------------
# Properties that the two suites cannot see
# ---------------------------------------------------------------------------

def test_position_array_is_read():
    """Gate 4: a strided case must differ from the gapless reading of it.

    Without this an implementation that ignores `seqinfo_?1` passes everything
    above, because the compact case it falls back to is also a valid layout.
    The gaps vary per sequence so that a uniform-gap implementation cannot pass
    by accident either.
    """
    seqlens_q = [16, 33, 48]
    seqlens_k = [24, 40, 32]
    gaps_q = [0, 13, 5]
    gaps_k = [7, 0, 21]
    case = VarlenCase(VARLEN_BITS_STRIDED, seqlens_q, seqlens_k,
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE,
                      slots_q=[s + g for s, g in zip(seqlens_q, gaps_q)],
                      slots_k=[s + g for s, g in zip(seqlens_k, gaps_k)])
    got = assert_matches_dense(case, None)

    # Now read the very same buffers as if there were no gaps. The Q side keeps
    # its layout so the output slices stay comparable; only the K position array
    # is replaced by the gapless prefix sum.
    gapless_k = torch.tensor([0] + np.cumsum(seqlens_k).tolist(),
                             dtype=torch.int32, device=case.device)
    other = case.out_like_q()
    other_lse = case.new_lse()
    launch_fwd(case.bits, case.q.tensor, case.k.tensor, case.v.tensor,
               other, other_lse,
               case.q.seqinfo0, case.k.seqinfo0, case.q.seqinfo1, gapless_k,
               case.q.max_seqlen, case.k.max_seqlen, case.n,
               case.num_head_q, case.num_head_k, case.hdim_qk, case.hdim_vo,
               None)
    # Per-sequence slices, never the whole buffer: `out_like_q()` fills with NaN
    # and a strided Q layout leaves the inter-sequence gaps unwritten, so
    # `torch.equal` on the buffers is False no matter what the kernel did -- the
    # assertion would pass vacuously and this gate could never fail.
    differs = any(not torch.equal(case.q_row_slice(got.out, z),
                                  case.q_row_slice(other, z))
                  for z in range(case.n))
    assert differs, \
        'gapless K positions produced the same answer; seqinfo_k1 is ignored'


@pytest.mark.parametrize('k_side', [VARLEN_SIDE_SEQUSED_PACKED, VARLEN_SIDE_SEQUSED_BHSD],
                         ids=['packed_0x150B', 'bhsd_0x040B'])
def test_seqused_k_shortens(k_side):
    """Gate 6: `seqused_k` must match the truncated call *and differ* from the
    full one.  Only the second assertion fails if the used length is ignored.
    """
    used_k = [17, 40, 9]
    full_k = [64, 64, 64]
    seqlens_q = [24, 48, 33]
    case = VarlenCase(make_varlen_bits(VARLEN_SIDE_COMPACT, k_side),
                      seqlens_q, used_k,
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE,
                      slots_k=full_k)
    # backward=True as well as forward: torch documents seqused_k as
    # inference-only, but nothing in the decoder makes it so -- the backward
    # kernels read the same six scalars -- and a restriction the code does not
    # actually have is one that rots. dq/dk/dv/delta are compared per sequence
    # against dense calls on the TRUNCATED K, so a backward that read the unused
    # tail would disagree here.
    got = assert_matches_dense(case, None, backward=True)

    # ... and unequal to a dense call over the whole slot. The builder fills the
    # unused tail from a disjoint but comparable range, so reading it changes
    # the answer -- which is the point of asserting this half at all.
    for z in range(case.n):
        kz_full = case.k_row_slice_full(case.k.tensor, z, full_k[z])
        vz_full = case.k_row_slice_full(case.v.tensor, z, full_k[z])
        qz = case.q.refs[z].unsqueeze(0).contiguous()
        want_full = run_dense_one(qz, kz_full.unsqueeze(0).contiguous(),
                                  vz_full.unsqueeze(0).contiguous(), None,
                                  case.num_head_q, case.num_head_k,
                                  case.hdim_qk, case.hdim_vo)
        assert not torch.equal(case.q_row_slice(got.out, z), want_full.out[0]), \
            f'sequence {z} matched the *full* K; seqused_k was ignored'


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', ['compact_0x0B0B', 'padded_0x0202',
                                  'strided_0x1313', 'mixed_0x000B',
                                  'seqused_packed_0x150B', 'seqused_bhsd_0x040B'])
@pytest.mark.parametrize('causal', [None, 'bottom_right'], ids=str)
def test_backward_matches_dense(mode, causal):
    """dq/dk/dv/delta, bitwise, against N dense backward calls.

    LSE is written by `attn_fwd` and read by both backward kernels; Delta is
    written by `bwd_preprocess` and read by both. So this is the only test that
    closes the write-then-read loop over all four.
    """
    bits, seqlens_q, seqlens_k, slots_q, slots_k = SUITE_B[mode]
    case = VarlenCase(bits, seqlens_q, seqlens_k,
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE,
                      slots_q=slots_q, slots_k=slots_k)
    assert_matches_dense(case, causal, backward=True)


@pytest.mark.parametrize('pattern', ['zero_q_live_k_middle', 'zero_q_live_k_trailing',
                                     'one_dead_k', 'varying_kq_delta'])
def test_backward_degenerate_lengths(pattern):
    """Backward over sequences that dispatch workgroups with nothing to do.

    A trailing empty Q sequence has a row offset of exactly `total_tokens`, so
    every Q address it forms is one past the end of the tensor. The stores are
    masked and the loops are zero-trip, but the K-partitioned grid still
    dispatches for it whenever seqlen_k > 0.
    """
    seqlens_q, seqlens_k = LENGTH_PATTERNS[pattern]
    case = VarlenCase(VARLEN_BITS_COMPACT, seqlens_q, seqlens_k,
                      num_head_q=2, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE)
    assert_matches_dense(case, None, backward=True)


def test_backward_gqa_compact():
    case = VarlenCase(VARLEN_BITS_COMPACT, AWKWARD_Q, AWKWARD_K,
                      num_head_q=8, num_head_k=2,
                      hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE)
    assert_matches_dense(case, None, backward=True)


# ---------------------------------------------------------------------------
# Gate 9 -- LSE_LAYOUT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('base_bits', [VARLEN_BITS_DENSE, VARLEN_BITS_COMPACT,
                                       VARLEN_BITS_PADDED],
                         ids=['dense', 'compact', 'padded'])
def test_lse_layout_th(base_bits):
    """`_TH` writes the same numbers at the transposed index.

    Asserted per position against the `_HT` run rather than against itself: a
    pair of runs that are consistently wrong about where a row lives would
    still round-trip, and `lse_slice` indexes by the layout the bits declare
    rather than by the kernel's formula.
    """
    seqlens_q = [24, 48, 33] if base_bits != VARLEN_BITS_DENSE else [48] * 3
    seqlens_k = [40, 32, 33] if base_bits != VARLEN_BITS_DENSE else [48] * 3
    kwargs = dict(num_head_q=3, num_head_k=3,
                  hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL, dtype=DTYPE)
    ht = VarlenCase(base_bits, seqlens_q, seqlens_k, **kwargs)
    th_bits = base_bits | (VarlenLseLayout.TH << 16)
    th = VarlenCase(th_bits, seqlens_q, seqlens_k, **kwargs)

    got_ht = run_case(ht, None, backward=True,
                      do_refs=[torch.ones((3, s, BLOCK_DMODEL), dtype=DTYPE,
                                          device=ht.device)
                               for s in seqlens_q])
    got_th = run_case(th, None, backward=True,
                      do_refs=[torch.ones((3, s, BLOCK_DMODEL), dtype=DTYPE,
                                          device=th.device)
                               for s in seqlens_q])

    # Shape is what the layout declares, not merely the same element count.
    zdim = ht.lse_batches()
    tokens = ht.lse_tokens()
    assert got_ht.lse.shape == (zdim * ht.num_head_q, tokens)
    assert got_th.lse.shape == (zdim * tokens, th.num_head_q)

    for z in range(ht.n):
        assert torch.equal(ht.lse_slice(got_ht.lse, z), th.lse_slice(got_th.lse, z)), \
            f'LSE differs between HT and TH on sequence {z}'
        assert torch.equal(ht.lse_slice(got_ht.delta, z), th.lse_slice(got_th.delta, z)), \
            f'Delta differs between HT and TH on sequence {z}'
        assert torch.equal(ht.q_row_slice(got_ht.dq, z), th.q_row_slice(got_th.dq, z))
        assert torch.equal(ht.k_row_slice(got_ht.dk, z), th.k_row_slice(got_th.dk, z))
        assert torch.equal(ht.k_row_slice(got_ht.dv, z), th.k_row_slice(got_th.dv, z))


def test_lse_layout_th_analytic():
    """A closed-form LSE, so a swapped index is wrong rather than merely moved.

    Identity-ish inputs make every element of `L` equal, which is useless for a
    layout test: a transposed buffer compares equal to a correct one. So `L`
    must vary along *both* axes.

        K rows all e_0; Q rows for head h all HEAD_SCALES[h] * e_0
            => every score is sm_scale * HEAD_SCALES[h]
        distinct seqlens_k
            => lse[h, t in z] = sm_scale * HEAD_SCALES[h] + log(seqlens_k[z])
    """
    head_scales = [1.0, 2.0, 3.0]
    seqlens_q = [8, 16, 24]
    seqlens_k = [4, 8, 63]
    num_heads = len(head_scales)
    device = 'cuda'

    for lse_layout in (VarlenLseLayout.HT, VarlenLseLayout.TH):
        bits = VARLEN_BITS_COMPACT | (lse_layout << 16)
        case = VarlenCase(bits, seqlens_q, seqlens_k,
                          num_head_q=num_heads, num_head_k=num_heads,
                          hdim_qk=BLOCK_DMODEL, hdim_vo=BLOCK_DMODEL,
                          dtype=DTYPE, device=device)
        case.q.tensor.zero_()
        case.k.tensor.zero_()
        case.v.tensor.zero_()
        for z in range(case.n):
            qz = case.q_row_slice(case.q.tensor, z)
            for h, scale in enumerate(head_scales):
                qz[h, :, 0] = scale
            case.k_row_slice(case.k.tensor, z)[:, :, 0] = 1.0
        got = run_case(case, None)
        for z in range(case.n):
            want = torch.tensor(
                [SM_SCALE * scale + float(np.log(seqlens_k[z]))
                 for scale in head_scales],
                device=device, dtype=torch.float32)
            sl = case.lse_slice(got.lse, z)
            assert sl.shape == (num_heads, seqlens_q[z])
            torch.testing.assert_close(
                sl, want[:, None].expand_as(sl), rtol=1e-3, atol=1e-3,
                msg=f'{lse_layout=} sequence {z}')
