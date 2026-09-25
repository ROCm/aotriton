#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import pytest
import torch
import os

from _common_backward import _do_test_op_bwd
from _common_test import SdpaContext, SdpaParams, SdpaContextFromNPZ
from attn_torch_function import attention, AttentionExtraArgs, BWD_FUSED, CausalType

FOR_RELEASE = bool(int(os.getenv('FOR_RELEASE', default='0')))
V3_API = 0 # V3 API is only meaningful for AOTriton

def fmt_hdim(val):
    return f'hdim{val}'

BWDOP_ids = ['Fused'] if BWD_FUSED else (['V3'] if V3_API else ['Split'])

POT_HEADDIMS = [16, 32, 64, 128, 256]
NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
# PRIME_HEADDIMS = [7, 23, 37, 53, 67, 73, 89, 113, 149, 179, 211, 241]
# AOTriton does not support compact prime head dims due to memory alignment requirements
PRIME_HEADDIMS = []

# @pytest.mark.parametrize('BATCH', [1])
# @pytest.mark.parametrize('N_HEADS', [1])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [8])
# @pytest.mark.parametrize('D_HEAD', [16, 32, 64, 128, 256])
# Irregular-only PyTorch set
# @pytest.mark.parametrize('D_HEAD', [8, 21, 72, 96, 160, 192, 203])
# @pytest.mark.parametrize('seqlen_q', [1, 4, 32, 128, 256, 512, 1024, 7, 394, 250, 399, 511, 1019])
# @pytest.mark.parametrize('seqlen_k', [1, 4, 32, 128, 256, 512, 1024, 3, 217, 339, 313, 491, 988])
# PyTorch set
# @pytest.mark.parametrize('D_HEAD', [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256])
@pytest.mark.parametrize('D_HEAD', POT_HEADDIMS + NPOT_HEADDIMS + PRIME_HEADDIMS, ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# Currently debugging
# @pytest.mark.parametrize('D_HEAD', range(8,128+1,4))
# @pytest.mark.parametrize('D_HEAD', [84,92,108, 203] + list(range(128, 256+1, 4)))
# @pytest.mark.parametrize('D_HEAD', [84,203])
# @pytest.mark.parametrize('D_HEAD', range(8,64+1,4))
# @pytest.mark.parametrize('seqlen_q', [128, 2048, 4096])
# @pytest.mark.parametrize('seqlen_k', [128, 2048, 4096])
# Minimal set
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2] if not FOR_RELEASE else [1.2])
# @pytest.mark.parametrize('sm_scale', [1.2])
# @pytest.mark.parametrize('storage_flip', [False])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_op_bwd(BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [8])
@pytest.mark.parametrize('D_HEAD', POT_HEADDIMS + NPOT_HEADDIMS + PRIME_HEADDIMS, ids=fmt_hdim)
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [128, 113])
# @pytest.mark.parametrize('seqlen_k', [128, 79])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2] if not FOR_RELEASE else [1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_op_bwd_with_matrix_bias(BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
@pytest.mark.parametrize('D_HEAD', [8, 203, 256], ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 0.125] if not FOR_RELEASE else ['l1', 'l2'])
@pytest.mark.parametrize('storage_flip', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_gqa(BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

# Softmax over a single key is exactly 1 whatever the bias is, so the fwd must pass V
# through untouched and the bwd must pass dO through to dV. The bwd recomputes p from
# the LSE the fwd saved, so both only hold while the two kernels apply the log2(e)
# factor to the bias at the same precision. Spelling the fwd scale as a bare
# `bias * 1.44269504089` evaluates it at the bias dtype, rounding log2(e) to 1.4453125
# in bf16 while the bwd applies it in fp32; dV then comes back as
# 2**(bias * (log2(e)_fp32 - log2(e)_bf16)), i.e. 0.973 at bias=16 and 0.891 at
# bias=64. Biases this large are the point of the test: the random biases in
# test_op_bwd_with_matrix_bias are small enough to hide the error under its tolerance.
@pytest.mark.parametrize('bias_val', [0.0, 4.0, 16.0, -16.0, 64.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_matrix_bias_fwd_bwd_symmetry(dtype, bias_val):
    D_HEAD = 16
    q = torch.zeros((1, 1, 1, D_HEAD), device='cuda', dtype=dtype, requires_grad=True)
    k = torch.zeros((1, 1, 1, D_HEAD), device='cuda', dtype=dtype, requires_grad=True)
    v = torch.ones((1, 1, 1, D_HEAD), device='cuda', dtype=dtype, requires_grad=True)
    b = torch.full((1, 1, 1, 1), bias_val, device='cuda', dtype=dtype)
    sm_scale = D_HEAD ** -0.5

    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False)
    tri_out, _, _ = attention(q, k, v, b, False, sm_scale, 0.0, ext)

    # Not asserted exactly: converting the LSE to base 2 and back costs fp32 a few
    # 1e-7. ATOL still sits ~800x below the smallest error the asymmetry produces
    # (7.8e-3, at bias=4 in bf16).
    ATOL = 1e-5
    assert torch.allclose(tri_out, v, atol=ATOL, rtol=ATOL), \
        f'single-key fwd should return V, off by {(tri_out - v).abs().max().item()}'

    dout = torch.ones_like(tri_out)
    dq, dk, dv = torch.autograd.grad(tri_out, [q, k, v], dout)
    assert torch.allclose(dv, dout, atol=ATOL, rtol=ATOL), \
        f'single-key bwd should return dO as dV, off by {(dv - dout).abs().max().item()}'

def test_large_bf16_nan_values():
    q = torch.full((1, 1, 1, 16), 133120.0, dtype=torch.bfloat16, device="cuda")
    k = torch.full((1, 1, 1, 16), 133120.0, dtype=torch.bfloat16, device="cuda")
    v = torch.full((1, 1, 1, 16), 133120.0, dtype=torch.bfloat16, device="cuda")
    b = None
    from torch.nn.functional import scaled_dot_product_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        out = scaled_dot_product_attention(q, k, v)
    print(out)

    causal = False
    sm_scale = 0.125
    dropout_p = 0
    ext = AttentionExtraArgs(return_encoded_softmax=causal,
                             autotune=False,
                             return_autotune=False)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)

    print(tri_out)
    assert not torch.isnan(tri_out).any(), "Output should not contain NaNs!"

# ---------------------------------------------------------------------------
# Common mistakes. See docs/attention-kernel-numerical-error-lessons/ for the
# history behind each case, and for run_mutants.sh, which shows each case
# catches the mistake it names. The JIT twin of test_common_mistakes in
# modules/flash/tests/test_backward.py; keep the two in step.
#
# test_op_bwd cannot see a precision mistake, for two reasons. It measures
# against PyTorch's own low-precision SDPA with fudge factors of 3x-14x on O
# and 36x-768x on the gradients. And at the sm_scale it uses, 1/D or
# 1/sqrt(D) over N(0, 1) inputs, the logits are too small for an error that
# grows with |S| to show at all. Baking sm_scale*log2(e) into Q costs 3x-15x
# of the floor below once the logits are large. These cases measure against
# that floor instead -- see _common_mistakes_reference.
# ---------------------------------------------------------------------------

COMMON_MISTAKES_CASES = ['sharp_softmax', 'zero_sm_scale', 'bottom_right_masked_rows']

# A correct kernel lands at or below 1.0x the floor (the Triton kernels on
# gfx90a: 0.7x-1.02x for O, dQ, dK and dV over six seeds). The mistakes
# measured 2.6x-17x on the same inputs, so 2x is both honest and wide.
COMMON_MISTAKES_FLOOR_MULT = 2.0

def _common_mistakes_reference(q, k, v, dout, sm_scale, mask, round_to=None, flush_subnormals=False):
    '''fp64 attention forward and backward from the same low-precision inputs.

    With round_to=None this is the exact answer. With round_to=dtype it rounds
    to dtype exactly where a correct flash kernel has to -- P before P@V and
    P^T@dO, dS before dS@K and dS^T@Q, and every output on store -- and nowhere
    else. Its distance from the exact answer is therefore the error floor: what
    a kernel that keeps everything else in fp32 still gets.

    flush_subnormals also zeroes whatever rounds below the dtype's smallest
    normal. gfx90a's fp16 MFMA flushes subnormal inputs, which is where most of
    P lands once seqlen_k is in the hundreds; without it fp16 dV measures
    2.5x a floor that no hardware-conforming kernel can reach.

    A fully masked row has no softmax. Its contract is P = 0 there, so O = 0 and
    dQ = 0, and the LSE is -inf here (the kernel may store +inf).
    '''
    q, k, v, dout = (t.detach().to(torch.float64) for t in (q, k, v, dout))
    def R(x):
        if round_to is None:
            return x
        x = x.to(round_to).to(torch.float64)
        if flush_subnormals:
            x = torch.where(x.abs() < torch.finfo(round_to).tiny, 0.0, x)
        return x
    s = (q @ k.transpose(-1, -2)) * sm_scale
    s = s.masked_fill(~mask, float('-inf'))
    lse = torch.logsumexp(s, dim=-1, keepdim=True)
    p = torch.where(mask, torch.exp(s - lse), 0.0)
    o = R(R(p) @ v)
    dp = dout @ v.transpose(-1, -2)
    # From the O the kernel stored, which is what its backward reads.
    delta = (dout * o).sum(dim=-1, keepdim=True)
    ds = p * (dp - delta)
    dq = R((R(ds) @ k) * sm_scale)
    dk = R((R(ds).transpose(-1, -2) @ q) * sm_scale)
    dv = R(R(p).transpose(-1, -2) @ dout)
    return o, lse.squeeze(-1), dq, dk, dv

def _common_mistakes_case(case):
    '''(seqlen_q, seqlen_k, input scale, causal, sm_scale) for one case.'''
    D_HEAD = 128
    if case == 'sharp_softmax':
        # Logits with a standard deviation of 16: a trained model's attention
        # is this sharp, and the error of rounding a scaled Q or S to the input
        # dtype grows with |S| while the floor does not.
        return 257, 519, 4.0, False, D_HEAD ** -0.5
    if case == 'zero_sm_scale':
        # Uniform attention over a causal prefix. Any kernel that multiplies a
        # masked -inf by the scale computes -inf * 0 = NaN.
        return 257, 519, 1.0, True, 0.0
    if case == 'bottom_right_masked_rows':
        # The first 111 query rows attend to nothing, and 111 is off the grid
        # of every BLOCK_M, so fully masked rows share a tile with live ones
        # and cannot take the whole-tile early exit.
        return 301, 190, 1.0, CausalType.BOTTOM_RIGHT, D_HEAD ** -0.5
    assert False, f'Unknown case {case}'

def _common_mistakes_mask(causal, seqlen_q, seqlen_k, device):
    i = torch.arange(seqlen_q, device=device)[:, None]
    j = torch.arange(seqlen_k, device=device)[None, :]
    if causal is False:
        return torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    if causal is True:
        return j <= i
    assert causal == CausalType.BOTTOM_RIGHT
    return j <= i + (seqlen_k - seqlen_q)

def core_test_common_mistakes(case, dtype, device_str='cuda'):
    BATCH, N_HEADS, D_HEAD = 2, 3, 128
    seqlen_q, seqlen_k, input_scale, causal, sm_scale = _common_mistakes_case(case)
    torch.manual_seed(0)
    def randn(seqlen, scale):
        return (torch.randn(BATCH, N_HEADS, seqlen, D_HEAD, device=device_str) * scale).to(dtype)
    q = randn(seqlen_q, input_scale).requires_grad_()
    k = randn(seqlen_k, input_scale).requires_grad_()
    v = randn(seqlen_k, 1.0).requires_grad_()
    dout = randn(seqlen_q, 1.0)

    # is_testing=False so a NaN LSE reaches the diagnostics below instead of
    # the wrapper's own assert. fillnan turns an unwritten element into a NaN
    # rather than whatever the allocator left.
    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             is_testing=False,
                             fillnan=True,
                             return_logsumexp=True)
    tri_out, _, L = attention(q, k, v, None, causal, sm_scale, 0.0, ext)
    tri_dq, tri_dk, tri_dv = torch.autograd.grad(tri_out, (q, k, v), dout)
    tri_lse = L.view(BATCH, N_HEADS, seqlen_q)

    mask = _common_mistakes_mask(causal, seqlen_q, seqlen_k, device_str)
    live_rows = mask.any(dim=-1)
    exact = _common_mistakes_reference(q, k, v, dout, sm_scale, mask)
    floors = [_common_mistakes_reference(q, k, v, dout, sm_scale, mask, round_to=dtype, flush_subnormals=ftz)
              for ftz in (False, True)]

    ctx = f'{case=} {dtype=} {seqlen_q=} {seqlen_k=} {sm_scale=}'
    names = ['out', 'lse', 'dq', 'dk', 'dv']
    tri = [tri_out, tri_lse, tri_dq, tri_dk, tri_dv]
    for name, t in zip(names, tri):
        assert not torch.isnan(t).any(), f'{ctx}: {name} has NaN'

    # **LSE, to fp32 accuracy.** It is a row reduction of the scores and
    # nothing in it is rounded to the input dtype, so a correct kernel is off
    # by ~1e-6 relative. Rounding a scaled Q (or S itself) to fp16 costs 2**-11
    # of the largest score, bf16 2**-8; a base-2 LSE is off by a factor of
    # 1.44. The bound sits between: 2**-16 of the largest score.
    ref_lse = exact[1]
    s_max = max(1.0, ref_lse[..., live_rows].abs().max().item())
    lse_err = (tri_lse[..., live_rows].double() - ref_lse[..., live_rows]).abs().max().item()
    assert lse_err <= 2.0 ** -16 * s_max, \
        f'{ctx}: LSE off by {lse_err:.3e}, bound {2.0 ** -16 * s_max:.3e} (max |S| {s_max:.1f})'

    # **Fully masked rows are exactly zero.** Rows with no key get P = 0, so O
    # and dQ vanish -- not NaN, not the inits leaking out.
    if not live_rows.all():
        dead = ~live_rows
        for name, t in (('out', tri_out), ('dq', tri_dq)):
            n_nonzero = int((t[:, :, dead] != 0).sum())
            assert n_nonzero == 0, f'{ctx}: {n_nonzero} nonzero elements in fully masked {name} rows'

    # **O and every gradient, against the floor.**
    def relrms(x, ref):
        return ((x.double() - ref).norm() / ref.norm()).item()
    for i in (0, 2, 3, 4):
        name, t, ref = names[i], tri[i], exact[i]
        if ref.norm() == 0:
            # sm_scale == 0 has an exactly zero dQ and dK.
            n_nonzero = int((t != 0).sum())
            assert n_nonzero == 0, f'{ctx}: {name} should be exactly zero, {n_nonzero} elements are not'
            continue
        err = relrms(t, ref)
        floor = max(relrms(f[i], ref) for f in floors)
        assert err <= COMMON_MISTAKES_FLOOR_MULT * floor, \
            f'{ctx}: {name} error {err:.3e} is {err / floor:.1f}x the floor {floor:.3e} (limit {COMMON_MISTAKES_FLOOR_MULT}x)'

# fp32 is left out on purpose: every mistake here is a rounding to the input
# dtype, which fp32 inputs make harmless, and the floor model does not capture
# fp32 accumulation error.
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('case', COMMON_MISTAKES_CASES)
def test_common_mistakes(case, dtype):
    core_test_common_mistakes(case, dtype)

def main_npz():
    SKIP_DK_DV = False
    SKIP_DQ = False
    SKIP_DB = True
    fn = sys.argv[1]
    ctx = SdpaContextFromNPZ(fn, dtype=None, device='cuda')
    q, k, v, b = ctx.dev_tensors
    assert b is None, 'TODO: support bias in SdpaContextFromNPZ'
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)

    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False)
    causal, sm_scale, dropout_p = ctx.sdpa_params[:3]
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    ctx.compute_ref_forward(ctx.sdpa_params)

    dout = ctx.dout
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff = ctx.validate_with_reference(tri_out, ctx.dout_tensors)
    assert is_allclose
    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    torch.set_printoptions(linewidth=200, threshold=4096)
    ctx.display_validation_results(tri_out, is_allclose, adiff, grads_allclose, grads_adiff)
    # Add more printing here
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    print(f'{is_allclose=}')
    print(f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}')
    print(f'{adiff=} {grads_adiff=}')


def main3():
    tup = (1, 12, 32, 8, 8, True, 1.2, 0.5, False, torch.bfloat16, 0)
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
    if bias_type == 0:
        bias_type = None
    storage_flip = False
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main2():
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-4-1
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-1-4
    # False-1.2-dtype0-0.0-False-4-4-72-1-4
    BATCH = 1
    N_HEADS = 2
    seqlen_q = 4
    seqlen_k = 4
    D_HEAD = 16
    # BATCH = 4
    # D_HEAD = 1
    # N_HEADS = 8
    # seqlen_q = 256
    # seqlen_k = 4
    # causal = True
    causal = False
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    # bias_type = None
    bias_type = 'matrix'
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main():
    BATCH = 1
    D_HEAD = 80
    '''
    N_HEADS = 2
    seqlens_q = 6432
    seqlens_k = 6432
    '''
    N_HEADS = 6432
    seqlens_q = 2
    seqlens_k = 2
    causal = False
    sm_scale = 1.2
    dropout_p = 0.5
    dtype = torch.bfloat16
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main_nsq_causal():
    BATCH = 1
    D_HEAD = 1
    '''
    N_HEADS = 2
    seqlens_q = 6432
    seqlens_k = 6432
    '''
    N_HEADS = 1
    seqlen_q = 2
    seqlen_k = 4
    causal = True
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main_bug_introduced_when_fixing_54():
    # Original problem: https://github.com/ROCm/aotriton/issues/54
    # Failed Fix: https://github.com/ROCm/aotriton/commit/14d673f4ea90a5a4e1cea5442d22bc7b1e9146cf
    BATCH = 1
    D_HEAD = 4
    N_HEADS = 1
    seqlen_q = 64
    seqlen_k = 64
    causal = False
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    main2()
    # main_bug_introduced_when_fixing_54()
    # main_nsq_causal()
    # main_npz()
