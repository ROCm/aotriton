#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every forward body (narrow, wide, xwide) over the feature cross product, against
fp64: windows (causal sentinels, numeric bands under both principal diagonals),
varlen, GQA, bias, dropout, NaN slack in padded heads, f16 and bf16, the lazy
running max, and rows that see no key. Outputs start NaN-filled, so each run
also checks that every element a body owns is written (O = 0, LSE = +inf on rows
that see no key).

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_fwd_bodies.py
"""

import functools

import pytest

from .jit_common import (  # noqa: F401
    FWD_ATOL,
    PD_SEQS,
    PD_SHAPES,
    PD_WINDOWS,
    PRINCIPAL_DIAGONAL,
    assert_written,
    band,
    check_fwd,
    dropout_keep,
    fwd_reference,
    keep_mask,
    np,
    pytestmark,
    torch,
    varlen_batch,
    with_nan_slack,
)

# body -> head dims: narrow at 64/128; wide at 160 (early K/V^T) and 256 (streamed);
# xwide at 384 and 512.
HEAD_DIMS = {'narrow': (64, 128), 'wide': (160, 256), 'xwide': (384, 512)}
CASES = [(body, d) for body, dims in HEAD_DIMS.items() for d in dims]
SEED, OFF1, OFF2, DROP_P = 1234, 1000, 5, 0.2


@functools.cache
def _build(body, head_dim, causal_type, bias, dropout, padded_head, dtype_str, rescale_threshold):
    from .fwd import build_fmha_fwd_gfx942_by_head_dim

    return build_fmha_fwd_gfx942_by_head_dim(head_dim,
                                             body=body,
                                             causal_type=causal_type,
                                             bias=bias,
                                             dropout=dropout,
                                             padded_head=padded_head,
                                             dtype_str=dtype_str,
                                             rescale_threshold=rescale_threshold)


def _run(key, q, k, v, out, batch, seqlen_q, **kwargs):
    """One launch on NaN-filled O and LSE (`kwargs['lse']`); every owned element written."""
    out.fill_(float('nan'))
    kwargs['lse'].fill_(float('nan'))
    _build(*key)(q, k, v, out, batch, seqlen_q, **kwargs)
    torch.cuda.synchronize()
    assert_written(f'{key}', O=out, LSE=kwargs['lse'])


# Window cases: causal_type 1/2 (top-left/bottom-right causal), 3 with an explicit window.
WINDOWS = ((0, None), (1, None), (2, None), (3, (100, 37)), (3, (0, 0)))
# Sq > Sk: under causal masks some rows see no key.
SHAPES = ((333, 517), (517, 333))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rescale_threshold", (0.0, 8.0))
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dropout", (False, True))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_dense(body, head_dim, window, bias, dropout, padded, dtype, rescale_threshold, shape):
    causal_type, win = window
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    if shape != SHAPES[0] and (not causal_type or dtype != 'bf16'):
        pytest.skip("Sq > Sk only matters to masked rows")
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    b, h, hk = 2, 4, 2
    sq, sk = shape
    d = head_dim - 5 if padded else head_dim     # NaN in the slack up to ceil8(d)
    torch.manual_seed(head_dim + causal_type)
    if padded:
        q = with_nan_slack(b, h, sq, d, dt=dt)
        k = with_nan_slack(b, hk, sk, d, dt=dt)
        v = with_nan_slack(b, hk, sk, d, dt=dt)
    else:
        q = torch.randn(b, h, sq, d, device='cuda', dtype=dt)
        k = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
        v = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
    out = with_nan_slack(b, h, sq, d, dt=dt) if padded else torch.empty_like(q)
    lse = torch.empty(b * h, sq, device='cuda')
    scale = d ** -0.5
    kwargs = dict(seqlen_k=sk, lse=lse, scale=scale)
    bias_t = torch.randn(b, h, sq, sk, device='cuda', dtype=dt) if bias else None
    if bias:
        kwargs['bias'] = bias_t
    keep = None
    if dropout:
        kwargs.update(dropout_p=DROP_P,
                      philox_seed=SEED,
                      philox_offset1=torch.tensor([OFF1], dtype=torch.int64, device='cuda'),
                      philox_offset2=OFF2)
        keep = dropout_keep(b, h, sq, sk, SEED, OFF1 + OFF2, DROP_P)
    if win is not None:
        kwargs['window'] = win
    key = (body, head_dim, causal_type, bias, dropout, padded, dtype, rescale_threshold)
    _run(key, q, k, v, out, b, sq, **kwargs)
    ref, ref_lse = fwd_reference(q,
                                 k.repeat_interleave(h // hk, 1),
                                 v.repeat_interleave(h // hk, 1),
                                 scale,
                                 bias=bias_t,
                                 keep=keep,
                                 p=DROP_P,
                                 mask=band(causal_type, win, 0, sq, sk))
    # Rows seeing one key copy V/(1-p): bf16's half ulp at |O| >= 4 is past FWD_ATOL.
    atol = max(FWD_ATOL, 2 ** -7 * ref.abs().max().item())
    check_fwd(f'{key} {shape}', out, lse.view(b, h, sq), ref, ref_lse, atol=atol)


def _check_varlen(tag, q, k, v, out, lse, cu_q, cu_k, scale, bias=None, mask_of=None):
    """Each sequence of a compact batch (HT LSE) against its own fp64 reference."""
    h, hk = q.shape[1], k.shape[1]
    for z in range(len(cu_q) - 1):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        mask = None if mask_of is None else mask_of(qs.stop - qs.start, ks.stop - ks.start)
        ref, ref_lse = fwd_reference(q[0, :, qs],
                                     k[0, :, ks].repeat_interleave(h // hk, 0),
                                     v[0, :, ks].repeat_interleave(h // hk, 0),
                                     scale,
                                     bias=None if bias is None else bias[0, :, qs, ks],
                                     mask=mask)
        check_fwd(f'{tag} seq{z}', out[0, :, qs], lse[:, qs], ref, ref_lse)


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("causal_type", (0, 2))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_varlen(body, head_dim, causal_type, bias):
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    h, hk = 4, 2
    cu_q, cu_k, varlen = varlen_batch(SEQS)
    tq, tk = cu_q[-1], cu_k[-1]
    torch.manual_seed(head_dim)
    q = torch.randn(1, h, tq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, hk, tk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    lse = torch.empty(h, tq, device='cuda')
    bias_t = torch.randn(1, h, tq, tk, device='cuda', dtype=torch.bfloat16) if bias else None
    kwargs = dict(seqlen_k=max(b for _, b in SEQS), num_seqlens=len(SEQS), lse=lse, varlen=varlen)
    if bias:
        kwargs['bias'] = bias_t
    key = (body, head_dim, causal_type, bias, False, False, 'bf16', 0.0)
    _run(key, q, k, v, out, 1, max(a for a, _ in SEQS), **kwargs)
    _check_varlen(f'{key}',
                  q,
                  k,
                  v,
                  out,
                  lse,
                  cu_q,
                  cu_k,
                  head_dim ** -0.5,
                  bias=bias_t,
                  mask_of=(lambda sq, sk: band(2, None, 0, sq, sk)) if causal_type else None)


@pytest.mark.parametrize("shape", PD_SHAPES)
@pytest.mark.parametrize("window", PD_WINDOWS)
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_principal_diagonal(body, head_dim, p, window, shape):
    """Numeric windows under both principal diagonals against fp64; rows that see
    no key get O = 0 and LSE = +inf."""
    h = 2
    key = (body, head_dim, 3, False, False, False, 'bf16', 0.0)
    scale = head_dim ** -0.5
    if shape == 'varlen':
        cu_q, cu_k, varlen = varlen_batch(PD_SEQS, p)
        torch.manual_seed(head_dim + cu_q[-1])
        q = torch.randn(1, h, cu_q[-1], head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, h, cu_k[-1], head_dim, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        lse = torch.empty(h, cu_q[-1], device='cuda')
        _run(key,
             q,
             k,
             v,
             out,
             1,
             max(a for a, _ in PD_SEQS),
             seqlen_k=max(c for _, c in PD_SEQS),
             num_seqlens=len(PD_SEQS),
             lse=lse,
             varlen=varlen,
             window=window)
        _check_varlen(f'{key} w{window} p{p}',
                      q,
                      k,
                      v,
                      out,
                      lse,
                      cu_q,
                      cu_k,
                      scale,
                      mask_of=lambda sq, sk: keep_mask(window, p, sq, sk))
        return
    sq, sk = shape
    batch = 2
    torch.manual_seed(head_dim + sq)
    q = torch.randn(batch, h, sq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch, h, sk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    lse = torch.empty(batch * h, sq, device='cuda')
    _run(key,
         q,
         k,
         v,
         out,
         batch,
         sq,
         seqlen_k=sk,
         lse=lse,
         window=window,
         principal_diagonal=PRINCIPAL_DIAGONAL[p])
    ref, ref_lse = fwd_reference(q, k, v, scale, mask=keep_mask(window, p, sq, sk))
    check_fwd(f'{key} w{window} p{p} {shape}', out, lse.view(batch, h, sq), ref, ref_lse)
