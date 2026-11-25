#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every dQ body (narrow, wide) over the feature cross product, against fp64:
windows (causal sentinels, numeric bands under both principal diagonals), varlen,
GQA, bias and dB, dropout, NaN slack in padded heads, f16 and bf16, and rows that
see no key. Outputs start NaN-filled, so each run also checks that every element
a body owns is written (dQ = 0 on rows that see no key). Tolerances are the
body's own reference test's (test_bwd_dq.py: narrow; test_bwd_dq_wide.py: wide).

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_bwd_dq_bodies.py
"""

import functools

import pytest

from .jit_common import (  # noqa: F401
    PD_SEQS,
    PD_SHAPES,
    PD_WINDOWS,
    PRINCIPAL_DIAGONAL,
    assert_written,
    band,
    dropout_keep,
    keep_mask,
    np,
    pytestmark,
    torch,
    varlen_batch,
    with_nan_slack,
)
from . import test_bwd_dq as narrow_ref
from . import test_bwd_dq_wide as wide_ref

# body -> head dims; the narrow body's dQ groups differ at each (1, 2, 2+1, 4 columns).
# The wide body: 32x32x8 at 160/256, 16x16x16 at 384/512.
HEAD_DIMS = {'narrow': (32, 64, 96, 128), 'wide': (160, 256, 384, 512)}
CASES = [(body, d) for body, dims in HEAD_DIMS.items() for d in dims]
SEED, OFF1, OFF2, DROP_P = 1234, 1000, 5, 0.2


@functools.cache
def _build(body, head_dim, causal_type, bias, dropout, padded_head, dtype_str):
    from .dq import build_fmha_bwd_dq_gfx942_by_head_dim

    return build_fmha_bwd_dq_gfx942_by_head_dim(head_dim,
                                                body=body,
                                                causal_type=causal_type,
                                                bias=bias,
                                                dropout=dropout,
                                                padded_head=padded_head,
                                                dtype_str=dtype_str)


def _run(key, args, dq, dbias, kwargs, db_owned=None):
    """One launch on NaN-filled dQ and dB; every owned element written. `db_owned`
    lists the (row slice, column slice) blocks of dB a varlen launch owns."""
    dq.fill_(float('nan'))
    if dbias is not None:
        dbias.fill_(float('nan'))
    _build(*key)(*args, **kwargs)
    torch.cuda.synchronize()
    assert_written(f'{key}', dQ=dq)
    if dbias is not None:
        for rows, cols in db_owned or [(slice(None), slice(None))]:
            assert_written(f'{key}', dB=dbias[..., rows, cols])


def _check(body, tag, got, ref, live=None):
    (narrow_ref if body == 'narrow' else wide_ref)._check(tag, got, ref, live)


# Window cases: causal_type 1/2 (top-left/bottom-right causal), 3 with an explicit window.
WINDOWS = ((0, None), (1, None), (2, None), (3, (100, 37)), (3, (0, 0)))
# Odd Sk: bias and dB rows start mid-dword; even Sk: clean dB tiles store dwords.
# Sq > Sk: under causal masks some rows see no key.
SHAPES = ((333, 517), (300, 520), (517, 333))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dropout", (False, True))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_dense(body, head_dim, window, bias, dropout, padded, dtype, shape):
    causal_type, win = window
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    if shape == SHAPES[1] and not bias:
        pytest.skip("the even pitch only matters to dB")
    if shape == SHAPES[2] and (not causal_type or dtype != 'bf16'):
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
        do = with_nan_slack(b, h, sq, d, dt=dt)
    else:
        q = torch.randn(b, h, sq, d, device='cuda', dtype=dt)
        k = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
        v = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
        do = torch.randn(b, h, sq, d, device='cuda', dtype=dt)
    dq = with_nan_slack(b, h, sq, d, dt=dt) if padded else torch.empty_like(q)
    scale = d ** -0.5
    bias_t = torch.randn(b, h, sq, sk, device='cuda', dtype=dt) if bias else None
    keep = dropout_keep(b, h, sq, sk, SEED, OFF1 + OFF2, DROP_P) if dropout else None
    ref, ref_ds, lse, delta, live = wide_ref._reference(q,
                                                        k.repeat_interleave(h // hk, 1),
                                                        v.repeat_interleave(h // hk, 1),
                                                        do,
                                                        scale,
                                                        keep_mask=band(causal_type, win, 0, sq, sk),
                                                        bias=bias_t,
                                                        keep=keep,
                                                        p=DROP_P)
    kwargs = dict(seqlen_k=sk, scale=scale)
    dbias = None
    if bias:
        dbias = torch.empty(b, h, sq, sk, device='cuda', dtype=dt)
        kwargs.update(bias=bias_t, dbias=dbias)
    if dropout:
        kwargs.update(dropout_p=DROP_P,
                      philox_seed=SEED,
                      philox_offset1=torch.tensor([OFF1], dtype=torch.int64, device='cuda'),
                      philox_offset2=OFF2)
    if win is not None:
        kwargs['window'] = win
    key = (body, head_dim, causal_type, bias, dropout, padded, dtype)
    args = (q, k, v, do, dq, lse.reshape(b * h, sq).contiguous(), delta.reshape(b * h, sq).contiguous(), b, sq)
    _run(key, args, dq, dbias, kwargs)
    tag = f'{key} {shape}'
    _check(body, f'dQ {tag}', dq, ref, live[0, 0])
    if bias:
        _check(body, f'dB {tag}', dbias, ref_ds)


def _varlen_case(body, head_dim, seqs, p, causal_type, window, bias):
    """A compact batch of `seqs` (h 4, hk 2), each sequence against its own fp64 reference."""
    h, hk = 4, 2
    cu_q, cu_k, varlen = varlen_batch(seqs, p)
    tq, tk = cu_q[-1], cu_k[-1]
    torch.manual_seed(head_dim + tq)
    q = torch.randn(1, h, tq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, hk, tk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dq = torch.empty_like(q)
    bias_t = torch.randn(1, h, tq, tk, device='cuda', dtype=torch.bfloat16) if bias else None
    scale = head_dim ** -0.5
    lse = torch.empty(h, tq, device='cuda')
    delta = torch.empty(h, tq, device='cuda')
    refs = []
    for z in range(len(seqs)):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        r = wide_ref._reference(q[0, :, qs],
                                k[0, :, ks].repeat_interleave(h // hk, 0),
                                v[0, :, ks].repeat_interleave(h // hk, 0),
                                do[0, :, qs],
                                scale,
                                keep_mask=band(causal_type, window, p, *seqs[z]),
                                bias=None if bias_t is None else bias_t[0, :, qs, ks])
        lse[:, qs], delta[:, qs] = r[2], r[3]
        refs.append((qs, ks, r))
    kwargs = dict(seqlen_k=max(c for _, c in seqs), num_seqlens=len(seqs), varlen=varlen, scale=scale)
    if window is not None:
        kwargs['window'] = window
    dbias = None
    if bias:
        dbias = torch.empty(1, h, tq, tk, device='cuda', dtype=torch.bfloat16)
        kwargs.update(bias=bias_t, dbias=dbias)
    key = (body, head_dim, causal_type, bias, False, False, 'bf16')
    owned = [(qs, ks) for qs, ks, _ in refs]
    _run(key, (q, k, v, do, dq, lse, delta, 1, max(a for a, _ in seqs)), dq, dbias, kwargs, owned)
    for z, (qs, ks, r) in enumerate(refs):
        tag = f'{key} w{window} p{p} seq{z}'
        _check(body, f'dQ {tag}', dq[0, :, qs], r[0], r[4][0])
        if bias:
            _check(body, f'dB {tag}', dbias[0, :, qs, ks], r[1])


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("causal_type", (0, 2))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_varlen(body, head_dim, causal_type, bias):
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    _varlen_case(body, head_dim, SEQS, 0, causal_type, None, bias)


@pytest.mark.parametrize("shape", PD_SHAPES)
@pytest.mark.parametrize("window", PD_WINDOWS)
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_principal_diagonal(body, head_dim, p, window, shape):
    """Numeric windows under both principal diagonals against fp64; rows that see
    no key get dQ = 0."""
    if shape == 'varlen':
        _varlen_case(body, head_dim, PD_SEQS, p, 3, window, False)
        return
    sq, sk = shape
    b, h, hk = 2, 4, 2
    torch.manual_seed(head_dim + sq)
    q = torch.randn(b, h, sq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(b, hk, sk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dq = torch.empty_like(q)
    scale = head_dim ** -0.5
    ref, _, lse, delta, live = wide_ref._reference(q,
                                                   k.repeat_interleave(h // hk, 1),
                                                   v.repeat_interleave(h // hk, 1),
                                                   do,
                                                   scale,
                                                   keep_mask=keep_mask(window, p, sq, sk))
    kwargs = dict(seqlen_k=sk, window=window, scale=scale, principal_diagonal=PRINCIPAL_DIAGONAL[p])
    key = (body, head_dim, 3, False, False, False, 'bf16')
    args = (q, k, v, do, dq, lse.reshape(b * h, sq).contiguous(), delta.reshape(b * h, sq).contiguous(), b, sq)
    _run(key, args, dq, None, kwargs)
    _check(body, f'dQ {key} w{window} p{p} {shape}', dq, ref, live[0, 0])
