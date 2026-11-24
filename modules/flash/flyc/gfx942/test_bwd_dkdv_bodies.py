#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every dK/dV body (narrow, wide, xwide, xxwide) over the feature cross product,
against fp64: windows (causal sentinels, numeric bands under both principal
diagonals), varlen, GQA, bias, dropout, NaN slack in padded heads, f16 and bf16,
and keys no query sees; both values of `mask_at` and of `dropout_scale_at`.
Outputs start NaN-filled, so each run also checks that every element a body owns
is written (dK = dV = 0 on keys no query sees). Tolerances are the body's own
reference test's (test_bwd_dkdv.py: narrow; test_bwd_dkdv_wide.py: the rest).

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_bwd_dkdv_bodies.py
"""

import functools

import pytest

from .jit_common import (  # noqa: F401
    BR,
    PD_SEQS,
    PD_SHAPES,
    PD_WINDOWS,
    PRINCIPAL_DIAGONAL,
    assert_written,
    band,
    dropout_keep,
    np,
    pytestmark,
    torch,
    varlen_batch,
    with_nan_slack,
)
from . import test_bwd_dkdv as narrow_ref
from . import test_bwd_dkdv_wide as wide_ref

# body -> head dims it serves, with its policy's knobs (tuning_dkdv.policy): narrow
# hd96 waves_per_eu 2 without bias, 1 with; wide kv_subs 2; xwide kv_subs 4;
# xxwide kv_subs 4 at 384, 3 at 512.
HEAD_DIMS = {'narrow': (32, 64, 96, 128), 'wide': (160, 192), 'xwide': (224, 256), 'xxwide': (384, 512)}
CASES = [(body, d) for body, dims in HEAD_DIMS.items() for d in dims]
SEED, OFF1, OFF2, DROP_P = 0x123456789ABCDEF0, 1000, 77, 0.3
MASK_ATS = ('score', 'prob')                # knobs.MASK_SCORE, MASK_PROB
DROPOUT_SCALE_ATS = ('store', 'elem')       # knobs.P_DROP_AT_STORE, P_DROP_PER_ELEM


@functools.cache
def _build(body, head_dim, causal_type, bias, dropout, padded_head, dtype_str, mask_at=None, dropout_scale_at=None):
    """`body`'s build; None knobs are the policy's."""
    from .dkdv import build_fmha_bwd_dkdv_gfx942_by_head_dim

    return build_fmha_bwd_dkdv_gfx942_by_head_dim(head_dim,
                                                  body=body,
                                                  causal_type=causal_type,
                                                  bias=bias,
                                                  dropout=dropout,
                                                  padded_head=padded_head,
                                                  dtype_str=dtype_str,
                                                  mask_at=mask_at,
                                                  dropout_scale_at=dropout_scale_at)


def _run(key, args, dk, dv, kwargs):
    """One launch on NaN-filled dK and dV; every owned element written."""
    dk.fill_(float('nan'))
    dv.fill_(float('nan'))
    _build(*key)(*args, **kwargs)
    torch.cuda.synchronize()
    assert_written(f'{key}', dK=dk, dV=dv)


def _ref_check(body, tag, got, exact, floor, unseen):
    if body == 'narrow' and exact.norm().item() == 0.0:
        # Zero analytically (each key seen by one row, P = 1): dP - Delta rounds to ~1e-7.
        err = got.double().norm().item()
        assert err <= 1e-5 * exact.numel() ** 0.5, f"{tag}: expected ~0, got norm {err:.3e}"
        if unseen is not None and bool(unseen.any()):
            assert int((got[..., unseen, :] != 0).sum()) == 0, f"{tag}: KV rows no query sees are nonzero"
    elif body == 'narrow':
        narrow_ref._check(tag, got, exact, floor, unseen)
    else:
        wide_ref._check(tag, got, exact, dead_cols=unseen)


def _dead_lse(lse, lse_mode):
    """The forward's +inf LSE on rows that see no key, or a finite 0.0 there."""
    return lse if lse_mode == 'inf' else torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)


def _dense_case(body,
                head_dim,
                shape,
                causal_type=0,
                window=None,
                p=0,
                bias=False,
                dropout=False,
                padded=False,
                dtype='bf16',
                mask_at=None,
                dropout_scale_at=None,
                lse_mode='inf'):
    """B2 H4 Hk2 (GQA) on `shape`, against fp64."""
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    b, h, hk = 2, 4, 2
    sq, sk = shape
    d = head_dim - 5 if padded else head_dim     # NaN in the slack up to ceil8(d)
    torch.manual_seed(head_dim + sq + sk + causal_type)
    if padded:
        q, do = (with_nan_slack(b, h, sq, d, dt=dt) for _ in range(2))
        k, v = (with_nan_slack(b, hk, sk, d, dt=dt) for _ in range(2))
        dk, dv = (with_nan_slack(b, hk, sk, d, dt=dt) for _ in range(2))
    else:
        q, do = (torch.randn(b, h, sq, d, device='cuda', dtype=dt) for _ in range(2))
        k, v = (torch.randn(b, hk, sk, d, device='cuda', dtype=dt) for _ in range(2))
        dk, dv = torch.empty_like(k), torch.empty_like(v)
    live = band(causal_type, window, p, sq, sk)
    bias_t = torch.randn(b, h, sq, sk, device='cuda', dtype=dt) if bias else None
    keep = dropout_keep(b, h, sq, sk, SEED, OFF1 + OFF2, DROP_P).double() if dropout else None
    scale = d ** -0.5
    dk_r, dv_r, dk_f, dv_f, lse, delta = narrow_ref._reference(q,
                                                               k,
                                                               v,
                                                               do,
                                                               scale,
                                                               dt,
                                                               live=live,
                                                               bias=bias_t,
                                                               keep=keep,
                                                               p=DROP_P)
    kwargs = dict(seqlen_k=sk, scale=scale)
    if window is not None:
        kwargs.update(window=window, principal_diagonal=PRINCIPAL_DIAGONAL[p])
    if bias:
        kwargs['bias'] = bias_t
    if dropout:
        kwargs.update(dropout_p=DROP_P,
                      philox_seed=SEED,
                      philox_offset1=torch.tensor([OFF1], dtype=torch.int64, device='cuda'),
                      philox_offset2=OFF2)
    key = (body, head_dim, causal_type, bias, dropout, padded, dtype, mask_at, dropout_scale_at)
    args = (q,
            k,
            v,
            do,
            dk,
            dv,
            _dead_lse(lse, lse_mode).reshape(b * h, sq).contiguous(),
            delta.reshape(b * h, sq).contiguous(),
            b,
            sq)
    _run(key, args, dk, dv, kwargs)
    unseen = None if live is None else ~live.any(0)
    tag = f'{key} w{window} p{p} {shape} lse={lse_mode}'
    _ref_check(body, f'dV {tag}', dv, dv_r, dv_f, unseen)
    _ref_check(body, f'dK {tag}', dk, dk_r, dk_f, unseen)


def _varlen_case(body,
                 head_dim,
                 seqs,
                 p=0,
                 causal_type=0,
                 window=None,
                 bias=False,
                 dtype='bf16',
                 mask_at=None,
                 lse_mode='inf'):
    """A compact batch of `seqs` (H4 Hk2), each sequence against its own fp64 reference."""
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    h, hk = 4, 2
    cu_q, cu_k, varlen = varlen_batch(seqs, p)
    tq, tk = cu_q[-1], cu_k[-1]
    torch.manual_seed(head_dim + len(seqs) + tq)
    q = torch.randn(1, h, tq, head_dim, device='cuda', dtype=dt)
    k = torch.randn(1, hk, tk, head_dim, device='cuda', dtype=dt)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    bias_t = torch.randn(1, h, tq, tk, device='cuda', dtype=dt) if bias else None
    scale = head_dim ** -0.5
    lse = torch.empty(h, tq, device='cuda')
    delta = torch.empty(h, tq, device='cuda')
    refs = []
    for z, (sq, sk) in enumerate(seqs):
        qs, ks = slice(cu_q[z], cu_q[z + 1]), slice(cu_k[z], cu_k[z + 1])
        live = band(causal_type, window, p, sq, sk)
        r = narrow_ref._reference(q[:, :, qs],
                                  k[:, :, ks],
                                  v[:, :, ks],
                                  do[:, :, qs],
                                  scale,
                                  dt,
                                  live=live,
                                  bias=None if bias_t is None else bias_t[:, :, qs, ks])
        lse[:, qs] = r[4][0]
        delta[:, qs] = r[5][0]
        refs.append((ks, r, None if live is None else ~live.any(0)))
    kwargs = dict(seqlen_k=max(c for _, c in seqs), num_seqlens=len(seqs), varlen=varlen, scale=scale)
    if window is not None:
        kwargs['window'] = window
    if bias:
        kwargs['bias'] = bias_t
    key = (body, head_dim, causal_type, bias, False, False, dtype, mask_at, None)
    args = (q, k, v, do, dk, dv, _dead_lse(lse, lse_mode), delta, 1, max(a for a, _ in seqs))
    _run(key, args, dk, dv, kwargs)
    for z, (ks, r, unseen) in enumerate(refs):
        tag = f'{key} w{window} p{p} seq{z} lse={lse_mode}'
        _ref_check(body, f'dV {tag}', dv[:, :, ks], r[1], r[3], unseen)
        _ref_check(body, f'dK {tag}', dk[:, :, ks], r[0], r[2], unseen)


# Window cases: causal_type 1/2 (top-left/bottom-right causal), 3 with an explicit window.
WINDOWS = ((0, None), (1, None), (2, None), (3, (100, 37)), (3, (0, 0)))
# Sq < Sk: top-left causal and small windows leave keys no query sees; Sq > Sk:
# bottom-right causal leaves rows that see no key.
SHAPES = ((333, 517), (517, 333))


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
    if shape != SHAPES[0] and (not causal_type or dtype != 'bf16'):
        pytest.skip("Sq > Sk only matters to masked rows")
    _dense_case(body,
                head_dim,
                shape,
                causal_type=causal_type,
                window=win,
                bias=bias,
                dropout=dropout,
                padded=padded,
                dtype=dtype)


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("causal_type", (0, 2))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_varlen(body, head_dim, causal_type, bias):
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    _varlen_case(body, head_dim, SEQS, causal_type=causal_type, bias=bias)


@pytest.mark.parametrize("shape", PD_SHAPES)
@pytest.mark.parametrize("window", PD_WINDOWS)
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_principal_diagonal(body, head_dim, p, window, shape):
    """Numeric windows under both principal diagonals against fp64; keys no query
    sees get dK = dV = 0."""
    if shape == 'varlen':
        _varlen_case(body, head_dim, PD_SEQS, p=p, causal_type=3, window=window)
    else:
        _dense_case(body, head_dim, shape, causal_type=3, window=window, p=p)


@pytest.mark.parametrize("dropout_scale_at", DROPOUT_SCALE_ATS)
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_dropout_scale_at(body, head_dim, causal, dropout_scale_at):
    """Where dropout's 1/(1-p) meets P for dV: both values against fp64."""
    _dense_case(body,
                head_dim,
                (300, 517),
                causal_type=2 if causal else 0,
                dropout=True,
                dropout_scale_at=dropout_scale_at)


# ---------------------------------------------------------------- the mask point against fp64
# `mask_at` (knobs.MASK_SCORE / MASK_PROB) is a knob of every dK/dV body. Each value
# is held to fp64 over windows under both principal diagonals, Sq <, == and > Sk,
# decode, chunked prefill, mixed varlen, bias, dropout, NaN slack in padded heads,
# f16 and bf16. Rows that see no key get the forward's +inf LSE, or a finite 0.0, so
# that the mask alone has to zero their P; keys no row sees must come back exactly 0.
REF_WINDOWS = (None, (BR, 0), (8, 0), (0, 8), (-1, 3), (100, 37))
REF_SHAPES = ((1, 200), (48, 200), (150, 150), (200, 70))


@pytest.mark.parametrize("lse_mode", ("inf", "finite"))
@pytest.mark.parametrize("shape", REF_SHAPES)
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window", REF_WINDOWS)
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dropout", (False, True))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("mask_at", MASK_ATS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_mask_at_reference(body, head_dim, mask_at, bias, dropout, padded, dtype, window, p, shape, lse_mode):
    causal_type = 0 if window is None else 3
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    if window is None and (p or lse_mode == 'finite'):
        pytest.skip("no window: no shift, and every row sees a key")
    _dense_case(body,
                head_dim,
                shape,
                causal_type=causal_type,
                window=window,
                p=p,
                bias=bias,
                dropout=dropout,
                padded=padded,
                dtype=dtype,
                mask_at=mask_at,
                lse_mode=lse_mode)


@pytest.mark.parametrize("lse_mode", ("inf", "finite"))
@pytest.mark.parametrize("p", (0, 1))
@pytest.mark.parametrize("window", REF_WINDOWS[1:])
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("mask_at", MASK_ATS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_mask_at_reference_varlen(body, head_dim, mask_at, dtype, window, p, lse_mode):
    """A mixed varlen batch (decode, chunked prefill, square, Sq > Sk, a sequence the
    shifted band can miss), each sequence against its own fp64 reference."""
    _varlen_case(body,
                 head_dim,
                 PD_SEQS,
                 p=p,
                 causal_type=3,
                 window=window,
                 dtype=dtype,
                 mask_at=mask_at,
                 lse_mode=lse_mode)
