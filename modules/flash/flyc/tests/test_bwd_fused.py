#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fused single-pass backward (`tuning_fused_bwd`), JIT, against fp32 and the split path.

Two references, for two different questions:

- **fp32 torch autograd** says the kernel computes the gradient: dQ, dK and dV
  against a float32 forward+backward built from the same bf16/f16 inputs.
  Covers every Phase-1 axis: head dims on and off the ladder (padded heads 48,
  72, 100), ragged and odd lengths (1, 17, 37, 100, 200), Sq != Sk both ways,
  causal (bottom-right, the forward's default), GQA and MQA, bf16 and f16, and
  every KV rung (64, 128, 256).
- **The split flyc path** (dK/dV + dQ kernels, delta from torch) says the
  features the split kernels own -- dropout's philox mask, the bias input,
  sliding windows, packed varlen -- reach the fused kernel bit-for-bit in
  meaning: both arms call the same code for them, so they must agree to
  bf16 noise.

Outputs start as NaN so an element the kernel forgets to write fails loudly.

Builds for the first GPU's arch through `tuning_fused_bwd` (gfx950 only);
skips elsewhere, or without `AOTRITON_FLYDSL_KERNEL_ROOT`. Use `pytest -s` if a
case faults -- the ROCr message is otherwise swallowed and a GPU fault looks
like a bare SIGABRT.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/tests/test_bwd_fused.py
"""

import math
import sys
from pathlib import Path

import pytest

import jit_arch
import tuning_dkdv
import tuning_dq
import tuning_fused_bwd
from fmha_meta import CAUSAL_BOTTOM_RIGHT, CAUSAL_NONE, CAUSAL_WINDOW, FmhaInputMetadata

torch = pytest.importorskip("torch")
ARCH = jit_arch.require(tuning_fused_bwd)

import fmha_utils  # noqa: E402

_DTYPES = {"bf16": "bfloat16", "f16": "float16"}
# Max |err| / max |ref| per gradient. bf16 lands at 3e-3..6e-3 on these
# shapes, f16 at 5e-4; the bound leaves headroom for causal rows whose dS
# carries a large P.
_TOL = 2e-2


def _reference(q, k, v, do, causal, scale):
    """fp32 forward + autograd backward; LSE as the forward reports it."""
    sq, sk = q.shape[2], k.shape[2]
    g = q.shape[1] // k.shape[1]
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    s = (qf @ kf.repeat_interleave(g, 1).transpose(-1, -2)) * scale
    if causal:
        i = torch.arange(sq, device=q.device)[:, None]
        j = torch.arange(sk, device=q.device)[None, :]
        s = s.masked_fill(j > i + (sk - sq), float("-inf"))
    lse = torch.logsumexp(s, -1)
    o = torch.softmax(s, -1).nan_to_num() @ vf.repeat_interleave(g, 1)
    o.backward(do.float())
    lse = torch.where(torch.isinf(lse), torch.full_like(lse, float("inf")), lse)
    return o.detach(), lse.detach(), qf.grad, kf.grad, vf.grad


def _meta(*, num_heads, num_kv_heads, head_dim, dtype_str="bf16", causal=False, window=False, **feats):
    """JIT metadata; a causal build without a window is bottom-right."""
    causal_type = CAUSAL_WINDOW if window else (CAUSAL_BOTTOM_RIGHT if causal else CAUSAL_NONE)
    return FmhaInputMetadata(num_heads=num_heads,
                             num_kv_heads=num_kv_heads,
                             block_dmodel=None,
                             head_dim=head_dim,
                             dtype_str=dtype_str,
                             causal_type=causal_type,
                             **feats)


def _fused(max_seqlen_k, **problem):
    meta, knobs = tuning_fused_bwd.plan(ARCH, _meta(**problem), max_seqlen_k=max_seqlen_k)
    return tuning_fused_bwd.builder(ARCH, meta, knobs)


def _split(dispatcher, overrides=None, **problem):
    fn, _ = jit_arch.build(dispatcher, ARCH, _meta(**problem), overrides and (ARCH, overrides))
    return fn


def _padded(shape, d, dtype, fill=None):
    """A `[..., :d]` view of a ceil8(d)-wide allocation: the 8xD input contract."""
    dp = (d + 7) // 8 * 8
    t = torch.randn(*shape, dp, device="cuda", dtype=dtype) if fill is None else torch.full(
        (*shape, dp), fill, device="cuda", dtype=dtype
    )
    return t[..., :d]


@pytest.mark.parametrize(
    "B,Hq,Hk,Sq,Sk,D,causal,dtype",
    [
        # every KV rung, square
        (2, 4, 4, 64, 64, 64, False, "bf16"),
        (2, 4, 4, 128, 128, 64, True, "bf16"),
        (2, 4, 4, 256, 256, 64, True, "bf16"),
        (2, 4, 4, 128, 128, 128, False, "bf16"),
        (2, 4, 4, 64, 64, 128, True, "f16"),
        (2, 4, 4, 128, 128, 32, True, "bf16"),
        (2, 4, 4, 256, 256, 32, False, "bf16"),
        # odd and ragged
        (2, 4, 4, 1, 64, 64, True, "bf16"),
        (2, 4, 4, 17, 17, 128, True, "bf16"),
        (2, 4, 4, 100, 37, 64, True, "bf16"),
        (2, 4, 4, 37, 100, 32, True, "bf16"),
        (2, 4, 4, 200, 128, 64, False, "f16"),
        (2, 4, 4, 500, 64, 64, True, "bf16"),
        (2, 4, 4, 150, 256, 64, True, "bf16"),
        # padded heads
        (1, 4, 4, 96, 96, 48, False, "bf16"),
        (1, 4, 4, 96, 96, 100, True, "bf16"),
        (1, 4, 4, 77, 120, 72, True, "f16"),
        (2, 4, 4, 33, 250, 40, True, "bf16"),
        # GQA and MQA
        (2, 8, 2, 100, 77, 64, True, "bf16"),
        (2, 8, 1, 64, 64, 64, False, "bf16"),
        (2, 6, 2, 130, 128, 128, False, "f16"),
        (2, 8, 2, 256, 256, 64, True, "bf16"),
    ],
)
def test_against_fp32(B, Hq, Hk, Sq, Sk, D, causal, dtype):
    dt = getattr(torch, _DTYPES[dtype])
    torch.manual_seed(0)
    q, do = _padded((B, Hq, Sq), D, dt), _padded((B, Hq, Sq), D, dt)
    k, v = _padded((B, Hk, Sk), D, dt), _padded((B, Hk, Sk), D, dt)
    scale = 1.0 / math.sqrt(D)
    o32, lse, dq_ref, dk_ref, dv_ref = _reference(q, k, v, do, causal, scale)
    o = _padded((B, Hq, Sq), D, dt, fill=0.0)
    o.copy_(o32.to(dt))
    lse = lse.reshape(B * Hq, Sq).contiguous()
    dq = _padded((B, Hq, Sq), D, dt, fill=float("nan"))
    dk = _padded((B, Hk, Sk), D, dt, fill=float("nan"))
    dv = _padded((B, Hk, Sk), D, dt, fill=float("nan"))
    f = _fused(Sk,
               num_heads=Hq,
               num_kv_heads=Hk,
               head_dim=D,
               causal=causal,
               dtype_str=dtype)
    f(q, k, v, o, do, dq, dk, dv, lse, B, Sq, seqlen_k=Sk, scale=scale)
    torch.cuda.synchronize()
    for name, got, ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        err = (got.float() - ref).abs().max().item()
        assert not math.isnan(err), f"{name} has NaN: an element was not written, or the arithmetic broke"
        assert err <= _TOL * ref.abs().max().item(), f"{name}: max err {err:.3e} vs |ref| {ref.abs().max().item():.3e}"


def _split_and_fused(*, B, H, Hk, sq, sk, D, causal=False, window=None, bias=False, dropout=None, varlen=False):
    """Run both arms on one problem; return the three relative disagreements."""
    dt = torch.bfloat16
    torch.manual_seed(1)
    if varlen:
        lens_q, lens_k = [sq, max(1, sq // 2), sq - 7], [sk, max(1, sk // 3), sk - 5]
        cu = lambda ls: torch.tensor([0] + list(torch.tensor(ls).cumsum(0)), device="cuda", dtype=torch.int32)  # noqa: E731
        vl, nseq, Bt, Sq, Sk = fmha_utils.varlen_compact(cu(lens_q), cu(lens_k), max(lens_q), max(lens_k)), 3, 1, sum(lens_q), sum(lens_k)
        msq, msk = max(lens_q), max(lens_k)
    else:
        vl, nseq, Bt, Sq, Sk, msq, msk = None, 0, B, sq, sk, sq, sk
    q, do = (torch.randn(Bt, H, Sq, D, device="cuda", dtype=dt) for _ in range(2))
    k, v = (torch.randn(Bt, Hk, Sk, D, device="cuda", dtype=dt) for _ in range(2))
    o = torch.randn(Bt, H, Sq, D, device="cuda", dtype=dt) * 0.2
    lse = (torch.rand(Bt * H, Sq, device="cuda") * 2 + math.log(msk) + 1).float()
    delta = (o.float() * do.float()).sum(-1).reshape(Bt * H, Sq).contiguous()
    kw = dict(scale=1 / math.sqrt(D), varlen=vl, num_seqlens=nseq)
    if window is not None:
        kw["window"] = window
    if bias:
        kw["bias"] = torch.randn(Bt, H, Sq, Sk, device="cuda", dtype=dt)
    if dropout is not None:
        kw.update(dropout_p=dropout, philox_seed=1234, philox_offset1=None, philox_offset2=77)
    feats = dict(causal=causal, window=window is not None, bias=bias, dropout=dropout is not None)
    heads = dict(num_heads=H, num_kv_heads=Hk, head_dim=D)
    fdk = _split(tuning_dkdv, **heads, **feats)
    # No dB: the fused kernel does not produce it.
    fdq = _split(tuning_dq, {"store_db": False}, **heads, **feats)
    ffu = _fused(msk, **heads, **feats)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dq2, dk2, dv2 = (torch.full_like(t, float("nan")) for t in (q, k, v))
    fdk(q, k, v, do, dk, dv, lse, delta, Bt, msq, seqlen_k=msk, **kw)
    fdq(q, k, v, do, dq, lse, delta, Bt, msq, seqlen_k=msk, **kw)
    ffu(q, k, v, o, do, dq2, dk2, dv2, lse, Bt, msq, seqlen_k=msk, **kw)
    torch.cuda.synchronize()
    return [((a.float() - b.float()).abs().max() / b.float().abs().max()).item() for a, b in ((dq2, dq), (dk2, dk), (dv2, dv))]


@pytest.mark.parametrize(
    "case",
    [
        dict(B=2, H=4, Hk=4, sq=100, sk=100, D=64, dropout=0.2),
        dict(B=2, H=8, Hk=2, sq=64, sk=77, D=64, causal=True, dropout=0.3),
        dict(B=2, H=4, Hk=4, sq=100, sk=100, D=64, bias=True),
        dict(B=2, H=4, Hk=2, sq=64, sk=64, D=128, bias=True),
        dict(B=2, H=4, Hk=4, sq=128, sk=128, D=64, causal=True, window=(16, 0)),
        dict(B=2, H=4, Hk=4, sq=100, sk=120, D=64, causal=True, window=(40, -8)),
        dict(B=1, H=4, Hk=4, sq=100, sk=90, D=64, varlen=True),
        dict(B=1, H=4, Hk=2, sq=64, sk=120, D=64, varlen=True, causal=True),
    ],
    ids=["dropout", "dropout-causal-gqa", "bias", "bias-hd128-gqa", "window", "window-ragged", "varlen", "varlen-causal-gqa"],
)
def test_agrees_with_split(case):
    errs = _split_and_fused(**case)
    assert all(e < _TOL for e in errs), f"fused vs split (dq, dk, dv): {errs}"


def _keep_mask(B, H, sq, sk, p, seed, offset2):
    """The dropout keep mask, `[B, H, sq, sk]`, read back through the split dK/dV kernel.

    With Q = K = 0 and LSE = 0 every `P` is 1, and with dO the identity
    `dV[j, i] = keep[i, j] * dropout_scale`. The mask is a function of element
    coordinates, the seed/offset and `(max_seqlen_q, max_seqlen_k)` alone --
    the reproducibility contract `init_philox` states -- so a head_dim 128
    build reads out the mask every build draws. Needs `sq <= 128`.
    """
    width = 128
    assert sq <= width
    f = _split(tuning_dkdv,
               num_heads=H,
               num_kv_heads=H,
               head_dim=width,
               dropout=True,
               dtype_str="bf16")
    zq = torch.zeros(B, H, sq, width, device="cuda", dtype=torch.bfloat16)
    zk = torch.zeros(B, H, sk, width, device="cuda", dtype=torch.bfloat16)
    do = torch.zeros_like(zq)
    do[..., torch.arange(sq), torch.arange(sq)] = 1.0
    rows = torch.zeros(B * H, sq, device="cuda")
    dk, dv = torch.zeros_like(zk), torch.zeros_like(zk)
    f(zq,
      zk,
      zk,
      do,
      dk,
      dv,
      rows,
      rows,
      B,
      sq,
      seqlen_k=sk,
      scale=1.0,
      dropout_p=p,
      philox_seed=seed,
      philox_offset1=None,
      philox_offset2=offset2)
    torch.cuda.synchronize()
    return dv[..., :sq].float().transpose(-1, -2) != 0


@pytest.mark.parametrize(
    "B,Hq,Hk,Sq,Sk,D,causal,dtype",
    [
        # The 32-row family with dropout. Its builds sat at 261 SGPR spills and
        # the backend's lost-SGPR-copy defect overwrote the dK/dV row strides
        # in their spill lanes: dQ right, ~90% of dK/dV never written, the rest
        # on the wrong rows (a fault under stock flydsl 0.4.2). See
        # `FusedKvOutputs`.
        (2, 4, 2, 125, 128, 32, True, "bf16"),
        # The 16-row family's worst dropout build (194 SGPR spills): NaN or
        # `hipErrorIllegalAddress` from one binary, fixed by the same change.
        (2, 4, 2, 64, 64, 96, False, "f16"),
    ],
)
def test_dropout_against_fp32(B, Hq, Hk, Sq, Sk, D, causal, dtype):
    """Dropout against fp32 autograd, with the mask the kernels draw.

    The split-path comparison above cannot catch a defect both arms share, and
    it only covers the 16-row family; this checks the gradient itself, on the
    two builds the lost-SGPR-copy defect broke.
    """
    p, seed, offset2 = 0.2, 1234, 77
    keep = _keep_mask(B, Hq, Sq, Sk, p, seed, offset2)
    assert abs(keep.float().mean().item() - (1 - p)) < 0.02, "the mask read-back is not a dropout mask"
    dt = getattr(torch, _DTYPES[dtype])
    torch.manual_seed(0)
    q, do = (torch.randn(B, Hq, Sq, D, device="cuda", dtype=dt) for _ in range(2))
    k, v = (torch.randn(B, Hk, Sk, D, device="cuda", dtype=dt) for _ in range(2))
    scale = 1.0 / math.sqrt(D)
    g = Hq // Hk
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    s = (qf @ kf.repeat_interleave(g, 1).transpose(-1, -2)) * scale
    if causal:
        i = torch.arange(Sq, device="cuda")[:, None]
        j = torch.arange(Sk, device="cuda")[None, :]
        s = s.masked_fill(j > i + (Sk - Sq), float("-inf"))
    lse = torch.logsumexp(s, -1)
    o = (torch.softmax(s, -1).nan_to_num() * keep / (1 - p)) @ vf.repeat_interleave(g, 1)
    o.backward(do.float())
    dq, dk, dv = (torch.full_like(t, float("nan")) for t in (q, k, v))
    f = _fused(Sk,
               num_heads=Hq,
               num_kv_heads=Hk,
               head_dim=D,
               causal=causal,
               dropout=True,
               dtype_str=dtype)
    f(q,
      k,
      v,
      o.detach().to(dt),
      do,
      dq,
      dk,
      dv,
      lse.reshape(B * Hq, Sq).contiguous(),
      B,
      Sq,
      seqlen_k=Sk,
      scale=scale,
      dropout_p=p,
      philox_seed=seed,
      philox_offset1=None,
      philox_offset2=offset2)
    torch.cuda.synchronize()
    for name, got, ref in (("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)):
        err = (got.float() - ref).abs().max().item()
        assert not math.isnan(err), f"{name} has NaN: an element was not written, or the arithmetic broke"
        assert err <= _TOL * ref.abs().max().item(), f"{name}: max err {err:.3e} vs |ref| {ref.abs().max().item():.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
