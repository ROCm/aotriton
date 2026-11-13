# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TFLOPS (and an fp64 spot check) of the gfx942 dK/dV bodies for head dims 160..512:
the body fmha_tuning_bwd_dkdv_gfx942 routes the rung to, or --body.

dK/dV FLOPs are 4 GEMMs: 8 * B * H * pairs * D, with pairs halved (roughly) by causal.
LSE and Delta come from torch, so the time is the kernel's alone.

    AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL tree> python bench_bwd_dkdv_wide_gfx942.py \
        --batch 2 --heads 32 --seqlen 8192 --head_dim 192 --causal
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ['AOTRITON_FLYDSL_KERNEL_ROOT'])

import torch  # noqa: E402

LADDER = (160, 192, 224, 256, 384, 512)


def build(rung, body=None, **kw):
    import importlib

    import fmha_tuning_bwd_dkdv_gfx942 as tuning
    source, _, builder = tuning.BODIES[body or tuning.body(rung)]
    return getattr(importlib.import_module(source[:-3]), builder)(rung, **kw)


def side_inputs(q, k, v, do, sm_scale, causal, bias):
    """fp32 (LSE, Delta) per (batch, head), one head at a time to bound memory."""
    b, h, sq, _ = q.shape
    hk, sk = k.shape[1], k.shape[2]
    lse = torch.empty(b, h, sq, device='cuda')
    delta = torch.empty_like(lse)
    for z in range(b):
        for i in range(h):
            s = (q[z, i].float() @ k[z, i // (h // hk)].float().T) * sm_scale
            if bias is not None:
                s += bias[z, i].float()
            if causal:
                r = torch.arange(sq, device='cuda')[:, None]
                c = torch.arange(sk, device='cuda')[None, :]
                s.masked_fill_(c > r + (sk - sq), float('-inf'))
            lse[z, i] = torch.logsumexp(s, -1)
            o = (torch.softmax(s, -1).nan_to_num() @ v[z, i // (h // hk)].float()).to(q.dtype)
            delta[z, i] = (o.float() * do[z, i].float()).sum(-1)
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float('inf')))
    return lse, delta


def spot_check(q, k, v, do, dk, dv, sm_scale, causal, bias):
    """fp64 dK/dV of key head 0 of batch 0 (its group of query heads)."""
    h, hk = q.shape[1], k.shape[1]
    g = h // hk
    sq, sk = q.shape[2], k.shape[2]
    kd = k[0, 0].double().requires_grad_(True)
    vd = v[0, 0].double().requires_grad_(True)
    for i in range(g):
        s = (q[0, i].double() @ kd.T) * sm_scale
        if bias is not None:
            s = s + bias[0, i].double()
        if causal:
            r = torch.arange(sq, device='cuda')[:, None]
            c = torch.arange(sk, device='cuda')[None, :]
            s = s.masked_fill(c > r + (sk - sq), float('-inf'))
        o = torch.softmax(s, -1).nan_to_num() @ vd
        (o * do[0, i].double()).sum().backward()
    for name, got, ref in (('dK', dk[0, 0], kd.grad), ('dV', dv[0, 0], vd.grad)):
        err = (got.double() - ref).abs().max().item()
        print(f'{name}: max abs err {err:.3e} (ref max {ref.abs().max().item():.3e}) '
              f'nonfinite {(~torch.isfinite(got)).sum().item()}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--heads_k', type=int, default=None)
    p.add_argument('--seqlen', type=int, default=8192)
    p.add_argument('--seqlen_k', type=int, default=None)
    p.add_argument('--head_dim', type=int, default=192)
    p.add_argument('--causal', action='store_true', help='bottom-right causal (causal_type 2)')
    p.add_argument('--bias', action='store_true', help='(B, H, Sq, Sk) attention bias, N(0, 1)')
    p.add_argument('--dropout_p', type=float, default=0.0)
    p.add_argument('--kv_subs', type=int, default=None)
    p.add_argument('--body', choices=('wide', 'xwide', 'xxwide'), default=None, help='default: the tuning module\'s')
    p.add_argument('--grid_axis_order', type=int, default=None)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--no_check', action='store_true')
    a = p.parse_args()
    hk = a.heads_k or a.heads
    sk = a.seqlen_k or a.seqlen
    rung = next((r for r in LADDER if r >= a.head_dim), None)
    if rung is None:
        sys.exit(f'head_dim {a.head_dim} is past the gfx942 dK/dV ladder {LADDER}')
    kw = {k: v for k, v in (('kv_subs', a.kv_subs), ('grid_axis_order', a.grid_axis_order)) if v is not None}
    launch = build(rung,
                   body=a.body,
                   causal_type=2 if a.causal else 0,
                   bias=a.bias,
                   dropout=a.dropout_p > 0,
                   padded_head=rung != a.head_dim,
                   **kw)
    dt = torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(a.batch, a.heads, a.seqlen, a.head_dim, dtype=dt, device='cuda')
    do = torch.randn_like(q)
    k = torch.randn(a.batch, hk, sk, a.head_dim, dtype=dt, device='cuda')
    v = torch.randn_like(k)
    bias = torch.randn(a.batch, a.heads, a.seqlen, sk, dtype=dt, device='cuda') if a.bias else None
    sm_scale = a.head_dim ** -0.5
    lse, delta = side_inputs(q, k, v, do, sm_scale, a.causal, bias)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    extra = {}
    if a.dropout_p > 0:
        extra = dict(dropout_p=a.dropout_p, philox_seed=1234, philox_offset2=0)

    def run():
        launch(q,
               k,
               v,
               do,
               dk,
               dv,
               lse.view(-1, a.seqlen),
               delta.view(-1, a.seqlen),
               a.batch,
               a.seqlen,
               seqlen_k=sk,
               scale=sm_scale,
               bias=bias,
               stream=torch.cuda.current_stream(),
               **extra)

    run()
    torch.cuda.synchronize()
    if not a.no_check and a.dropout_p == 0:
        spot_check(q, k, v, do, dk, dv, sm_scale, a.causal, bias)
    for _ in range(3):
        run()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(a.iters):
        run()
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / a.iters
    pairs = a.seqlen * sk
    if a.causal:
        d = sk - a.seqlen
        pairs = sum(max(0, min(sk, i + d + 1)) for i in range(a.seqlen))
    flops = 8 * a.batch * a.heads * pairs * a.head_dim
    print(f'dK/dV B{a.batch} H{a.heads}/{hk} S{a.seqlen}/{sk} D{a.head_dim} (rung {rung}) causal={a.causal} '
          f'bias={a.bias} dropout={a.dropout_p}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOPS')


if __name__ == '__main__':
    main()
