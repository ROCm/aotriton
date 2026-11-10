# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TFLOPS of the gfx942 wide-head dQ kernels, with a fp64 check of one (batch, head).

LSE and O come from the gfx942 forward, Delta = rowsum(dO * O) in fp32. dQ FLOPs
are 3 * 2 * B * H * Sq * Sk * D, halved for causal.

    AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL tree> python bench_bwd_dq_wide_gfx942.py \
        --batch 2 --heads 32 --seqlen 8192 --head_dim 192 --causal
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ['AOTRITON_FLYDSL_KERNEL_ROOT'])

import torch  # noqa: E402
import fmha_tuning_gfx942 as fwd_tuning  # noqa: E402

LADDER = (160, 192, 224, 256, 384, 512)


def build_dq(rung, **kwargs):
    if rung <= 256:
        from fmha_bwd_dq_gfx942_wide import build_fmha_bwd_dq_gfx942_wide as build
    else:
        from fmha_bwd_dq_gfx942_xwide import build_fmha_bwd_dq_gfx942_xwide as build
    return build(rung, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--heads_k', type=int, default=None)
    p.add_argument('--seqlen', type=int, default=8192)
    p.add_argument('--seqlen_k', type=int, default=None)
    p.add_argument('--head_dim', type=int, default=192)
    p.add_argument('--causal', action='store_true')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'f16'])
    p.add_argument('--bias', action='store_true')
    p.add_argument('--dropout_p', type=float, default=0.0)
    p.add_argument('--grid_axis_order', type=int, default=None)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--no_check', action='store_true')
    a = p.parse_args()
    hk = a.heads_k or a.heads
    sk = a.seqlen_k or a.seqlen
    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    rung = next((r for r in LADDER if r >= a.head_dim), None)
    if rung is None:
        sys.exit(f'head_dim {a.head_dim} is past the ladder {LADDER}')
    padded = rung != a.head_dim
    torch.manual_seed(0)
    # BSHD storage, passed as BHSD views.
    q = torch.randn(a.batch, a.seqlen, a.heads, a.head_dim, dtype=dt, device='cuda')
    k = torch.randn(a.batch, sk, hk, a.head_dim, dtype=dt, device='cuda')
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    o = torch.empty_like(q)
    dq = torch.empty_like(q)
    lse = torch.empty(a.batch * a.heads, a.seqlen, dtype=torch.float32, device='cuda')
    bias = torch.randn(a.batch, a.heads, a.seqlen, sk, dtype=dt, device='cuda') if a.bias else None
    dbias = torch.empty_like(bias) if a.bias else None
    ct = 2 if a.causal else 0
    T = (lambda x: x.transpose(1, 2))
    fwd = fwd_tuning.build(rung, causal_type=ct, dtype_str=a.dtype, bias=a.bias, padded_head=padded,
                           dropout=a.dropout_p > 0)
    drop = dict(dropout_p=a.dropout_p, philox_seed=1234, philox_offset2=5) if a.dropout_p > 0 else {}
    fwd(T(q), T(k), T(v), T(o), a.batch, a.seqlen, seqlen_k=sk, lse=lse, bias=bias, **drop)
    delta = (do.float() * o.float()).sum(-1).transpose(1, 2).reshape(a.batch * a.heads, a.seqlen).contiguous()
    kw = dict(causal_type=ct,
              dtype_str=a.dtype,
              bias=a.bias,
              dropout=a.dropout_p > 0,
              padded_head=padded)
    if a.grid_axis_order is not None:
        kw['grid_axis_order'] = a.grid_axis_order
    launch = build_dq(rung, **kw)

    def run():
        launch(T(q), T(k), T(v), T(do), T(dq), lse, delta, a.batch, a.seqlen, seqlen_k=sk, bias=bias, dbias=dbias,
               stream=torch.cuda.current_stream(), **drop)

    run()
    torch.cuda.synchronize()
    if not a.no_check and a.dropout_p == 0:
        # fp64 on (batch 0, head 0).
        qd, kd, vd, dod = (x[0, :, 0].double() for x in (q, k, v, do))
        s = (qd @ kd.T) * a.head_dim ** -0.5
        if bias is not None:
            s = s + bias[0, 0].double()
        if a.causal:
            i = torch.arange(a.seqlen, device='cuda')[:, None]
            j = torch.arange(sk, device='cuda')[None, :]
            s.masked_fill_(j > i + (sk - a.seqlen), float('-inf'))
        pr = torch.softmax(s, -1).nan_to_num()
        dp = dod @ vd.T
        dlt = (dod * (pr @ vd)).sum(-1, keepdim=True)
        ref = (pr * (dp - dlt)) @ kd * a.head_dim ** -0.5
        got = dq[0, :, 0].double()
        err = (got - ref).abs().max().item() / ref.abs().max().item()
        print(f'check (b0, h0): max err {err:.2e} of max |dQ|, nan={bool(torch.isnan(got).any())}')
    for _ in range(3):
        run()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(a.iters):
        run()
    e1.record()
    torch.cuda.synchronize()
    t = e0.elapsed_time(e1) / a.iters
    fl = 6 * a.batch * a.heads * a.seqlen * sk * a.head_dim * (0.5 if a.causal else 1)
    print(f'dq B{a.batch} H{a.heads} S{a.seqlen} Sk{sk} D{a.head_dim} (rung {rung}) causal={a.causal} '
          f'bias={a.bias} p={a.dropout_p}: {t:.3f} ms {fl / t / 1e9:.1f} TFLOPS')


if __name__ == '__main__':
    main()
