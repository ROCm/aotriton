# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TFLOPS of the gfx942 dK/dV kernel (fmha_bwd_dkdv_gfx942.py) alone, and its error against fp64.

FLOPs are the kernel's four GEMMs, 8 * B * H * pairs * D, with pairs the
(q, kv) count the mask keeps. LSE and Delta come from torch.

    AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL tree> python bench_bwd_dkdv_gfx942.py \
        --batch 2 --heads 32 --seqlen 8192 --head_dim 128 --causal
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ['AOTRITON_FLYDSL_KERNEL_ROOT'])

import torch  # noqa: E402
import fmha_tuning_bwd_dkdv_gfx942 as tuning  # noqa: E402


def band(sq, sk, causal, device):
    """Bottom-right causal keep mask, or None."""
    if not causal:
        return None
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(sk, device=device)[None, :]
    return j <= i + (sk - sq)


def lse_delta(q, k, v, do, scale, keep):
    """Natural-base LSE (+inf on dead rows) and rowsum(dO * O) in fp32, one head at a time."""
    b, h, sq, _ = q.shape
    g = h // k.shape[1]
    lse = torch.empty(b, h, sq, device=q.device)
    delta = torch.empty_like(lse)
    for z in range(b):
        for hh in range(h):
            s = (q[z, hh].float() @ k[z, hh // g].float().T) * scale
            if keep is not None:
                s.masked_fill_(~keep, float('-inf'))
            m = torch.logsumexp(s, -1)
            o = (torch.softmax(s, -1).nan_to_num() @ v[z, hh // g].float()).to(q.dtype)
            lse[z, hh] = torch.where(torch.isneginf(m), torch.full_like(m, float('inf')), m)
            delta[z, hh] = (o.float() * do[z, hh].float()).sum(-1)
    return lse, delta


def reference(q, k, v, do, scale, keep):
    """fp64 dK, dV."""
    g = q.shape[1] // k.shape[1]
    kd = k.double().repeat_interleave(g, 1).requires_grad_()
    vd = v.double().repeat_interleave(g, 1).requires_grad_()
    s = q.double() @ kd.transpose(-1, -2) * scale
    if keep is not None:
        s = s.masked_fill(~keep, float('-inf'))
    o = torch.softmax(s, -1).nan_to_num() @ vd
    o.backward(do.double())
    b, hk, sk, d = k.shape
    return kd.grad.view(b, hk, g, sk, d).sum(2), vd.grad.view(b, hk, g, sk, d).sum(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--heads_k', type=int, default=None)
    p.add_argument('--seqlen', type=int, default=8192)
    p.add_argument('--seqlen_k', type=int, default=None)
    p.add_argument('--head_dim', type=int, default=128)
    p.add_argument('--causal', action='store_true', help='bottom-right causal')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'f16'])
    p.add_argument('--num_waves', type=int, default=None)
    p.add_argument('--kv_blocks', type=int, default=None)
    p.add_argument('--waves_per_eu', type=int, default=None)
    p.add_argument('--grid_axis_order', type=int, default=None)
    p.add_argument('--interleave_valu', type=int, default=None)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--check', action='store_true', help='compare with fp64 (small shapes only)')
    a = p.parse_args()
    hk = a.heads_k or a.heads
    sk = a.seqlen_k or a.seqlen
    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    rung = tuning.rung(a.head_dim, a.head_dim)
    if rung is None:
        sys.exit(f'head_dim {a.head_dim} is past the ladder {tuning.LADDER}')
    from fmha_bwd_dkdv_gfx942 import build_fmha_bwd_dkdv_gfx942

    knobs = {n: getattr(a, n) for n in ('num_waves', 'kv_blocks', 'waves_per_eu', 'grid_axis_order', 'interleave_valu')
             if getattr(a, n) is not None}
    launch = build_fmha_bwd_dkdv_gfx942(rung,
                                        causal_type=2 if a.causal else 0,
                                        dtype_str=a.dtype,
                                        padded_head=rung != a.head_dim,
                                        **knobs)
    torch.manual_seed(0)
    q = torch.randn(a.batch, a.heads, a.seqlen, a.head_dim, device='cuda', dtype=dt)
    k = torch.randn(a.batch, hk, sk, a.head_dim, device='cuda', dtype=dt)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    scale = a.head_dim ** -0.5
    keep = band(a.seqlen, sk, a.causal, 'cuda')
    lse, delta = lse_delta(q, k, v, do, scale, keep)
    lse2 = lse.view(a.batch * a.heads, a.seqlen)
    delta2 = delta.view(a.batch * a.heads, a.seqlen)
    dk = torch.full_like(k, float('nan'))
    dv = torch.full_like(v, float('nan'))

    def run():
        launch(q, k, v, do, dk, dv, lse2, delta2, a.batch, a.seqlen, seqlen_k=sk, scale=scale,
               stream=torch.cuda.current_stream())

    run()
    torch.cuda.synchronize()
    if a.check:
        rdk, rdv = reference(q, k, v, do, scale, keep)
        for name, got, ref in (('dK', dk, rdk), ('dV', dv, rdv)):
            d = got.double() - ref
            rel = (d.square().sum() / ref.square().sum()).sqrt().item()
            print(f'{name}: nonfinite {(~torch.isfinite(got)).sum().item()}  max_abs_err {d.abs().max().item():.3e}  '
                  f'rel_rms {rel:.3e}  bias {d.sum().item() / ref.abs().sum().item():+.2e}')
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
        pairs = sum(max(0, min(sk, i + sk - a.seqlen + 1)) for i in range(a.seqlen))
    flops = 8 * a.batch * a.heads * pairs * a.head_dim
    print(f'dkdv B{a.batch} H{a.heads}/{hk} S{a.seqlen}/{sk} D{a.head_dim} causal={a.causal} {a.dtype} {knobs}: '
          f'{ms:.3f} ms {flops / ms / 1e9:.1f} TFLOPS')


if __name__ == '__main__':
    main()
