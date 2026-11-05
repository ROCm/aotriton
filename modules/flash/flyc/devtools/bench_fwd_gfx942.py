# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness and TFLOPS of fmha_fwd_gfx942 against torch SDPA.

    AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL tree> python bench_fwd_gfx942.py \
        --batch 4 --heads 32 --seqlen 8192 --head_dim 128 --causal
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ['AOTRITON_FLYDSL_KERNEL_ROOT'])

import torch  # noqa: E402
from fmha_fwd_gfx942 import build_fmha_fwd_gfx942  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--heads_k', type=int, default=None)
    p.add_argument('--seqlen', type=int, default=8192)
    p.add_argument('--seqlen_k', type=int, default=None)
    p.add_argument('--head_dim', type=int, default=128)
    p.add_argument('--causal', action='store_true')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'f16'])
    p.add_argument('--num_waves', type=int, default=8)
    p.add_argument('--waves_per_eu', type=int, default=2)
    p.add_argument('--block_n', type=int, default=None)
    p.add_argument('--rescale_threshold', type=float, default=0.0)
    p.add_argument('--grid_axis_order', type=int, default=None)
    p.add_argument('--bias', action='store_true', help='(B, H, Sq, Sk) attention bias, N(0, 1)')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--no_check', action='store_true')
    p.add_argument('--dist', default='normal', choices=['normal', 'uniform', 'sharp'],
                   help='normal: N(0,1); uniform: U[0,1) (one-signed, exposes P rounding bias); '
                        'sharp: N(0,16) Q/K (large logits)')
    p.add_argument('--sm_scale', type=float, default=None)
    a = p.parse_args()
    hk = a.heads_k or a.heads
    sk = a.seqlen_k or a.seqlen
    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    torch.manual_seed(0)
    def gen(*shape, qk=False):
        if a.dist == 'uniform':
            return torch.rand(*shape, dtype=dt, device='cuda')
        x = torch.randn(*shape, dtype=dt, device='cuda')
        return x * 4 if a.dist == 'sharp' and qk else x
    q = gen(a.batch, a.seqlen, a.heads, a.head_dim, qk=True)
    k = gen(a.batch, sk, hk, a.head_dim, qk=True)
    v = gen(a.batch, sk, hk, a.head_dim)
    sm_scale = a.head_dim ** -0.5 if a.sm_scale is None else a.sm_scale
    o = torch.empty_like(q)
    lse = torch.empty(a.batch, a.heads, a.seqlen, dtype=torch.float32, device='cuda')
    # causal is bottom-right (causal_type 2), matching the reference below.
    launch = build_fmha_fwd_gfx942(a.head_dim,
                                   causal_type=2 if a.causal else 0,
                                   dtype_str=a.dtype,
                                   num_waves=a.num_waves,
                                   block_n=a.block_n,
                                   rescale_threshold=a.rescale_threshold,
                                   grid_axis_order=a.grid_axis_order,
                                   bias=a.bias,
                                   waves_per_eu=a.waves_per_eu)
    lse_2d = lse.view(a.batch * a.heads, a.seqlen)
    bias = torch.randn(a.batch, a.heads, a.seqlen, sk, dtype=dt, device='cuda') if a.bias else None

    def run():
        # BSHD storage, passed as BHSD-shaped views.
        launch(q.transpose(1, 2),
               k.transpose(1, 2),
               v.transpose(1, 2),
               o.transpose(1, 2),
               a.batch,
               a.seqlen,
               seqlen_k=sk,
               scale=sm_scale,
               lse=lse_2d,
               bias=bias,
               stream=torch.cuda.current_stream())

    run()
    torch.cuda.synchronize()
    if not a.no_check:
        # fp64 reference from the same bf16/f16 inputs; the checks follow
        # docs/attention-kernel-numerical-error-lessons.
        qt, kt, vt = (x.transpose(1, 2).double() for x in (q, k, v))
        if hk != a.heads:
            kt = kt.repeat_interleave(a.heads // hk, dim=1)
            vt = vt.repeat_interleave(a.heads // hk, dim=1)
        nb = max(1, (1 << 26) // (a.heads * a.seqlen * sk))
        err = 0.0
        sq_err = sq_ref = signed = mag = 0.0
        lse_err = smax = 0.0
        for b0 in range(0, a.batch, nb):
            sl = slice(b0, b0 + nb)
            s = (qt[sl] @ kt[sl].transpose(-1, -2)) * sm_scale
            if bias is not None:
                s = s + bias[sl].double()
            if a.causal:
                i = torch.arange(a.seqlen, device='cuda')[:, None]
                j = torch.arange(sk, device='cuda')[None, :]
                s.masked_fill_(j > i + (sk - a.seqlen), float('-inf'))
            ref = torch.softmax(s, dim=-1).nan_to_num() @ vt[sl]
            ref_lse = torch.logsumexp(s, dim=-1)
            got = o[sl].transpose(1, 2).double()
            d = got - ref
            err = max(err, d.abs().max().item())
            sq_err += (d ** 2).sum().item()
            sq_ref += (ref ** 2).sum().item()
            signed += d.sum().item()
            mag += ref.abs().sum().item()
            live = torch.isfinite(ref_lse)
            lse_err = max(lse_err, (lse[sl].double() - ref_lse)[live].abs().max().item())
            smax = max(smax, s[torch.isfinite(s)].abs().max().item())
            del s, ref, d
        nonfinite = (~torch.isfinite(o)).sum().item()
        print(f'nonfinite {nonfinite}  max_abs_err {err:.3e}  rel_rms {(sq_err / sq_ref) ** 0.5:.3e}  '
              f'bias {signed / mag:+.2e} (bf16 bound {2 ** -10:.1e})  '
              f'lse_err {lse_err:.2e} (bound {2 ** -16 * smax:.2e})')
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
    flops = 4 * a.batch * a.heads * pairs * a.head_dim
    print(f'B{a.batch} H{a.heads}/{hk} S{a.seqlen}/{sk} D{a.head_dim} causal={a.causal} '
          f'{a.dtype}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOPS')


if __name__ == '__main__':
    main()
