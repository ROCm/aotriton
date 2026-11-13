# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Correctness and TFLOPS of the gfx942 backward dQ kernel against fp64.

TFLOPS count dQ's own work: S, dP and dQ, 3 * 2 * B * H * pairs * D.

    AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL tree> python bench_bwd_dq_gfx942.py \
        --batch 2 --heads 32 --seqlen 8192 --head_dim 128 --causal
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.environ['AOTRITON_FLYDSL_KERNEL_ROOT'])

import torch  # noqa: E402
import fmha_tuning_bwd_dq_gfx942 as tuning  # noqa: E402


def reference_rows(q, k, v, do, sm_scale, causal, bias, check):
    """(LSE, Delta) as the forward and bwd_preprocess would write them, plus fp64 dQ
    when `check`; [B, H, S, D] operands, one (batch, head) at a time."""
    b, h, sq, _ = q.shape
    sk = k.shape[2]
    lse = torch.empty(b, h, sq, dtype=torch.float32, device='cuda')
    delta = torch.empty_like(lse)
    dq = torch.empty(q.shape, dtype=torch.float64, device='cuda') if check else None
    db = torch.empty(b, h, sq, sk, dtype=torch.float64, device='cuda') if check and bias is not None else None
    hk = k.shape[1]
    for z in range(b):
        for i in range(h):
            j = i // (h // hk)
            qd, kd, vd, dod = (x.double() for x in (q[z, i], k[z, j], v[z, j], do[z, i]))
            s = (qd @ kd.T) * sm_scale
            if bias is not None:
                s = s + bias[z, i].double()
            if causal:
                r = torch.arange(sq, device='cuda')[:, None]
                c = torch.arange(sk, device='cuda')[None, :]
                s.masked_fill_(c > r + (sk - sq), float('-inf'))
            m = torch.logsumexp(s, dim=-1)
            p = torch.exp(s - m[:, None]).nan_to_num()
            o = (p @ vd).to(q.dtype).double()
            lse[z, i] = m.masked_fill(torch.isneginf(m), float('inf')).float()
            delta[z, i] = (dod * o).sum(-1).float()
            if check:
                ds = p * (dod @ vd.T - delta[z, i].double()[:, None])
                dq[z, i] = ds @ kd * sm_scale
                if db is not None:
                    db[z, i] = ds
            del s, p
    return lse, delta, dq, db


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--heads_k', type=int, default=None)
    p.add_argument('--seqlen', type=int, default=8192)
    p.add_argument('--seqlen_k', type=int, default=None)
    p.add_argument('--head_dim', type=int, default=128)
    p.add_argument('--causal', action='store_true')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'f16'])
    p.add_argument('--num_waves', type=int, default=None)
    p.add_argument('--waves_per_eu', type=int, default=None)
    p.add_argument('--block_n', type=int, default=None)
    p.add_argument('--grid_axis_order', type=int, default=None)
    p.add_argument('--lpt', type=int, default=None, help='1: longest q tiles first')
    p.add_argument('--bias', action='store_true', help='(B, H, Sq, Sk) attention bias, N(0, 1); writes dB')
    p.add_argument('--dropout', type=float, default=0.0, help='dropout_p; timing only, --check ignores it')
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--check', action='store_true', help='compare against fp64 (slow at long sequences)')
    a = p.parse_args()
    hk = a.heads_k or a.heads
    sk = a.seqlen_k or a.seqlen
    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    torch.manual_seed(0)
    # BSHD storage, passed as BHSD-shaped views.
    q = torch.randn(a.batch, a.seqlen, a.heads, a.head_dim, dtype=dt, device='cuda').transpose(1, 2)
    k = torch.randn(a.batch, sk, hk, a.head_dim, dtype=dt, device='cuda').transpose(1, 2)
    v = torch.randn(a.batch, sk, hk, a.head_dim, dtype=dt, device='cuda').transpose(1, 2)
    do = torch.randn(a.batch, a.seqlen, a.heads, a.head_dim, dtype=dt, device='cuda').transpose(1, 2)
    dq = torch.empty_like(do)
    bias = torch.randn(a.batch, a.heads, a.seqlen, sk, dtype=dt, device='cuda') if a.bias else None
    dbias = torch.empty_like(bias) if a.bias else None
    sm_scale = a.head_dim ** -0.5
    lse, delta, dq_ref, db_ref = reference_rows(q, k, v, do, sm_scale, a.causal, bias, a.check)
    knobs = {k: v for k, v in (('num_waves', a.num_waves),
                               ('waves_per_eu', a.waves_per_eu),
                               ('block_n', a.block_n),
                               ('grid_axis_order', a.grid_axis_order),
                               ('lpt', None if a.lpt is None else bool(a.lpt))) if v is not None}
    # Off-ladder head dims run on the next rung with padding, as dispatch does.
    rung = tuning.rung(a.head_dim, a.head_dim)
    if rung is None:
        sys.exit(f'head_dim {a.head_dim} is past the gfx942 dQ ladder {tuning.LADDER}')
    if rung != a.head_dim:
        knobs['padded_head'] = True
    # causal is bottom-right (causal_type 2), matching the reference.
    launch = tuning.build(rung, causal_type=2 if a.causal else 0, dtype_str=a.dtype, bias=a.bias, dropout=a.dropout > 0, **knobs)
    lse_2d = lse.view(a.batch * a.heads, a.seqlen)
    delta_2d = delta.view(a.batch * a.heads, a.seqlen)

    def run():
        launch(q,
               k,
               v,
               do,
               dq,
               lse_2d,
               delta_2d,
               a.batch,
               a.seqlen,
               seqlen_k=sk,
               scale=sm_scale,
               bias=bias,
               dbias=dbias,
               dropout_p=a.dropout if a.dropout > 0 else None,
               philox_seed=1234,
               stream=torch.cuda.current_stream())

    run()
    torch.cuda.synchronize()
    if a.check:
        d = dq.double() - dq_ref
        msg = (f'nonfinite {(~torch.isfinite(dq)).sum().item()}  max_abs_err {d.abs().max().item():.3e}  '
               f'rel_rms {(d.square().sum() / dq_ref.square().sum()).sqrt().item():.3e}  '
               f'ref_max {dq_ref.abs().max().item():.3e}')
        if db_ref is not None:
            msg += f'  dB max_abs_err {(dbias.double() - db_ref).abs().max().item():.3e}'
        print(msg)
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
        dd = sk - a.seqlen
        pairs = sum(max(0, min(sk, i + dd + 1)) for i in range(a.seqlen))
    flops = 6 * a.batch * a.heads * pairs * a.head_dim
    print(f'dQ B{a.batch} H{a.heads}/{hk} S{a.seqlen}/{sk} D{a.head_dim} causal={a.causal} '
          f'{a.dtype}: {ms:.3f} ms {flops / ms / 1e9:.1f} TFLOPS')


if __name__ == '__main__':
    main()
