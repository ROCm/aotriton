#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shows that the fp16 dV excess of the Triton kernels over the rounding floor
# is the hardware flushing subnormal P, not a kernel mistake: a floor that
# flushes P below fp16's smallest normal reproduces the kernel's dV. See
# README.md next to this file ("Not mistakes: what the hardware does").
#
#   TRITON_F32_DEFAULT=ieee python ftz_check.py
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules' / 'flash' / 'kernel'))
from attn_torch_function import attention, AttentionExtraArgs

DTYPE = torch.float16

def relrms(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()

def dv_reference(q, k, dout, sm_scale, flush):
    q, k, dout = q.double(), k.double(), dout.double()
    p = torch.softmax(q @ k.transpose(-1, -2) * sm_scale, dim=-1)
    exact = p.transpose(-1, -2) @ dout
    p = p.to(DTYPE).double()
    if flush:
        p = torch.where(p.abs() < torch.finfo(DTYPE).tiny, 0.0, p)
    return (p.transpose(-1, -2) @ dout).to(DTYPE).double(), exact

def main():
    B, H, D, seqlen_q, seqlen_k = 2, 3, 128, 257, 519
    for scale in (1.0, 2.0, 3.0):
        torch.manual_seed(0)
        q = (torch.randn(B, H, seqlen_q, D, device='cuda') * scale).to(DTYPE).requires_grad_()
        k = (torch.randn(B, H, seqlen_k, D, device='cuda') * scale).to(DTYPE).requires_grad_()
        v = torch.randn(B, H, seqlen_k, D, device='cuda').to(DTYPE).requires_grad_()
        dout = torch.randn(B, H, seqlen_q, D, device='cuda').to(DTYPE)
        o, _, _ = attention(q, k, v, None, False, D ** -0.5, 0.0, AttentionExtraArgs())
        dv, = torch.autograd.grad(o, (v,), dout)
        floor, exact = dv_reference(q.detach(), k.detach(), dout, D ** -0.5, flush=False)
        floor_ftz, _ = dv_reference(q.detach(), k.detach(), dout, D ** -0.5, flush=True)
        print(f'input scale {scale}: kernel {relrms(dv, exact):.2e}  '
              f'floor {relrms(floor, exact):.2e}  flushed floor {relrms(floor_ftz, exact):.2e}  '
              f'kernel vs flushed floor {relrms(dv, floor_ftz):.2e}')

if __name__ == '__main__':
    main()
