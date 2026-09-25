#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Worst error/floor ratio and LSE margin of the pristine Triton kernels across
# seeds, for each test_common_mistakes case. test_common_mistakes allows 2.0x
# the floor and 1.0 of the LSE bound. See README.md next to this file.
#
#   TRITON_F32_DEFAULT=ieee python margins.py
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules' / 'flash' / 'kernel'))
import test_backward as tb
res = {}
for case in tb.COMMON_MISTAKES_CASES:
    for dtype in (torch.float16, torch.bfloat16):
        worst = {}
        for seed in range(6):
            seqlen_q, seqlen_k, sc, causal, sm = tb._common_mistakes_case(case)
            torch.manual_seed(seed)
            B, H, D = 2, 3, 128
            mk = lambda n, s: (torch.randn(B, H, n, D, device='cuda') * s).to(dtype)
            q, k, v, dout = mk(seqlen_q, sc).requires_grad_(), mk(seqlen_k, sc).requires_grad_(), mk(seqlen_k, 1.0).requires_grad_(), mk(seqlen_q, 1.0)
            ext = tb.AttentionExtraArgs(is_testing=False, fillnan=True, return_logsumexp=True)
            o, _, L = tb.attention(q, k, v, None, causal, sm, 0.0, ext)
            g = torch.autograd.grad(o, (q, k, v), dout)
            mask = tb._common_mistakes_mask(causal, seqlen_q, seqlen_k, 'cuda')
            live = mask.any(-1)
            ex = tb._common_mistakes_reference(q, k, v, dout, sm, mask)
            fl = [tb._common_mistakes_reference(q, k, v, dout, sm, mask, round_to=dtype, flush_subnormals=f) for f in (False, True)]
            smax = max(1.0, ex[1][..., live].abs().max().item())
            lse = (L.view(B, H, seqlen_q)[..., live].double() - ex[1][..., live]).abs().max().item() / (2.0 ** -16 * smax)
            worst['lse/bound'] = max(worst.get('lse/bound', 0), lse)
            for i, t in zip((0, 2, 3, 4), (o,) + g):
                if ex[i].norm() == 0:
                    continue
                r = lambda x: ((x.double() - ex[i]).norm() / ex[i].norm()).item()
                worst[['o','','dq','dk','dv'][i]] = max(worst.get(['o','','dq','dk','dv'][i], 0), r(t) / max(r(f[i]) for f in fl))
        print(case, dtype, ' '.join(f'{k}={v:.2f}' for k, v in worst.items()))
