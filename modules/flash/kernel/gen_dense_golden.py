#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capture dense (Varlen_bits == 0) kernel outputs for the bit-identity gate.

`test_forward.py` / `test_backward.py` compare against a *reference* with
tolerances, so they cannot see a bit-level change.  Gate 1 of the VarlenBits
port (execution plan section 7) requires that the dense path stay bit-identical
across the rewrite, which needs a golden captured from the pre-change kernel.

Run this once before touching the kernels:

    TRITON_F32_DEFAULT=ieee python gen_dense_golden.py

then `test_varlen_bits.py::test_dense_bit_identity` replays the same shapes and
compares with `torch.equal`.
"""

import os

import numpy as np
import torch

from attn_torch_function import attention, AttentionExtraArgs

# Beside this file, not beside the cwd: test_varlen_bits.py probes this path with
# os.path.exists(), so a bare relative name makes the bit-identity gate skip
# silently whenever pytest is invoked from anywhere but this directory.
GOLDEN_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'dense_golden.npz')

# (batch, num_head_q, num_head_k, seqlen_q, seqlen_k, d_head, causal, sm_scale)
GOLDEN_CASES = [
    (1, 1, 1, 64, 64, 16, False, 1.2),
    (2, 4, 4, 128, 128, 64, False, 1.2),
    (2, 4, 4, 128, 256, 64, True, 1.2),
    (1, 4, 2, 143, 203, 64, False, 0.5),   # GQA, ragged seqlens
    (2, 2, 2, 64, 63, 48, True, 1.2),      # non-power-of-two head dim
]

DTYPE = torch.float16


def run_case(case):
    batch, hq, hk, sq, sk, dhead, causal, sm_scale = case
    torch.manual_seed(20)
    dev = 'cuda'
    q = torch.randn((batch, hq, sq, dhead), device=dev, dtype=DTYPE)
    k = torch.randn((batch, hk, sk, dhead), device=dev, dtype=DTYPE)
    v = torch.randn((batch, hk, sk, dhead), device=dev, dtype=DTYPE)
    for t in (q, k, v):
        t.requires_grad_(True)
    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False)
    out, _, _ = attention(q, k, v, None, causal, sm_scale, 0.0, ext)
    dout = torch.randn_like(out)
    out.backward(dout)
    return {
        'out': out.detach(),
        'dq': q.grad,
        'dk': k.grad,
        'dv': v.grad,
    }


def case_key(case):
    return '_'.join(str(x) for x in case)


def collect():
    blob = {}
    for case in GOLDEN_CASES:
        for name, tensor in run_case(case).items():
            blob[f'{case_key(case)}.{name}'] = tensor.cpu().numpy()
    return blob


def main():
    blob = collect()
    np.savez(GOLDEN_NPZ, **blob)
    print(f'wrote {GOLDEN_NPZ} with {len(blob)} arrays')


if __name__ == '__main__':
    main()
