#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch

import os
import triton
from attn_torch_function import attention, AttentionExtraArgs, AOTRITON_USE_PRINT_AUTOTUNING

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None
USE_TFLOPS = bool(int(os.getenv('USE_TFLOPS', default='1')))
print(f'{USE_TFLOPS=}')

n_ctx = os.getenv('N_CTX', default=list(range(10, 14)))
if isinstance(n_ctx, str):
    n_ctx = map(lambda x: int(x), n_ctx.split(','))
X_VALS = list(map(lambda x: 2 ** x, n_ctx))
print(f'{X_VALS=}')

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64

D_HEAD = int(os.getenv('D_HEAD', default=D_HEAD))
BATCH = int(os.getenv('BATCH', default=BATCH))
N_HEADS = int(os.getenv('N_HEADS', default=N_HEADS))
# vary seq length for fixed head and batch=4
USE_CAUSAL = bool(int(os.getenv('USE_CAUSAL', default='1')))
ALL_CAUSALS = [False, True] if USE_CAUSAL else [False]
print(f'{ALL_CAUSALS=}')

USE_AUTOTUNE = bool(int(os.getenv('USE_AUTOTUNE', default='1')))
print(f'{USE_AUTOTUNE=}')

configs = []
for mode in ['fwd']:
    for causal in ALL_CAUSALS:
        configs.append(triton.testing.Benchmark(
            x_names=['N_CTX'],
            x_vals=list(X_VALS),
            # x_vals=[2**i for i in range(10, 14)],  # 2 ** 15 not working for now
            # x_vals=[2**12],
            line_arg='provider',
            line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
            line_names=['Triton(TFLOPS)' if USE_TFLOPS else 'Triton(ms)'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
            styles=[('red', '-'), ('blue', '-')],
            ylabel='TFLOPS' if USE_TFLOPS else 'ms',
            plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}',
            args={
                'H': N_HEADS,
                'BATCH': BATCH,
                'D_HEAD': D_HEAD,
                'dtype': torch.float16,
                'mode': mode,
                'causal': causal,
                })
        )


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    print(f"{N_CTX=}")
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    split_kernel = False
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        split_kernel = True if causal else split_kernel
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        b = None
        sm_scale = 1.3
        dropout_p = 0.0
        ext = AttentionExtraArgs(return_encoded_softmax=False,
                                 autotune=USE_AUTOTUNE,
                                 return_autotune=False)
        fn = lambda: attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        if FLASH_VER == 1:
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            fn = lambda: flash_attn_func(qkv, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2. * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    if USE_TFLOPS:
        return total_flops / ms * 1e-9
    else:
        return ms


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path='.', print_data=True)
