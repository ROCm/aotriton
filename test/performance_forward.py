#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch

import triton
from attn_torch_function import attention

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

BATCH, N_HEADS, N_CTX, D_HEAD = 8, 64, 4096, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd']:
    # for causal in [False, True]:
    for causal in [False]:
        for D_HEAD in [64, 128]:
            configs.append(triton.testing.Benchmark(
                x_names=['N_CTX'],
                # x_vals=[2**i for i in range(10, 15)],
                x_vals=[2**13],
                line_arg='provider',
                line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []),
                styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
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
        sm_scale = 1.3
        autotune = False
        return_encoded_softmax = False
        fn = lambda: attention(q, k, v, causal, sm_scale, split_kernel, return_encoded_softmax, autotune)
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
    return total_flops / ms * 1e-9


# only works on post-Ampere GPUs right now
bench_flash_attention.run(save_path='.', print_data=True)
