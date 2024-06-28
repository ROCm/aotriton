#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch

import triton
from attn_torch_function_sbhd import attention

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

BATCH, N_HEADS, N_CTX, D_HEAD = 8, 16, 1024, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd']:
    # for causal in [False, True]:
    for causal in [True]:
        for D_HEAD in [64]:
            configs.append(triton.testing.Benchmark(
                x_names=['N_CTX'],
                # x_vals=[2**i for i in range(10, 15)],
                x_vals=[2**10],
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
                    'dtype': torch.bfloat16,
                    'mode': mode,
                    'causal': causal,
                    })
            )

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.bfloat16, device="cuda"):
    print(f"{N_CTX=}")
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    dropout_p = 0.1

    if provider == "triton":
        qkvdims = (N_CTX, BATCH, H, 3, D_HEAD)
        qkv = torch.randn(qkvdims, dtype=dtype, device="cuda", requires_grad=True)
        qkv = torch.permute(qkv, (1, 2, 0, 3, 4))
        q = qkv[:, :, :, 0, :]
        k = qkv[:, :, :, 1, :]
        v = qkv[:, :, :, 2, :]
        print("q.shape: ", q.shape, ", q.stride(): ", q.stride())
        print("k.shape: ", k.shape, ", k.stride(): ", k.stride())
        print("v.shape: ", v.shape, ", v.stride(): ", v.stride())
        sm_scale = 1.3
        autotune = False
        return_encoded_softmax = False
        b = None
        fn = lambda: attention(q, k, v, b, causal, sm_scale, dropout_p, return_encoded_softmax, autotune)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print("ms: ", ms)
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
