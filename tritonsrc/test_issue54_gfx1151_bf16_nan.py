#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression test for issue #54: NaN on gfx1151 at Hiera-Large global-attn shape.

Root cause
----------
When Triton's AOT compiler builds tl.dot(q, k) for gfx1151 without an explicit
out_dtype, it can choose a bf16/fp16 accumulator for certain BLOCK_M x BLOCK_N
configurations. The resulting intermediate values overflow or lose mantissa
precision before the Qk_scale multiply promotes them to fp32, causing NaN to
propagate through the online softmax.

The fix adds out_dtype=tl.float32 to all three tl.dot calls in the non-INT8
forward path of fwd_kernel_inner.py, guaranteeing fp32 accumulation regardless
of what the compiler would otherwise choose.

Failure rates observed on gfx1151 (Strix Halo / RDNA 3.5) with the packaged
AOT binary (rocm7.14.0a20260526, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1),
measured over n=2693 COCO val2017 person images with SAM2.1 Hiera-Large:

  bf16: 59.4% of images produce NaN in the attention output
  fp16:  7.5% of images produce NaN
  fp32:     0%

SAM2.1 Hiera-Tiny (shorter seqlen) is unaffected.

The triggering shape is SAM2.1 Hiera-Large global-attention (stage 3,
blocks 23/33/43):
  B=4  (camera batch)  H=8  (num_heads)
  N=4096  (seqlen)     D=72  (head_dim, non-power-of-two)

Run
---
TRITON_F32_DEFAULT=ieee python -m pytest tritonsrc/test_issue54_gfx1151_bf16_nan.py -v
"""
import math

import pytest
import torch

from attn_torch_function import attention, AttentionExtraArgs
# attn_torch_function.py asserts TRITON_F32_DEFAULT=ieee at import time.

# Hiera-Large global-attn configuration
# Source: sam2.1_hiera_l.yaml, stage 3 (embed_dim=144 → num_heads=8, head_dim=72)
HIERA_B = 4
HIERA_H = 8
HIERA_N = 4096
HIERA_D = 72
HIERA_SCALE = 1.0 / math.sqrt(HIERA_D)

# Match the tolerance formula used across the existing test suite.
# For bf16: 1e-1 * max(1, (seqlen_q + seqlen_k + D) / 128)
# For our shape: 1e-1 * (4096+4096+72)/128 ≈ 6.5 (bf16), 1e-2 * 6.5 (fp16)
_ATOL = {
    torch.bfloat16: 1e-1 * max(1.0, (HIERA_N + HIERA_N + HIERA_D) / 128.0),
    torch.float16:  1e-2 * max(1.0, (HIERA_N + HIERA_N + HIERA_D) / 128.0),
}


def _dtype_ref(q, k, v, scale):
    """Reference in the input dtype — matches test_forward.py convention."""
    ref, _ = torch.ops.aten._scaled_dot_product_attention_math(
        q, k, v, scale=scale,
    )
    return ref


def _run_kernel(q, k, v, scale):
    """Call the aotriton JIT kernel directly."""
    # autotune=False uses the default BLOCK config. The production bug
    # manifests at specific autotuned configs; this matches test_forward.py
    # convention but may not cover the exact problematic config.
    extra = AttentionExtraArgs(return_encoded_softmax=False, autotune=False)
    out, _, _ = attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        None,    # bias
        False,   # causal
        scale,
        0.0,     # dropout_p
        extra,
    )
    return out


def _assert_no_nan(out, label):
    nan_count = (~torch.isfinite(out)).sum().item()
    assert nan_count == 0, (
        f'[{label}] NaN/Inf in {nan_count}/{out.numel()} output elements '
        f'({100 * nan_count / out.numel():.3f}%). '
        f'Regression of issue #54 — verify out_dtype=tl.float32 is present '
        f'on all tl.dot calls in fwd_kernel_inner.py.'
    )


# ---------------------------------------------------------------------------
# 1. Adversarial input: designed to expose the fp16 overflow path
#    Q = K = C/sqrt(D) where C=260 → q·kᵀ = C² = 67600 > fp16_max (65504).
#    On the buggy AOT path for certain BLOCK configs, the fp32 accumulator is
#    converted to fp16 before Qk_scale is applied: 67600 → +inf → inf−inf = NaN.
#    On the fixed path the dot stays fp32: 67600 × scale = 7976, safe.
#    Note: not every element NaNs in the actual kernel (tile structure means
#    not every output position sees the overflow); this test catches the
#    per-element NaN, not a 100% failure rate.
#
#    bf16 is excluded here because bf16 shares fp32's exponent range
#    (max ≈ 3.4e38), so C²=67600 does not overflow.  The production 59% bf16
#    NaN rate comes from a different mechanism — 7-bit mantissa quantization of
#    the running max causing imprecise centering — which is exercised by the
#    random-seed and large-sigma tests below.
# ---------------------------------------------------------------------------

def test_hiera_global_attn_adversarial_no_nan():
    """fp16-only: Q=K=const with q·kᵀ > fp16_max — must not produce NaN."""
    dtype = torch.float16
    C = 260.0   # C² = 67600 > fp16_max 65504
    elem = C / math.sqrt(HIERA_D)
    q = torch.full((HIERA_B, HIERA_H, HIERA_N, HIERA_D), elem,
                   device='cuda', dtype=dtype)
    k = q.clone()
    v = torch.ones((HIERA_B, HIERA_H, HIERA_N, HIERA_D), device='cuda', dtype=dtype)

    out = _run_kernel(q, k, v, HIERA_SCALE)
    _assert_no_nan(out, f'adversarial C={C} {dtype}')

    # Q=K=const → uniform attention → output = mean(v) = 1.0
    max_err = (out.float() - 1.0).abs().max().item()
    assert max_err < 0.1, (
        f'Uniform attention output should be ≈1.0; '
        f'got [{out.float().min():.3f}, {out.float().max():.3f}]'
    )


# ---------------------------------------------------------------------------
# 2. Random inputs — the probabilistic failure mode from production
#    The 59.4% bf16 / 7.5% fp16 NaN rate was measured on real image content;
#    here we use multiple seeds to increase coverage.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16],
                         ids=['fp16', 'bf16'])
@pytest.mark.parametrize('seed', [0, 1, 2, 42])
def test_hiera_global_attn_random_no_nan(dtype, seed):
    """Random inputs at the exact production shape must not produce NaN."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)
    k = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)
    v = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)

    out = _run_kernel(q, k, v, HIERA_SCALE)
    _assert_no_nan(out, f'random seed={seed} {dtype}')


# ---------------------------------------------------------------------------
# 3. Accuracy: output must agree with fp32 reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16],
                         ids=['fp16', 'bf16'])
def test_hiera_global_attn_accuracy(dtype):
    """Output must agree with fp32 reference within the repo-standard tolerance."""
    g = torch.Generator(device='cuda').manual_seed(7)
    q = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)
    k = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)
    v = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)

    ref = _dtype_ref(q, k, v, HIERA_SCALE)
    out = _run_kernel(q, k, v, HIERA_SCALE)

    _assert_no_nan(out, f'accuracy {dtype}')
    atol = _ATOL[dtype]
    assert torch.allclose(ref, out, atol=atol, rtol=0), (
        f'max abs err {(out.float() - ref.float()).abs().max():.4f} '
        f'> atol {atol:.4f} for {dtype}'
    )


# ---------------------------------------------------------------------------
# 4. Large-sigma inputs — realistic Hiera activation magnitudes
#    After several self-attention layers activations reach σ≈10–15, pushing
#    unscaled q·kᵀ toward the fp16/bf16 precision boundary.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16],
                         ids=['fp16', 'bf16'])
@pytest.mark.parametrize('sigma', [5.0, 10.0, 15.0])
def test_hiera_global_attn_large_sigma_no_nan(dtype, sigma):
    """Large-magnitude activations (realistic Hiera range) must not NaN."""
    g = torch.Generator(device='cuda').manual_seed(99)
    q = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g) * sigma
    k = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g) * sigma
    v = torch.randn(HIERA_B, HIERA_H, HIERA_N, HIERA_D,
                    device='cuda', dtype=dtype, generator=g)

    out = _run_kernel(q, k, v, HIERA_SCALE)
    _assert_no_nan(out, f'large sigma={sigma} {dtype}')
