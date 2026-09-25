#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`bwd_kernel_fuse` at every tile/warp config the AOT build compiles.

**A Triton codegen bug, and what this test really checks is the toolchain.**
At `BLOCK_M=64, BLOCK_N=16, num_warps=2` on wave64, `bwd_kernel_fuse` returns a
wrong dK and dV -- by four orders of magnitude, not by a tolerance. Every other
cell `modules/flash/aot/bwd_kernel_fuse.py:gen_autotune_configs` compiles is
clean:

    BLOCK_M  BLOCK_N  warps   dK        dV     (relative to the 16/16/4 tile)
      any except below         3.8e-4    5.5e-4
       64       16      2      3.2e+1    5.8e+1   <-- this test

It is reached in production whenever the build used 3.7.0: the shipped
`bwd_kernel_fuse.zip` for
`Q='*fp16:16';BLOCK_DMODEL=64;CAUSAL_TYPE=0;ENABLE_DROPOUT=False;PADDED_HEAD=
False;BIAS_TYPE=0` packs it three times (waves_per_eu 1, 2 and 3) and the
tuning database selects the third, which is what makes
`test_transformers.py::test_fused_attention_vs_math_ref_grads_cudagraph` fail
at batch 8, seq_len_q 1024, seq_len_k 256, head_dim 64, fp16.

**Which Triton built the kernel is the whole story:**

    triton-3.7.0+gitdb82b800.aotriton0.14    fails
    triton-3.8.0+git4cff872c.rocm10.0.0      fails
    triton-3.8.0+gitaa3cf1a1.aotriton0.14    passes

so the fix is to build with the `aa3cf1a1` wheel, not to drop the config. The
version number is not what decides it -- two 3.8.0 builds disagree, and which
commit repairs it, or whether it is one of aotriton's patches rather than an
upstream commit, is not established here. A JIT venv that happens to carry
some other 3.8.0 still reproduces.

It is not arithmetic: the error does not depend on `waves_per_eu` or
`num_stages`, predates the qk-ordering change in the inner kernels (the source
at `5de6fa2f~1` fails identically), needs `BLOCK_M=64` and `num_warps=2`
together with `BLOCK_N=16` -- each is clean in every other pairing -- and
varies run to run, 23 to 39 on one shape. Same family as ROCm/aotriton#54,
which `fwd_kernel_inner.py` already avoids by refusing `composed_dot_both`.

Shape is the reporting UT's. Inputs are `torch.rand` like the UT's, not
`randn`: the distribution is not load-bearing here -- the failure is four
orders of magnitude -- but keeping it identical removes a variable.

    BWD_FUSED=1 TRITON_F32_DEFAULT=ieee \\
        python -m pytest modules/flash/kernel/test_bwd_fuse_tile_config.py
"""

import itertools
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not bool(int(os.getenv("BWD_FUSED", default="0"))) or not torch.cuda.is_available(),
    reason="needs BWD_FUSED=1 (this kernel is only reachable through it) and a GPU",
)

BATCH, N_HEADS, SEQLEN_Q, SEQLEN_K, D_HEAD = 8, 4, 1024, 256, 64
DTYPE = torch.float16

# `gen_autotune_configs`' wave64 space, minus the two triples it already drops.
# `waves_per_eu` is not swept: it does not move the result, and sweeping it
# would quadruple a test that compiles a kernel per case.
BLOCK_SIZES = (16, 32, 64)
NUM_WARPS = (2, 4)

# Every tile computes the same math, so the tiles must agree with EACH OTHER
# to fp16 rounding. That is the invariant worth asserting here, and it needs no
# tolerance guessing: an absolute reference would have to carry a per-gradient
# fudge (dQ runs ~40x the fp16 math reference's own error on this shape in
# every config, including the good ones) and would then be measuring fp16
# rounding rather than the miscompile. Agreement separates cleanly: the good
# tiles sit at fp16 rounding, the broken one is off by 32x the gradient's own
# magnitude. Relative to max|baseline| so the bound travels to other shapes:
#
#     every other cell   dQ 5.1e-4   dK 3.8e-4   dV 5.5e-4
#     M64-N16-W2         dQ 5.1e-4   dK 3.2e+1   dV 5.8e+1
#
# 1e-2 is 18x above the noise and 3000x below the failure.
BASELINE = (16, 16, 4)
RTOL = 1e-2


def _cases():
    for block_m, block_n, warps in itertools.product(BLOCK_SIZES, BLOCK_SIZES, NUM_WARPS):
        if block_m < block_n:
            continue  # `gen_autotune_configs` calls this a duplicate
        if block_m == 64 and block_n == 64 and warps == 4:
            continue  # already dropped there
        yield pytest.param(block_m, block_n, warps, id=f"M{block_m}-N{block_n}-W{warps}")


def _grads(block_m, block_n, num_warps):
    """dQ, dK, dV from one tile config, on the reporting UT's shape."""
    from attn_torch_function import AttentionExtraArgs, attention

    # Read by `backward_fused` at launch, so setting them here is what picks
    # the config -- there is no argument for it on the public entry point.
    os.environ["BWD_FUSE_BLOCK_M"] = str(block_m)
    os.environ["BWD_FUSE_BLOCK_N"] = str(block_n)
    os.environ["BWD_FUSE_NUM_WARPS"] = str(num_warps)
    os.environ["BWD_FUSE_NUM_STAGES"] = "1"

    scale = 1.0 / math.sqrt(D_HEAD)
    torch.manual_seed(42)
    q = torch.rand((BATCH, N_HEADS, SEQLEN_Q, D_HEAD), device="cuda", dtype=DTYPE, requires_grad=True)
    k = torch.rand((BATCH, N_HEADS, SEQLEN_K, D_HEAD), device="cuda", dtype=DTYPE, requires_grad=True)
    v = torch.rand((BATCH, N_HEADS, SEQLEN_K, D_HEAD), device="cuda", dtype=DTYPE, requires_grad=True)

    ext = AttentionExtraArgs(return_encoded_softmax=False, autotune=False, return_autotune=False)
    out, _, _ = attention(q, k, v, None, False, scale, 0.0, ext)
    torch.manual_seed(43)
    upstream = torch.rand_like(out)
    return [g.float() for g in torch.autograd.grad(out, (q, k, v), upstream)]


@pytest.mark.parametrize("block_m,block_n,num_warps", list(_cases()))
def test_every_compiled_tile_agrees(block_m, block_n, num_warps):
    want = _grads(*BASELINE)
    got = _grads(block_m, block_n, num_warps)
    for name, g, w in zip(("dQ", "dK", "dV"), got, want):
        err = (g - w).abs().max().item() / w.abs().max().item()
        assert err <= RTOL, (
            f"BLOCK_M={block_m} BLOCK_N={block_n} num_warps={num_warps}: {name} differs "
            f"from the BLOCK_M={BASELINE[0]} BLOCK_N={BASELINE[1]} num_warps={BASELINE[2]} "
            f"baseline by {err:.3g} relative. Every tile computes the same math, so this is a "
            f"miscompile; the tile is packed into the shipped bwd_kernel_fuse.zip. "
            f"See this file's docstring"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
