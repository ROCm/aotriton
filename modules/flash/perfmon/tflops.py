# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash attention TFLOPS, ported from `modules/flash/visperf/static/flash.js`
(`_attnValidElements`, `attnFwdTflops`, `attnBwdTflops`).

Only the two matmuls (QK^T and PV) count; softmax, logsumexp and dropout do
not, so a dropout table shows the cost of dropout as lower TFLOPS. `valid`
is the exact number of unmasked elements under causal masking. The backward
is the forward formula with the time divided by 2.5 (two backward matmuls
plus the recomputation).

The port is line for line; python/test/perfmon/test_flash_tflops.py fails if
the JavaScript formulas change.
"""


def attn_valid_elements(seqlen_q: int, seqlen_k: int, causal: bool) -> float:
    if causal:
        if seqlen_q <= seqlen_k:
            return seqlen_q * seqlen_k - (seqlen_q * seqlen_q - seqlen_q) / 2
        return (seqlen_k * seqlen_k + seqlen_k) / 2
    return seqlen_q * seqlen_k


def attn_fwd_tflops(seqlen_q, seqlen_k, hdim, causal, median_ms, batch, n_heads) -> float:
    valid = attn_valid_elements(seqlen_q, seqlen_k, causal)
    flops_per_matmul = 2 * valid * hdim
    total_flops = 2 * flops_per_matmul * batch * n_heads
    return total_flops / (median_ms * 1e-3) / 1e12


def attn_bwd_tflops(seqlen_q, seqlen_k, hdim, causal, median_ms, batch, n_heads) -> float:
    return attn_fwd_tflops(seqlen_q, seqlen_k, hdim, causal, median_ms / 2.5, batch, n_heads)
