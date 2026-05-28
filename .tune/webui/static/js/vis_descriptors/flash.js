// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Flash attention TFLOPS formulas.
// Source: ROCm/triton perf-kernels/flash-attention.py @ 0ec280cf, bench_flash_attention()
// Counts only the two matmuls (QK^T and PV); excludes softmax, LSE, dropout.
// BATCH=1, N_HEADS=1 for kernel-mode tuning (single-head benchmark).

function _attnValidElements(seqlen_q, seqlen_k, causal) {
  if (causal) {
    return seqlen_q <= seqlen_k
      ? seqlen_q * seqlen_k - (seqlen_q * seqlen_q - seqlen_q) / 2
      : (seqlen_k * seqlen_k + seqlen_k) / 2;
  }
  return seqlen_q * seqlen_k;
}

function attnFwdTflops(seqlen_q, seqlen_k, hdim, causal, median_ms, batch, n_heads) {
  const valid = _attnValidElements(seqlen_q, seqlen_k, causal);
  const flops_per_matmul = 2 * valid * hdim;
  const total_flops = 2 * flops_per_matmul * batch * n_heads;
  return total_flops / (median_ms * 1e-3) / 1e12;
}

function attnBwdTflops(seqlen_q, seqlen_k, hdim, causal, median_ms, batch, n_heads) {
  // backward = 2.5x forward FLOPs (2.0 bwd matmuls + 0.5 recompute)
  return attnFwdTflops(seqlen_q, seqlen_k, hdim, causal, median_ms / 2.5, batch, n_heads);
}

const FLASH_DESCRIPTOR = {
  id: 'flash',
  label: 'Flash Attention',

  dims: [
    { key: 'dtype',     label: 'dtype',   values: ['float16', 'bfloat16', 'float32'] },
    { key: 'hdim',      label: 'hdim',    values: [16, 32, 48, 64, 80, 96, 128, 160, 192, 256] },
    { key: 'causal',    label: 'causal',  values: [0, 1], valueLabels: ['F', 'T'] },
    { key: 'bias_type', label: 'bias',    values: [0, 1] },
    { key: 'dropout',   label: 'dropout', values: [0, 1] },
  ],

  matrixAxes: { row: 'seqlen_q', col: 'seqlen_k' },

  defaultColDims: ['dtype', 'bias_type', 'causal', 'dropout'],
  defaultRowDims: ['hdim'],
  defaultFixed:   {},

  ops: new Set(['attn_fwd_op', 'attn_bwd_op']),

  // kernel names that use the forward FLOPs formula
  fwdKernels: new Set(['attn_fwd', 'attn_fwd_op']),

  tflops(row) {
    const batch   = row.batch   || 1;
    const n_heads = row.n_heads || 1;
    const isBwd = !this.fwdKernels.has(row._kernel);
    if (isBwd) {
      return attnBwdTflops(row.seqlen_q, row.seqlen_k, row.hdim, row.causal, row.median_ms, batch, n_heads);
    }
    return attnFwdTflops(row.seqlen_q, row.seqlen_k, row.hdim, row.causal, row.median_ms, batch, n_heads);
  },

  tooltip(row) {
    const batch   = row.batch   || 1;
    const n_heads = row.n_heads || 1;
    const valid = _attnValidElements(row.seqlen_q, row.seqlen_k, row.causal);
    const fpm   = 2 * valid * row.hdim;
    const total = 2 * fpm * batch * n_heads;
    return [
      `${row.seqlen_q}×${row.seqlen_k} sq×sk, hdim=${row.hdim}, ${row.dtype}`,
      `causal=${row.causal ? 'T' : 'F'}, median=${row.median_ms.toFixed(3)} ms`,
      `─── FLOPs breakdown ───`,
      `BATCH=${batch}, N_HEADS=${n_heads}`,
      `valid elements: ${valid.toLocaleString()}`,
      `flops/matmul = 2×valid×hdim = ${fpm.toLocaleString()}`,
      `total = 2×matmuls×BATCH×N_HEADS = ${total.toLocaleString()}`,
      `TFLOPS = ${total.toLocaleString()}/(${row.median_ms.toFixed(4)}ms×1e-3)/1e12`,
    ];
  },
};
