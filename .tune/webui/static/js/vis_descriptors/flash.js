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

  // ---------------------------------------------------------------------------
  // Level-2 (psel/copt) drilldown schema.
  // Source: v3python/rules/flash/*.py gen_autotune_configs().
  // Listed explicitly — do not infer field names from DB.
  // ---------------------------------------------------------------------------
  cellDetail: {
    // Kernels that support level-2 drilldown.
    // Ops use a single backend_index so a psel×copt matrix is degenerate.
    kernels: {
      attn_fwd: {
        psels: ['PERSISTENT_TYPE', 'GRID_CU_MULTIP',
                'BLOCK_M', 'BLOCK_N', 'PRE_LOAD_V', 'NUM_XCDS'],
        copts: ['num_warps', 'num_stages', 'waves_per_eu'],
      },
      bwd_kernel_dk_dv: {
        psels: ['BLOCK_M', 'BLOCK_N', 'NUM_XCDS'],
        copts: ['num_warps', 'num_stages', 'waves_per_eu'],
      },
      bwd_kernel_dq: {
        psels: ['BLOCK_M', 'BLOCK_N', 'NUM_XCDS'],
        copts: ['num_warps', 'num_stages', 'waves_per_eu'],
      },
      bwd_kernel_fuse: {
        psels: ['BLOCK_M', 'BLOCK_N', 'NUM_XCDS'],
        copts: ['num_warps', 'num_stages', 'waves_per_eu'],
      },
    },

    // Skipped test cases — must match v3python/tune/pq/compute_best_results.SKIP_TEST_CASES.
    skipTestCases: new Set([]),

    // Accuracy gate: must match compute_best_results.ACCURACY_MULTIPLIER.
    accuracyMultiplier: 10.0,

    // TFLOPS for one psel/copt candidate, given its median_ms and the
    // level-1 row context (so we know seqlen_q/seqlen_k/hdim/causal/etc).
    candidateTflops(cand, cellRow) {
      if (!cand || !(cand.median_ms > 0)) return 0;
      const synth = Object.assign({}, cellRow, { median_ms: cand.median_ms });
      return FLASH_DESCRIPTOR.tflops(synth);
    },
  },

  // Audit one candidate against the per-(test_case, tensor) thresholds.
  // Mirrors v3python/tune/pq/compute_best_results.evaluate_group's per-row
  // gating (kernel mode; op-mode early-reject not used here).
  //   adiffs[tc][tensor] = [target_fudge, abs_err, ref_err]
  //   threshold[tc][tensor] = absolute_error  (from most_accurate_*)
  // Returns true if the candidate passes the accuracy gate.
  passesAccuracy(cand, threshold) {
    const skip = FLASH_DESCRIPTOR.cellDetail.skipTestCases;
    const mult = FLASH_DESCRIPTOR.cellDetail.accuracyMultiplier;
    const adiffs = cand.adiffs || {};
    for (const [tc, tensors] of Object.entries(adiffs)) {
      if (skip.has(tc)) continue;
      for (const [tname, vals] of Object.entries(tensors)) {
        if (!vals || !vals.length) continue;                  // inapplicable
        const ref_err = vals.length > 2 ? vals[2] : null;
        if (ref_err === null || ref_err === undefined) continue;
        const abs_err = vals.length > 1 ? vals[1] : null;
        if (abs_err === null || abs_err === undefined) return false;  // broken
        const min_err = threshold[tc] && threshold[tc][tname];
        if (min_err !== undefined && min_err !== null
            && abs_err > mult * min_err) {
          return false;
        }
      }
    }
    return true;
  },

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
