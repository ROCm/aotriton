// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Summary statistics over per-iteration timing samples. Standard library
// only, so it is unit-tested without ROCm.

#ifndef PERFMON_CORE_STATS_H
#define PERFMON_CORE_STATS_H

#include <vector>

namespace perfmon {

struct Stats {
  int n = 0;
  double mean = 0.0;
  double median = 0.0;
  double stddev = 0.0;  // sample stddev (N-1 denominator)
  double min = 0.0;
  double p05 = 0.0;
  double p95 = 0.0;
};

// Linear-interpolated percentile of an already-ascending-sorted vector.
// `p` in [0, 1]. Requires `sorted_values` non-empty.
//
// Uses the "rank = p * (n - 1)" convention (the same one NumPy's default
// `numpy.percentile` interpolation uses): for n==1 always returns the sole
// element regardless of p.
double percentile(const std::vector<double>& sorted_values, double p);

// Computes n, mean, median, sample stddev (N-1), min, p05, p95 from
// `values` (order-independent; compute_stats sorts its own copy).
// Requires `values.size() >= 1`.
//
// stddev is 0.0 (not NaN/inf) when n == 1, since there are zero degrees of
// freedom for a sample variance -- documented here rather than left to
// whatever a 0/0 division happens to produce.
//
// Deterministic for a fixed input: no randomness, no unordered
// accumulation that could reorder floating-point sums between calls.
Stats compute_stats(std::vector<double> values);

}  // namespace perfmon

#endif  // PERFMON_CORE_STATS_H
