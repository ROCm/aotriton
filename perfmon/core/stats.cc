// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace perfmon {

double percentile(const std::vector<double>& sorted_values, double p) {
  if (sorted_values.empty()) {
    throw std::invalid_argument("percentile: sorted_values must be non-empty");
  }
  const size_t n = sorted_values.size();
  if (n == 1) {
    return sorted_values[0];
  }
  const double rank = p * static_cast<double>(n - 1);
  const size_t lo = static_cast<size_t>(std::floor(rank));
  const size_t hi = static_cast<size_t>(std::ceil(rank));
  if (lo == hi) {
    return sorted_values[lo];
  }
  const double frac = rank - static_cast<double>(lo);
  return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]);
}

Stats compute_stats(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("compute_stats: values must be non-empty");
  }
  std::sort(values.begin(), values.end());

  Stats s;
  s.n = static_cast<int>(values.size());
  s.min = values.front();

  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  s.mean = sum / static_cast<double>(s.n);

  if (s.n % 2 == 1) {
    s.median = values[s.n / 2];
  } else {
    s.median = 0.5 * (values[s.n / 2 - 1] + values[s.n / 2]);
  }

  if (s.n == 1) {
    s.stddev = 0.0;  // zero degrees of freedom -- see stats.h
  } else {
    double sq_diff_sum = 0.0;
    for (double v : values) {
      const double d = v - s.mean;
      sq_diff_sum += d * d;
    }
    s.stddev = std::sqrt(sq_diff_sum / static_cast<double>(s.n - 1));
  }

  s.p05 = percentile(values, 0.05);
  s.p95 = percentile(values, 0.95);

  return s;
}

}  // namespace perfmon
