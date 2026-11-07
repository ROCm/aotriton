// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "iterations.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace perfmon {

int adaptive_iterations(double warm_call_us) {
  if (!(warm_call_us > 0.0) || !std::isfinite(warm_call_us)) return kMaxIters;
  const double want = std::ceil(kIterationBudgetSeconds * 1e6 / warm_call_us);
  if (want >= kMaxIters) return kMaxIters;
  if (want <= kMinIters) return kMinIters;
  return static_cast<int>(want);
}

int parse_iterations_override(const char* value) {
  if (value == nullptr || *value == '\0') return 0;
  const std::string text(value);
  const std::string bad = "PERFMON_ITERATIONS must be a positive integer <= " +
                          std::to_string(kMaxIterationsOverride) + ", got '" + text + "'";
  if (text.size() > 6) throw std::invalid_argument(bad);
  long n = 0;
  for (char c : text) {
    if (c < '0' || c > '9') throw std::invalid_argument(bad);
    n = n * 10 + (c - '0');
  }
  if (n < 1 || n > kMaxIterationsOverride) throw std::invalid_argument(bad);
  return static_cast<int>(n);
}

}  // namespace perfmon
