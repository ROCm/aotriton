// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "check.h"
#include "iterations.h"

#include <limits>
#include <stdexcept>

using namespace perfmon;

int main() {
  CHECK_EQ(kMaxIters, 100);
  CHECK_EQ(kMinIters, 20);
  CHECK_EQ(kIterationBudgetSeconds, 10.0);

  // Calls of 100 ms or less keep every iteration.
  CHECK_EQ(adaptive_iterations(10.0), 100);
  CHECK_EQ(adaptive_iterations(100000.0), 100);
  // Above 100 ms: ceil(10 s / 102 ms) = 99.
  CHECK_EQ(adaptive_iterations(102000.0), 99);
  CHECK_EQ(adaptive_iterations(136000.0), 74);   // 73.5 rounds up
  CHECK_EQ(adaptive_iterations(380000.0), 27);
  CHECK_EQ(adaptive_iterations(500000.0), 20);
  CHECK_EQ(adaptive_iterations(3.2e6), 20);
  CHECK_EQ(adaptive_iterations(9.3e6), 20);
  // Not a usable reading: the default count.
  CHECK_EQ(adaptive_iterations(0.0), 100);
  CHECK_EQ(adaptive_iterations(-1.0), 100);
  CHECK_EQ(adaptive_iterations(std::numeric_limits<double>::quiet_NaN()), 100);
  CHECK_EQ(adaptive_iterations(std::numeric_limits<double>::infinity()), 100);

  CHECK_EQ(parse_iterations_override(nullptr), 0);
  CHECK_EQ(parse_iterations_override(""), 0);
  CHECK_EQ(parse_iterations_override("1"), 1);
  CHECK_EQ(parse_iterations_override("1000"), 1000);
  CHECK_EQ(parse_iterations_override("5000"), 5000);
  CHECK_THROWS(parse_iterations_override("5001"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override("0"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override("-5"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override("abc"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override("10x"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override(" 10"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override("1.5"), "PERFMON_ITERATIONS");
  CHECK_THROWS(parse_iterations_override("99999999999999"), "PERFMON_ITERATIONS");
  CHECK_MAIN_RETURN();
}
