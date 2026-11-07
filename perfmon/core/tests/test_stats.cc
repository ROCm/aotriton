// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "check.h"
#include "stats.h"

#include <stdexcept>
#include <vector>

using namespace perfmon;

int main() {
  {
    Stats s = compute_stats({4.0, 1.0, 3.0, 2.0});
    CHECK_EQ(s.n, 4);
    CHECK_NEAR(s.mean, 2.5, 1e-12);
    CHECK_NEAR(s.median, 2.5, 1e-12);
    CHECK_NEAR(s.min, 1.0, 1e-12);
    // Sample stddev: sqrt(((1.5^2 + 0.5^2) * 2) / 3).
    CHECK_NEAR(s.stddev, 1.2909944487358056, 1e-12);
    // rank = p * (n - 1), linear interpolation.
    CHECK_NEAR(s.p05, 1.15, 1e-12);
    CHECK_NEAR(s.p95, 3.85, 1e-12);
  }
  {
    Stats s = compute_stats({7.0});
    CHECK_EQ(s.n, 1);
    CHECK_NEAR(s.median, 7.0, 0);
    CHECK_NEAR(s.stddev, 0.0, 0);
    CHECK_NEAR(s.p05, 7.0, 0);
    CHECK_NEAR(s.p95, 7.0, 0);
  }
  {
    std::vector<double> v;
    for (int i = 100; i >= 1; --i) v.push_back(i);
    Stats s = compute_stats(v);
    CHECK_EQ(s.n, 100);
    CHECK_NEAR(s.median, 50.5, 1e-12);
    CHECK_NEAR(s.mean, 50.5, 1e-12);
    CHECK_NEAR(s.p05, 5.95, 1e-12);
    CHECK_NEAR(s.p95, 95.05, 1e-12);
    // Order-independent: the same values shuffled give the same result.
    std::vector<double> w(v.rbegin(), v.rend());
    Stats t = compute_stats(w);
    CHECK_EQ(t.mean, s.mean);
    CHECK_EQ(t.stddev, s.stddev);
  }
  CHECK_NEAR(percentile({1.0, 2.0, 3.0}, 0.5), 2.0, 0);
  CHECK_THROWS(compute_stats({}), "non-empty");
  CHECK_THROWS(percentile({}, 0.5), "non-empty");
  CHECK_MAIN_RETURN();
}
