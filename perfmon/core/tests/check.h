// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// A minimal check harness: no test framework is available without network.

#ifndef PERFMON_CORE_TESTS_CHECK_H
#define PERFMON_CORE_TESTS_CHECK_H

#include <cmath>
#include <cstdio>
#include <string>

namespace check {
inline int failures = 0;
}

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                   #cond);                                                 \
      ++check::failures;                                                   \
    }                                                                      \
  } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

// Checks that `expr` throws an exception whose what() contains `needle`.
#define CHECK_THROWS(expr, needle)                                          \
  do {                                                                      \
    bool _thrown = false;                                                   \
    try {                                                                   \
      (void)(expr);                                                         \
    } catch (const std::exception& _e) {                                    \
      _thrown = true;                                                       \
      if (std::string(_e.what()).find(needle) == std::string::npos) {       \
        std::fprintf(stderr, "%s:%d: %s threw \"%s\", expected \"%s\"\n",   \
                     __FILE__, __LINE__, #expr, _e.what(), needle);         \
        ++check::failures;                                                  \
      }                                                                     \
    }                                                                       \
    if (!_thrown) {                                                         \
      std::fprintf(stderr, "%s:%d: %s did not throw\n", __FILE__, __LINE__, \
                   #expr);                                                  \
      ++check::failures;                                                    \
    }                                                                       \
  } while (0)

#define CHECK_MAIN_RETURN()                                                 \
  do {                                                                      \
    if (check::failures) {                                                  \
      std::fprintf(stderr, "%d check(s) failed\n", check::failures);        \
      return 1;                                                             \
    }                                                                       \
    std::printf("all checks passed\n");                                     \
    return 0;                                                               \
  } while (0)

#endif  // PERFMON_CORE_TESTS_CHECK_H
