// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_INTERNAL_UTIL_H
#define AOTRITON_V2_INTERNAL_UTIL_H

#include <aotriton/config.h>
#include <climits>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace AOTRITON_NS {

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
    static inline uint32_t bit_width(uint32_t x) {
      if (x == 0)
        return 0;
      unsigned long index;
      _BitScanReverse(&index, x);
      return index + 1;
    }
#else
    static inline uint32_t bit_width(uint32_t x) {
      if (x == 0)
        return 0;
      return (int)(CHAR_BIT * sizeof(x)) - __builtin_clz(x);
    }
#endif

inline uint32_t bit_ceil(uint32_t x) {
  return 1 << bit_width(x - 1);
}

inline int32_t bit_ceil(int32_t x) {
  return 1 << bit_width(x - 1);
}

inline bool is_power_of_2(uint32_t x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

inline int32_t round_value(int32_t value,
                           const std::vector<int32_t>& to_values,
                           int32_t oob_value = -1) {
  auto iter = std::lower_bound(to_values.begin(), to_values.end(), value);
  if (iter == to_values.end()) {
    return oob_value;
  }
  return *iter;
}

}

#endif
