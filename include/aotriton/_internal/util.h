// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_INTERNAL_UTIL_H
#define AOTRITON_V2_INTERNAL_UTIL_H

#include <aotriton/config.h>
#include <climits>
#include <cstdint>

namespace AOTRITON_NS {

inline uint32_t bit_width(uint32_t x) {
  if (x == 0)
    return 0;
  return (int)(CHAR_BIT * sizeof(x)) - __builtin_clz(x);
}

inline uint32_t bit_ceil(uint32_t x) {
  return 1 << bit_width(x - 1);
}

inline int32_t bit_ceil(int32_t x) {
  return 1 << bit_width(x - 1);
}

inline bool is_power_of_2(uint32_t x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

}

#endif
