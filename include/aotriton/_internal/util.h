// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_INTERNAL_UTIL_H
#define AOTRITON_V2_INTERNAL_UTIL_H

#include <climits>
#include <cstdint>

namespace aotriton {

uint32_t bit_width(uint32_t x) {
  if (x == 0)
    return 0;
  return (int)(CHAR_BIT * sizeof(x)) - __builtin_clz(x);
}

uint32_t bit_ceil(uint32_t x) {
  return 1 << bit_width(x - 1);
}

}

#endif
