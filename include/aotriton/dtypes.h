// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_DTYPES_H
#define AOTRITON_V2_API_DTYPES_H

#include <aotriton_config.h>
#include <stdint.h>

namespace AOTRITON_NS {

enum DType : int32_t {
  kUnknown = 0,
  kFloat32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
  kInt8 = 10,
  kInt16 = 11,
  kInt32 = 12,
  kInt64 = 13,
  kUInt8 = 20,
  kUInt16 = 21,
  kUInt32 = 22,
  kUInt64 = 23,
};

}

#endif
