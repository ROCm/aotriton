// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_CPP_TUNE_H
#define AOTRITON_V2_API_CPP_TUNE_H

#include <aotriton/config.h>

namespace AOTRITON_NS::v2 {

struct AOTRITON_API CppTune {
#if AOTRITON_BUILD_FOR_TUNING
  // TODO: Move them into a base class since they are common to all kernels
  int force_kernel_index = -1;
  int total_number_of_kernels = -1;
  const char* selected_kernel_psels = nullptr;
  const char* selected_kernel_copts = nullptr;
  // Fields to extract kernel image
  bool peek_kernel_image = false;  // Set true to examine the image without launching it
  const void* kernel_image = nullptr;
  size_t image_size = 0;
#endif
};

enum AOTRITON_API CppTuneSpecialKernelIndex : int {
  kDefault = -1,
  kSkipGPUCall = -2,
};

}

#endif
