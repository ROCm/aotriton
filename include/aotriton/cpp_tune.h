// Copyright © 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_CPP_TUNE_H
#define AOTRITON_V2_API_CPP_TUNE_H

#include <aotriton/config.h>

namespace AOTRITON_NS::v2 {

struct CppTune {
#if AOTRITON_BUILD_FOR_TUNING
  // TODO: Move them into a base class since they are common to all kernels
  int force_kernel_index = -1;
  int total_number_of_kernels = -1;
  const char* selected_kernel_psels = nullptr;
  const char* selected_kernel_copts = nullptr;
#endif
};

enum CppTuneSpecialKernelIndex : int {
  kDefault = -1,
  kSkipGPUCall = -2,
};

}

#endif
