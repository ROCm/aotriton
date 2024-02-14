// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#ifndef AOTRITON_USE_ZSTD
#error "Must define AOTRITON_USE_ZSTD explicitly. "\
       "Inconsistent definition of this macro causes ABI incompatibility."
#endif

#include "../runtime.h"
#if AOTRITON_USE_ZSTD
#include <vector>
#endif

namespace aotriton {

class TritonKernel {
public:
  TritonKernel(const void* image, size_t image_size, dim3 block, int shared_memory_size);

  hipError_t invoke(const char* kernel_name, dim3 grid, std::vector<void*>& args, hipStream_t stream);

#if AOTRITON_USE_ZSTD
  void clear_decompressed_image();
#endif
private:
  const void* kernel_image_ = nullptr;
  size_t image_size_ = 0;
  dim3 block_ { 256, 1, 1 };
  hipModule_t mod_ = nullptr;
  hipFunction_t fun_ = nullptr;
  int shared_memory_size_;
#if AOTRITON_USE_ZSTD
  std::vector<char> decompressed_kernel_image_;
  void* decompress_kernel();
#endif
};

}

#endif
