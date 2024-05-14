// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#ifndef AOTRITON_USE_ZSTD
#error "Must define AOTRITON_USE_ZSTD explicitly. "\
       "Inconsistent definition of this macro causes ABI incompatibility."
#endif

#include "../runtime.h"
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <tuple>

namespace aotriton {

class TritonKernel {
public:
  TritonKernel(const void* image, size_t image_size, dim3 block, int shared_memory_size);

  hipError_t invoke(const char* kernel_name, dim3 grid, std::vector<void*>& args, hipStream_t stream);

#if AOTRITON_USE_ZSTD
  void clear_decompressed_image();
#endif
private:
  std::tuple<hipFunction_t, hipError_t> load_for_device(int device_id, const char* kernel_name);
  hipFunction_t cfind_function(int device_id) const;

  const void* kernel_image_ = nullptr;
  size_t image_size_ = 0;
  struct DeviceFunction {
    DeviceFunction(int device_id_,
                   hipModule_t mod_,
                   hipFunction_t func_);
    ~DeviceFunction();
    int device_id = -1;
    hipModule_t mod = nullptr;
    hipFunction_t func = nullptr; 
  };
  std::unordered_map<int, DeviceFunction> funcache_;
  std::shared_mutex mutex_;

  dim3 block_ { 256, 1, 1 };
  int shared_memory_size_;
#if AOTRITON_USE_ZSTD
  std::vector<char> decompressed_kernel_image_;
  void* decompress_kernel();
#endif
};

}

#endif
