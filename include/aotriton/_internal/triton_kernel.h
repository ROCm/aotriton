// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#include <aotriton/config.h>
#include "../runtime.h"
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <tuple>

namespace AOTRITON_NS {

class PackedKernel;

class TritonKernel {
public:
  using Essentials = std::tuple<void*, int, dim3>;

  TritonKernel(const char* package_path, const char* stem_name);

  hipError_t invoke(const char* kernel_name, dim3 grid, std::vector<void*>& args, hipStream_t stream);

  void clear_decompressed_image();
private:
  std::tuple<hipFunction_t, hipError_t> load_for_device(int device_id, const char* kernel_name);
  hipFunction_t cfind_function(int device_id) const;

  const char* package_path_ = nullptr;
  const char* stem_name_ = nullptr;
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

  int shared_memory_size_ = 0;
  dim3 block_ { 256, 1, 1 };
  void* kernel_image_ = nullptr;
  Essentials decompress_kernel();
  std::shared_object<PackedKernel> packed_kernel_;
};

}

#endif
