// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#include "../runtime.h"
#include <aotriton/config.h>
#include <memory>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace AOTRITON_NS {

class PackedKernel;

class TritonKernel {
public:
  struct Essentials {
    const void* image = nullptr;
    size_t size = 0;
    int shared_memory_size = 0;
    dim3 block { 0, 0, 0 };
  };

  TritonKernel(const char* package_path, const char* stem_name);

  hipError_t invoke(const char* kernel_name,
                    dim3 grid,
                    std::vector<void*>& args,
#if AOTRITON_BUILD_FOR_TUNING
                    bool peek_kernel_image,
#endif
                    hipStream_t stream);

  void clear_decompressed_image();

#if AOTRITON_BUILD_FOR_TUNING
  // Will not work unless invoke is called at least once, i.e., If-and-only-iF decompressed
  Essentials get_image_info_iff_decompressed() const;
#endif
private:
  std::tuple<hipFunction_t, hipError_t> load_for_device(int device_id, const char* kernel_name);
  hipFunction_t cfind_function(int device_id) const;

  const char* package_path_ = nullptr;
  const char* stem_name_ = nullptr;
  size_t image_size_ = 0;
  struct DeviceFunction {
    DeviceFunction(int device_id_, hipModule_t mod_, hipFunction_t func_);
    ~DeviceFunction();
    int device_id = -1;
    hipModule_t mod = nullptr;
    hipFunction_t func = nullptr;
  };
  std::unordered_map<int, DeviceFunction> funcache_;
  std::shared_mutex funcache_mutex_;

  Essentials essentials_;
  bool kernel_loaded_ = false;
  void decompress_kernel();
  std::shared_ptr<PackedKernel> packed_kernel_ = nullptr;
  std::shared_mutex packedkernel_mutex_;
};

}

#endif
