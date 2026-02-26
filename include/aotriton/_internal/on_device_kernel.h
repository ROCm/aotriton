// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_API_BINARY_KERNEL_H
#define AOTRITON_V3_API_BINARY_KERNEL_H

#include <filesystem>
#include <string_view>
#include <shared_mutex>
#include <unordered_map>
#include <aotriton/config.h>

using pstring_type = std::filesystem::path::string_type;
using pstring_view = std::basic_string_view<std::filesystem::path::value_type>;

namespace AOTRITON_NS {

class PackedKernel;

class OnDeviceKernel {
public:
  OnDeviceKernel() {
  }
  ~OnDeviceKernel();

  void clear_decompressed_image();
protected:
  bool kernel_loaded_ = false;

  hipFunction_t cfind_function(int device_id) const;
  std::tuple<hipFunction_t, hipError_t> load_for_device(int device_id,
                                                        std::string_view kernel_function_name,
                                                        std::string_view stem_name,
                                                        pstring_view package_path);
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
  void decompress_kernel(pstring_view package_path,
                         std::string_view stem_name);
  std::shared_ptr<PackedKernel> packed_kernel_ = nullptr;
  std::shared_mutex packedkernel_mutex_;
};

}

#endif // AOTRITON_V3_API_BINARY_KERNEL_H
